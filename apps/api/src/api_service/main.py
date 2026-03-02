"""FastAPI API service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .ai import AIGateway, AIGatewayError
from .audit import list_audit_events, log_audit_event
from .auth import AuthService, role_allowed
from .db import Base, SessionLocal, engine


class ProcessSignalsRequest(BaseModel):
    signals: list[dict[str, Any]]


class AnalyzeDriversRequest(BaseModel):
    data: dict[str, Any]


class RecommendInterventionsRequest(BaseModel):
    region: str
    riskProfile: dict[str, Any]


class LoginRequest(BaseModel):
    username: str
    password: str


auth_service = AuthService()
ai_gateway = AIGateway()


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        auth_service.seed_defaults(db)
    finally:
        db.close()
    yield


app = FastAPI(
    title="National Risk Intelligence API",
    version="1.0.0",
    description="FastAPI API service for governance-intel platform.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _db() -> Session:
    return SessionLocal()


def _log_request(
    db: Session,
    *,
    user: Any,
    action: str,
    resource: str,
    outcome: str = "success",
    detail: dict[str, Any] | None = None,
) -> None:
    log_audit_event(
        db,
        actor_username=user.username,
        actor_role=user.role,
        action=action,
        resource=resource,
        outcome=outcome,
        detail=detail,
    )


def _current_user(db: Session, authorization: str | None):
    user = auth_service.user_from_auth_header(db, authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
    return user


def _ai_unavailable(exc: AIGatewayError) -> HTTPException:
    return HTTPException(status_code=503, detail=f"AI service unavailable: {exc}")


def _metadata_from(payload: dict[str, Any], confidence: str) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    return {
        "modelVersion": metadata.get("model_version", "unavailable"),
        "lastUpdated": metadata.get("feature_snapshot_timestamp", datetime.utcnow().isoformat()),
        "confidence": confidence,
    }


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "status": "operational",
        "service": "National Risk Intelligence API",
        "version": "1.0.0",
        "ai_engine_status": ai_gateway.engine_name,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "ai_engine": ai_gateway.engine_name,
        "ai_inference_url_configured": ai_gateway.remote_enabled,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/auth/login")
async def login(request: LoginRequest) -> dict[str, Any]:
    db = _db()
    try:
        response = auth_service.login(db, request.username, request.password)
        if not response:
            log_audit_event(
                db,
                actor_username=request.username,
                actor_role="anonymous",
                action="auth.login",
                resource="session",
                outcome="denied",
                detail={"reason": "invalid_credentials"},
            )
            raise HTTPException(status_code=401, detail="Invalid credentials")
        user = response["user"]
        log_audit_event(
            db,
            actor_username=user["username"],
            actor_role=user["role"],
            action="auth.login",
            resource="session",
            outcome="success",
        )
        return response
    finally:
        db.close()


@app.get("/auth/me")
async def auth_me(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        user = _current_user(db, authorization)
        _log_request(db, user=user, action="auth.me", resource="user")
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        }
    finally:
        db.close()


@app.get("/api/status")
async def api_status(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        user = _current_user(db, authorization)
        if not role_allowed(user.role, "viewer"):
            raise HTTPException(status_code=403, detail="Insufficient role")
        _log_request(db, user=user, action="status.read", resource="api_status")
        regions = await ai_gateway.list_regions(100)
        active_assessments = sum(1 for row in regions.get("regions", []) if float(row["risk_score"]) >= 0.55)
        total_assessments = len(list_audit_events(db, limit=200))
        return {
            "status": "operational",
            "active_assessments": active_assessments,
            "total_signals": int(regions.get("feature_count", 0)),
            "total_assessments": total_assessments,
            "database_status": "connected",
            "ai_engine_status": ai_gateway.engine_name,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except AIGatewayError as exc:
        raise _ai_unavailable(exc) from exc
    finally:
        db.close()


@app.get("/api/admin/policy")
async def admin_policy(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        user = _current_user(db, authorization)
        if not role_allowed(user.role, "admin"):
            raise HTTPException(status_code=403, detail="Admin role required")
        _log_request(db, user=user, action="policy.read", resource="policy_pack")
        return {
            "policy_pack": "policy-pack-v0.1",
            "restraint_rules": ["proportional response", "civilian protection", "minimum force posture"],
            "updated_by": user.username,
        }
    finally:
        db.close()


def _authorize_read(
    db: Session,
    authorization: str | None,
    *,
    action: str,
    resource: str,
):
    user = _current_user(db, authorization)
    if not role_allowed(user.role, "viewer"):
        raise HTTPException(status_code=403, detail="Insufficient role")
    _log_request(db, user=user, action=action, resource=resource)
    return user


@app.get("/api/dashboard/stability")
async def dashboard_stability(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="dashboard.read", resource="stability")
        prediction = await ai_gateway.predict_risk("national", "7d")
        series = prediction["predictions"]
        current = series[0]["risk_level"] if series else 0
        previous = series[-1]["risk_level"] if len(series) > 1 else current
        change = round(current - previous, 2)
        trend = "up" if change > 0 else "down" if change < 0 else "stable"
        return {
            "data": {
                "value": max(0, 100 - current),
                "trend": trend,
                "confidence": "High" if float(prediction["confidence"]) >= 0.8 else "Medium",
                "change": f"{change:+.1f} risk points",
            },
            "success": True,
            "metadata": _metadata_from(
                prediction,
                "High" if float(prediction["confidence"]) >= 0.8 else "Medium",
            ),
        }
    except AIGatewayError as exc:
        raise _ai_unavailable(exc) from exc
    finally:
        db.close()


@app.get("/api/dashboard/stats")
async def dashboard_stats(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="dashboard.read", resource="stats")
        regions = await ai_gateway.list_regions(100)
        region_rows = regions.get("regions", [])
        active_alerts = sum(1 for row in region_rows if float(row["risk_score"]) >= 0.7)
        latest_snapshot = max(
            (row.get("feature_snapshot_timestamp", "") for row in region_rows),
            default=datetime.utcnow().isoformat(),
        )
        return {
            "data": {
                "activeAlerts": active_alerts,
                "regionsMonitored": len(region_rows),
                "dataSources": int(regions.get("feature_count", 0)),
                "lastUpdate": latest_snapshot,
            },
            "success": True,
            "metadata": {
                "modelVersion": regions.get("model_version", "unavailable"),
                "lastUpdated": latest_snapshot,
                "confidence": "Medium",
            },
        }
    except AIGatewayError as exc:
        raise _ai_unavailable(exc) from exc
    finally:
        db.close()


@app.get("/api/drivers/list")
async def driver_list(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="drivers.read", resource="drivers")
        analysis = await ai_gateway.analyze_drivers({"region_id": "national"})
        return {
            "data": [
                {
                    "label": row["name"],
                    "percentage": round(float(row["contribution"]) * 100, 2),
                    "trend": "up" if row["trend"] == "increasing" else "down",
                    "confidence": "High" if float(row["confidence"]) >= 0.8 else "Medium",
                }
                for row in analysis["drivers"]
            ],
            "success": True,
            "metadata": _metadata_from(analysis, "Medium"),
        }
    except AIGatewayError as exc:
        raise _ai_unavailable(exc) from exc
    finally:
        db.close()


@app.get("/api/alerts/list")
async def alert_list(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="alerts.read", resource="alerts")
        regions = await ai_gateway.list_regions(20)
        now = datetime.utcnow()
        rows = []
        for index, region in enumerate(regions.get("regions", [])[:10], start=1):
            risk_score = float(region["risk_score"])
            rows.append(
                {
                    "id": index,
                    "severity": "elevated" if risk_score >= 0.7 else "moderate",
                    "title": f"Elevated pressure in {region['region_id']}",
                    "description": f"Primary driver: {region['primary_driver']}. Risk score {risk_score:.2f}.",
                    "timestamp": now.isoformat(),
                    "region": region["region_id"],
                    "status": "active" if risk_score >= 0.55 else "monitoring",
                }
            )
        return {
            "data": rows,
            "success": True,
            "metadata": {
                "modelVersion": regions.get("model_version", "unavailable"),
                "lastUpdated": now.isoformat(),
                "confidence": "Medium",
            },
        }
    except AIGatewayError as exc:
        raise _ai_unavailable(exc) from exc
    finally:
        db.close()


@app.get("/api/regional/data")
async def regional_data(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="regional.read", resource="regional_data")
        regions = await ai_gateway.list_regions(100)
        rows = [
            {
                "region": row["region_id"],
                "population": "N/A",
                "stabilityIndex": max(0, int(round(100 - (float(row["risk_score"]) * 100)))),
                "trend": "Elevated" if float(row["risk_score"]) >= 0.7 else "Monitoring",
                "primaryDriver": row["primary_driver"],
                "confidence": "High" if float(row["confidence"]) >= 0.8 else "Medium",
            }
            for row in regions.get("regions", [])
        ]
        latest_snapshot = max(
            (row.get("feature_snapshot_timestamp", "") for row in regions.get("regions", [])),
            default=datetime.utcnow().isoformat(),
        )
        return {
            "data": rows,
            "success": True,
            "metadata": {
                "modelVersion": regions.get("model_version", "unavailable"),
                "lastUpdated": latest_snapshot,
                "confidence": "Medium",
            },
        }
    except AIGatewayError as exc:
        raise _ai_unavailable(exc) from exc
    finally:
        db.close()


@app.post("/api/process")
async def process_signals(
    request: ProcessSignalsRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    db = _db()
    try:
        user = _current_user(db, authorization)
        if not role_allowed(user.role, "analyst"):
            raise HTTPException(status_code=403, detail="Analyst role required")
        _log_request(
            db,
            user=user,
            action="signals.process",
            resource="ai_process",
            detail={"signal_count": len(request.signals)},
        )
        return await ai_gateway.process_signals(request.signals)
    except AIGatewayError as exc:
        raise _ai_unavailable(exc) from exc
    finally:
        db.close()


@app.get("/api/predict")
async def predict_risk(
    region: str = "national",
    timeframe: str = "7d",
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    db = _db()
    try:
        user = _current_user(db, authorization)
        if not role_allowed(user.role, "viewer"):
            raise HTTPException(status_code=403, detail="Insufficient role")
        _log_request(db, user=user, action="risk.predict", resource=region, detail={"timeframe": timeframe})
        return await ai_gateway.predict_risk(region, timeframe)
    except AIGatewayError as exc:
        raise _ai_unavailable(exc) from exc
    finally:
        db.close()


@app.post("/api/analyze-drivers")
async def analyze_drivers(
    request: AnalyzeDriversRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    db = _db()
    try:
        user = _current_user(db, authorization)
        if not role_allowed(user.role, "analyst"):
            raise HTTPException(status_code=403, detail="Analyst role required")
        _log_request(db, user=user, action="drivers.analyze", resource="driver_analysis")
        return await ai_gateway.analyze_drivers(request.data)
    except AIGatewayError as exc:
        raise _ai_unavailable(exc) from exc
    finally:
        db.close()


@app.post("/api/recommend-interventions")
async def recommend_interventions(
    request: RecommendInterventionsRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    db = _db()
    try:
        user = _current_user(db, authorization)
        if not role_allowed(user.role, "operator"):
            raise HTTPException(status_code=403, detail="Operator role required")
        _log_request(
            db,
            user=user,
            action="interventions.recommend",
            resource=request.region,
        )
        return await ai_gateway.recommend_interventions(request.region, request.riskProfile)
    except AIGatewayError as exc:
        raise _ai_unavailable(exc) from exc
    finally:
        db.close()


@app.get("/api/audit")
async def audit_log(
    authorization: str | None = Header(default=None),
    limit: int = 100,
) -> dict[str, Any]:
    db = _db()
    try:
        user = _current_user(db, authorization)
        if not role_allowed(user.role, "admin"):
            raise HTTPException(status_code=403, detail="Admin role required")
        _log_request(db, user=user, action="audit.read", resource="audit_events", detail={"limit": limit})
        records = list_audit_events(db, limit=min(max(limit, 1), 200))
        return {
            "data": [
                {
                    "id": record.id,
                    "actor_username": record.actor_username,
                    "actor_role": record.actor_role,
                    "action": record.action,
                    "resource": record.resource,
                    "outcome": record.outcome,
                    "detail_json": record.detail_json,
                    "created_at": record.created_at.isoformat(),
                }
                for record in records
            ],
            "success": True,
        }
    finally:
        db.close()
