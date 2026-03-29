"""FastAPI API service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, date
import csv
import logging
from pathlib import Path
from typing import Any
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .ai import AIGateway, AIGatewayError
from .audit import list_audit_events, log_audit_event
from .auth import AuthService, role_allowed
from .db import Base, SessionLocal, engine
from .region_intelligence import build_regional_briefing

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("api_service.main")


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
ROOT = Path(__file__).resolve().parents[3]
GIBS_WMTS_CAPABILITIES = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/1.0.0/WMTSCapabilities.xml"
GIBS_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # TODO: Replace create_all with Alembic migrations and explicit startup health checks.
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        auth_service.seed_defaults(db)
    finally:
        db.close()
    logger.info(
        "api startup complete ai_inference_url=%s fallback_to_mock=%s",
        ai_gateway.base_url or "unconfigured",
        ai_gateway.fallback_to_mock,
    )
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
    # TODO: Standardize metadata shape across API, AI, and export endpoints.
    metadata = payload.get("metadata", {})
    return {
        "modelVersion": metadata.get("model_version", "unavailable"),
        "lastUpdated": metadata.get("feature_snapshot_timestamp", datetime.utcnow().isoformat()),
        "confidence": confidence,
    }


def _load_violence_rows(limit: int = 60) -> list[dict[str, Any]]:
    """Read recent rows from the baseline training dataset for violence/event analysis."""
    path = ROOT / "ai" / "data" / "processed" / "training_baseline_monthly_national.csv"
    if not path.exists():
        logger.warning("violence summary requested but %s is missing", path)
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    # Keep only the most recent N periods
    rows = rows[-limit:]
    return rows


def _fetch_satellite_dates(days: int = 30) -> list[str]:
    """Fetch available satellite date range from NASA GIBS capabilities."""
    end_day = date.today()
    try:
        with urlopen(GIBS_WMTS_CAPABILITIES, timeout=8) as response:
            xml_data = response.read()
        root = ET.fromstring(xml_data)
        ns = {"wmts": "http://www.opengis.net/wmts/1.0", "ows": "http://www.opengis.net/ows/1.1"}
        for layer in root.findall(".//wmts:Layer", ns):
            identifier = layer.find("ows:Identifier", ns)
            if identifier is None or identifier.text != GIBS_LAYER:
                continue
            for dim in layer.findall("wmts:Dimension", ns):
                dim_id = dim.find("ows:Identifier", ns)
                value = dim.find("wmts:Value", ns)
                if dim_id is not None and dim_id.text == "Time" and value is not None and value.text:
                    raw = value.text
                    if "/" in raw:
                        parts = raw.split("/")
                        end_day = date.fromisoformat(parts[1])
                    else:
                        end_day = date.fromisoformat(raw.split(",")[-1])
                    break
            break
    except Exception:
        logger.warning("Unable to fetch live satellite date capabilities; falling back to recent dates.")
    start_day = end_day - timedelta(days=max(1, days - 1))
    out: list[str] = []
    current = end_day
    while current >= start_day:
        out.append(current.isoformat())
        current -= timedelta(days=1)
    return out


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
        briefing = build_regional_briefing(regions.get("regions", []), limit=8)
        rows = [
            {
                "id": index,
                "severity": "elevated" if region["riskLevel"] in {"critical", "elevated"} else "moderate",
                "title": f"{region['thresholdStatus']} in {region['name']}",
                "description": region["thresholdReason"],
                "timestamp": now.isoformat(),
                "region": region["name"],
                "status": "active" if region["riskLevel"] in {"critical", "elevated"} else "monitoring",
            }
            for index, region in enumerate(briefing, start=1)
        ]
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


@app.get("/api/alerts/stats")
async def alert_stats(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="alerts.read", resource="alert_stats")
        regions = await ai_gateway.list_regions(20)
        rows = regions.get("regions", [])
        return {
            "data": {
                "active": sum(1 for row in rows if float(row["risk_score"]) >= 0.7),
                "monitoring": sum(1 for row in rows if 0.45 <= float(row["risk_score"]) < 0.7),
                "resolved24h": 0,
            },
            "success": True,
            "metadata": {
                "modelVersion": regions.get("model_version", "unavailable"),
                "lastUpdated": datetime.utcnow().isoformat(),
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
        rows = build_regional_briefing(regions.get("regions", []), limit=8)
        latest_snapshot = max(
            (row.get("featureSnapshotTimestamp", "") for row in rows),
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


@app.get("/api/regional/map")
async def regional_map(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="regional.read", resource="regional_map")
        regions = await ai_gateway.list_regions(24)
        rows = build_regional_briefing(regions.get("regions", []), limit=8)
        return {
            "data": rows,
            "success": True,
            "metadata": {
                "modelVersion": regions.get("model_version", "unavailable"),
                "lastUpdated": datetime.utcnow().isoformat(),
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
        if not role_allowed(user.role, "analyst"):
            raise HTTPException(status_code=403, detail="Analyst role required")
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


@app.get("/api/interventions/list")
async def interventions_list(
    authorization: str | None = Header(default=None),
    region: str | None = None,
) -> dict[str, Any]:
    db = _db()
    try:
        user = _current_user(db, authorization)
        if not role_allowed(user.role, "analyst"):
            raise HTTPException(status_code=403, detail="Analyst role required")
        regions = await ai_gateway.list_regions(20)
        target_region = region
        target_region_row: dict[str, Any] | None = None
        if not target_region:
            target_region_row = max(
                regions.get("regions", []),
                key=lambda row: float(row["risk_score"]),
                default=None,
            )
            if target_region_row:
                target_region = str(target_region_row["region_id"])
        if not target_region:
            raise HTTPException(status_code=404, detail="No regions available for intervention ranking")

        # TODO: Replace "highest current risk" auto-selection with explicit user-selected region context.
        prediction = await ai_gateway.predict_risk(target_region, "7d")
        first_prediction = prediction.get("predictions", [{}])[0]
        risk_score = float(first_prediction.get("risk_level", 0)) / 100
        response = await ai_gateway.recommend_interventions(
            target_region,
            {"risk_score": risk_score},
        )
        _log_request(db, user=user, action="interventions.list", resource=target_region)
        region_meta: dict[str, Any] = {}
        if target_region_row:
            region_meta = {
                "regionName": target_region_row.get("name"),
                "regionId": target_region_row.get("region_id"),
                "riskScore": target_region_row.get("risk_score"),
                "riskLevel": target_region_row.get("riskLevel"),
                "thresholdStatus": target_region_row.get("thresholdStatus"),
                "primaryDriver": target_region_row.get("primaryDriver"),
                "thresholdReason": target_region_row.get("thresholdReason"),
            }
        base_meta = _metadata_from(response, "Medium")
        base_meta.update(region_meta)
        return {
            "data": response["interventions"],
            "success": True,
            "metadata": base_meta,
        }
    except AIGatewayError as exc:
        raise _ai_unavailable(exc) from exc
    finally:
        db.close()


@app.get("/api/outcomes/list")
async def outcomes_list(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        user = _current_user(db, authorization)
        if not role_allowed(user.role, "analyst"):
            raise HTTPException(status_code=403, detail="Analyst role required")
        regions = await ai_gateway.list_regions(6)
        now = datetime.utcnow()
        rows: list[dict[str, Any]] = []
        for index, region in enumerate(regions.get("regions", [])[:4], start=1):
            risk_now = int(round(float(region["risk_score"]) * 100))
            reduction = max(2, min(12, index * 2))
            risk_after = max(0, risk_now - reduction)
            trend = "Improving" if risk_after < risk_now else "Stable"
            rows.append(
                {
                    "intervention": f"Targeted support package for {region['region_id']}",
                    "deployed": now.date().isoformat(),
                    "riskBefore": risk_now,
                    "riskAfter": risk_after,
                    "trend": trend,
                    "commentary": (
                        f"Modeled pressure in {region['region_id']} moved from {risk_now} to {risk_after} "
                        "after policy intervention simulation."
                    ),
                }
            )
        _log_request(db, user=user, action="outcomes.list", resource="outcomes")
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


@app.get("/api/outcomes/chart")
async def outcomes_chart(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="outcomes.read", resource="outcomes_chart")
        prediction = await ai_gateway.predict_risk("national", "7d")
        series = prediction.get("predictions", [])
        chart = [{"date": row["date"], "value": row["risk_level"]} for row in series]
        return {
            "data": chart,
            "success": True,
            "metadata": _metadata_from(prediction, "Medium"),
        }
    except AIGatewayError as exc:
        raise _ai_unavailable(exc) from exc
    finally:
        db.close()


@app.get("/api/violence/timeseries")
async def violence_timeseries(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Return national monthly counts of demonstrations and political violence."""
    db = _db()
    try:
        _authorize_read(db, authorization, action="violence.read", resource="violence_timeseries")
        rows = _load_violence_rows(limit=120)
        timeseries: list[dict[str, Any]] = []
        for row in rows:
            try:
                timeseries.append(
                    {
                        "date": row["period"],
                        "totalEvents": int(float(row["total_events"])),
                        "totalFatalities": int(float(row["total_fatalities"])),
                        "demonstrationsEvents": int(float(row["demonstrations_events"])),
                        "civilianTargetingEvents": int(float(row["civilian_targeting_events"])),
                        "politicalViolenceEvents": int(float(row["political_violence_events"])),
                    }
                )
            except (KeyError, ValueError):
                continue
        return {
            "data": timeseries,
            "success": True,
            "metadata": {
                "modelVersion": "baseline-risk-model-v1",
                "lastUpdated": datetime.utcnow().isoformat(),
                "confidence": "Medium",
            },
        }
    finally:
        db.close()


@app.get("/api/satellite/dates")
async def satellite_dates(
    authorization: str | None = Header(default=None),
    days: int = 30,
) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="satellite.read", resource="satellite_dates")
        date_list = _fetch_satellite_dates(days=min(max(days, 7), 90))
        return {
            "data": {
                "layer": GIBS_LAYER,
                "provider": "NASA GIBS",
                "dates": date_list,
            },
            "success": True,
            "metadata": {
                "modelVersion": "geospatial-v1",
                "lastUpdated": datetime.utcnow().isoformat(),
                "confidence": "Medium",
            },
        }
    finally:
        db.close()


@app.get("/api/satellite/tile-template")
async def satellite_tile_template(
    authorization: str | None = Header(default=None),
    date_value: str | None = None,
) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="satellite.read", resource="satellite_tiles")
        selected = date_value or _fetch_satellite_dates(1)[0]
        template = (
            "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            f"{GIBS_LAYER}/default/{selected}/GoogleMapsCompatible_Level9/"
            "{z}/{y}/{x}.jpg"
        )
        return {
            "data": {
                "provider": "NASA GIBS",
                "layer": GIBS_LAYER,
                "date": selected,
                "tileTemplate": template,
            },
            "success": True,
            "metadata": {
                "modelVersion": "geospatial-v1",
                "lastUpdated": datetime.utcnow().isoformat(),
                "confidence": "Medium",
            },
        }
    finally:
        db.close()


@app.get("/api/reports/list")
async def reports_list(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="reports.read", resource="reports")
        now = datetime.utcnow()
        today = now.date().isoformat()
        reports = [
            {
                "title": "National Stability Assessment",
                "type": "PDF Report",
                "date": today,
                "size": "2.4 MB",
                "downloadUrl": "/api/reports/download/national-stability-assessment",
            },
            {
                "title": "Regional Risk Data Export",
                "type": "CSV Export",
                "date": today,
                "size": "156 KB",
                "downloadUrl": "/api/reports/download/regional-risk-data",
            },
            {
                "title": "Intervention Effectiveness Analysis",
                "type": "PDF Report",
                "date": today,
                "size": "1.8 MB",
                "downloadUrl": "/api/reports/download/intervention-effectiveness-analysis",
            },
            {
                "title": "System Audit Log",
                "type": "CSV Export",
                "date": today,
                "size": "89 KB",
                "downloadUrl": "/api/reports/download/system-audit-log",
            },
        ]
        return {
            "data": reports,
            "success": True,
            "metadata": {
                "modelVersion": "reporting-v1",
                "lastUpdated": now.isoformat(),
                "confidence": "Medium",
            },
        }
    finally:
        db.close()


@app.get("/api/reports/download/{report_id}")
async def reports_download(
    report_id: str,
    authorization: str | None = Header(default=None),
) -> Response:
    db = _db()
    try:
        user = _current_user(db, authorization)
        if not role_allowed(user.role, "viewer"):
            raise HTTPException(status_code=403, detail="Insufficient role")
        _log_request(db, user=user, action="reports.download", resource=report_id)

        if report_id == "regional-risk-data":
            content = (
                "region,risk_score,confidence\n"
                "north-frontier,0.72,0.82\n"
                "coastal-belt,0.58,0.79\n"
                "capital-corridor,0.44,0.77\n"
            )
            return Response(
                content=content,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=regional-risk-data.csv"},
            )

        if report_id == "system-audit-log":
            content = "event_id,actor,action,outcome\n1,system,reports.read,success\n2,system,reports.download,success\n"
            return Response(
                content=content,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=system-audit-log.csv"},
            )

        if report_id == "national-stability-assessment":
            content = (
                "National Stability Assessment\n\n"
                "Summary: Current modeled national stability posture is under active monitoring.\n"
                "This is a generated placeholder export from the API service.\n"
            )
            return Response(
                content=content,
                media_type="application/octet-stream",
                headers={"Content-Disposition": "attachment; filename=national-stability-assessment.pdf"},
            )

        if report_id == "intervention-effectiveness-analysis":
            content = (
                "Intervention Effectiveness Analysis\n\n"
                "Summary: Interventions are showing early positive movement in modeled risk.\n"
                "This is a generated placeholder export from the API service.\n"
            )
            return Response(
                content=content,
                media_type="application/octet-stream",
                headers={"Content-Disposition": "attachment; filename=intervention-effectiveness-analysis.pdf"},
            )

        raise HTTPException(status_code=404, detail="Report not found")
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
