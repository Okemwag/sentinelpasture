"""FastAPI API service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .ai import AIGateway
from .audit import list_audit_events, log_audit_event
from .auth import AuthService, role_allowed
from .db import Base, SessionLocal, engine
from .dashboard import DashboardService


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
dashboard_service = DashboardService()
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
        return {
            "status": "operational",
            "active_assessments": 3,
            "total_signals": 24,
            "total_assessments": 128,
            "database_status": "connected",
            "ai_engine_status": ai_gateway.engine_name,
            "timestamp": datetime.utcnow().isoformat(),
        }
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
        return dashboard_service.stability()
    finally:
        db.close()


@app.get("/api/dashboard/stats")
async def dashboard_stats(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="dashboard.read", resource="stats")
        return dashboard_service.stats()
    finally:
        db.close()


@app.get("/api/drivers/list")
async def driver_list(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="drivers.read", resource="drivers")
        return dashboard_service.drivers()
    finally:
        db.close()


@app.get("/api/alerts/list")
async def alert_list(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="alerts.read", resource="alerts")
        return dashboard_service.alerts()
    finally:
        db.close()


@app.get("/api/regional/data")
async def regional_data(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    db = _db()
    try:
        _authorize_read(db, authorization, action="regional.read", resource="regional_data")
        return dashboard_service.regional_data()
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
