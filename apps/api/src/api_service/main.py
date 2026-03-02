"""FastAPI API service."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from .ai import AIGateway
from .auth import AuthService, role_allowed
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


app = FastAPI(title="National Risk Intelligence API", version="1.0.0", description="FastAPI API service for governance-intel platform.")

auth_service = AuthService()
dashboard_service = DashboardService()
ai_gateway = AIGateway()


def _current_user(authorization: str | None):
    user = auth_service.user_from_auth_header(authorization)
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
    response = auth_service.login(request.username, request.password)
    if not response:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return response


@app.get("/auth/me")
async def auth_me(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    user = _current_user(authorization)
    return {"username": user.username, "full_name": user.full_name, "role": user.role}


@app.get("/api/status")
async def api_status(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    user = _current_user(authorization)
    if not role_allowed(user.role, "viewer"):
        raise HTTPException(status_code=403, detail="Insufficient role")
    return {
        "status": "operational",
        "active_assessments": 3,
        "total_signals": 24,
        "total_assessments": 128,
        "database_status": "not_configured",
        "ai_engine_status": ai_gateway.engine_name,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/admin/policy")
async def admin_policy(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    user = _current_user(authorization)
    if not role_allowed(user.role, "admin"):
        raise HTTPException(status_code=403, detail="Admin role required")
    return {
        "policy_pack": "policy-pack-v0.1",
        "restraint_rules": ["proportional response", "civilian protection", "minimum force posture"],
        "updated_by": user.username,
    }


@app.get("/api/dashboard/stability")
async def dashboard_stability() -> dict[str, Any]:
    return dashboard_service.stability()


@app.get("/api/dashboard/stats")
async def dashboard_stats() -> dict[str, Any]:
    return dashboard_service.stats()


@app.get("/api/drivers/list")
async def driver_list() -> dict[str, Any]:
    return dashboard_service.drivers()


@app.get("/api/alerts/list")
async def alert_list() -> dict[str, Any]:
    return dashboard_service.alerts()


@app.get("/api/regional/data")
async def regional_data() -> dict[str, Any]:
    return dashboard_service.regional_data()


@app.post("/api/process")
async def process_signals(request: ProcessSignalsRequest) -> dict[str, Any]:
    return await ai_gateway.process_signals(request.signals)


@app.get("/api/predict")
async def predict_risk(region: str = "national", timeframe: str = "7d") -> dict[str, Any]:
    return await ai_gateway.predict_risk(region, timeframe)


@app.post("/api/analyze-drivers")
async def analyze_drivers(request: AnalyzeDriversRequest) -> dict[str, Any]:
    return await ai_gateway.analyze_drivers(request.data)


@app.post("/api/recommend-interventions")
async def recommend_interventions(request: RecommendInterventionsRequest) -> dict[str, Any]:
    return await ai_gateway.recommend_interventions(request.region, request.riskProfile)

