"""FastAPI inference service aligned to the target monorepo contract."""

from datetime import datetime

from fastapi import FastAPI

from .schemas import (
    Driver,
    ExplainRequest,
    ExplainResponse,
    InterventionOption,
    InterventionRequest,
    InterventionResponse,
    RiskRequest,
    RiskResponse,
)

MODEL_VERSION = "policy-pack-v0.1"

app = FastAPI(
    title="Governance Intel AI Inference",
    version="0.1.0",
    description="Contract-aligned inference service for risk scoring and interventions.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "governance-intel-ai"}


@app.post("/infer/risk", response_model=RiskResponse)
async def infer_risk(request: RiskRequest) -> RiskResponse:
    score = min(0.92, 0.45 + (len(request.signals) * 0.05))
    timestamp = request.at_time or datetime.utcnow()
    drivers = [
        Driver(name="Market stress", contribution=0.34, direction="up"),
        Driver(name="Mobility anomaly", contribution=0.21, direction="up"),
        Driver(name="Service disruption", contribution=0.16, direction="up"),
    ]

    return RiskResponse(
        region_id=request.region_id,
        risk_score=round(score, 2),
        confidence_band="medium" if score < 0.7 else "high",
        confidence=0.82,
        top_drivers=drivers,
        known_data_gaps=["clinic load incomplete", "rural mobility lagging 24h"],
        model_version=MODEL_VERSION,
        feature_snapshot_timestamp=timestamp,
    )


@app.post("/infer/explain", response_model=ExplainResponse)
async def infer_explain(request: ExplainRequest) -> ExplainResponse:
    return ExplainResponse(
        region_id=request.region_id,
        summary="Risk is elevated due to converging economic and mobility stressors.",
        top_drivers=[
            Driver(name="Economic stress", contribution=0.31, direction="up"),
            Driver(name="Cross-corridor movement shift", contribution=0.22, direction="up"),
        ],
        uncertainty_notes=[
            "Education attendance data is two days stale.",
            "One market feed is operating with fallback estimates.",
        ],
        model_version=MODEL_VERSION,
        feature_snapshot_timestamp=datetime.utcnow(),
    )


@app.post("/infer/interventions", response_model=InterventionResponse)
async def infer_interventions(request: InterventionRequest) -> InterventionResponse:
    severity = "High" if request.risk_score >= 0.7 else "Moderate"
    return InterventionResponse(
        region_id=request.region_id,
        interventions=[
            InterventionOption(
                category="Targeted mediation deployment",
                expected_impact=severity,
                time_to_effect="Short",
                confidence="High",
                constraints_applied=["proportional response", "civilian protection"],
            ),
            InterventionOption(
                category="Market-day security hardening",
                expected_impact="Moderate",
                time_to_effect="Short",
                confidence="Medium",
                constraints_applied=["minimum force posture", "county coordination"],
            ),
        ],
        model_version=MODEL_VERSION,
        feature_snapshot_timestamp=datetime.utcnow(),
    )

