"""FastAPI inference service aligned to the target monorepo contract."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ai.bootstrap.demo_data import ensure_demo_assets
from ai.common.logging import configure_logging

from .schemas import (
    ExplainRequest,
    ExplainResponse,
    InterventionRequest,
    InterventionResponse,
    RiskRequest,
    RiskResponse,
)
from ..runtime.feature_fetcher import list_available_regions, load_feature_snapshot
from ..runtime.model_loader import load_baseline_model
from ..runtime.response_builder import build_explanation, build_interventions, score_snapshot, ScoredSnapshot


logger = configure_logging("ai.serving.api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    summary = ensure_demo_assets()
    logger.info("ai inference startup complete summary=%s", summary.as_dict())
    yield


app = FastAPI(
    title="Governance Intel AI Inference",
    version="0.1.0",
    description="Contract-aligned inference service for risk scoring and interventions.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "governance-intel-ai"}


@app.post("/infer/risk", response_model=RiskResponse)
async def infer_risk(request: RiskRequest) -> RiskResponse:
    logger.info("infer_risk request", extra={"region_id": request.region_id, "signal_count": len(request.signals)})
    model, scored = _scored_snapshot(request.region_id)
    signal_adjustment = min(len(request.signals) * 0.01, 0.08)
    adjusted_score = min(0.99, round(scored.risk_score + signal_adjustment, 4))

    return RiskResponse(
        region_id=scored.snapshot.region_id,
        risk_score=adjusted_score,
        confidence_band=scored.confidence_band,
        confidence=scored.confidence,
        top_drivers=scored.drivers,
        known_data_gaps=scored.known_data_gaps,
        model_version=model.model_version,
        feature_snapshot_timestamp=scored.snapshot.observed_at,
    )


@app.post("/infer/explain", response_model=ExplainResponse)
async def infer_explain(request: ExplainRequest) -> ExplainResponse:
    logger.info("infer_explain request", extra={"region_id": request.region_id})
    model, scored = _scored_snapshot(request.region_id)
    summary, notes = build_explanation(scored)
    return ExplainResponse(
        region_id=scored.snapshot.region_id,
        summary=summary,
        top_drivers=scored.drivers,
        uncertainty_notes=notes,
        model_version=model.model_version,
        feature_snapshot_timestamp=scored.snapshot.observed_at,
    )


@app.post("/infer/interventions", response_model=InterventionResponse)
async def infer_interventions(request: InterventionRequest) -> InterventionResponse:
    logger.info("infer_interventions request", extra={"region_id": request.region_id, "risk_score": request.risk_score})
    model, scored = _scored_snapshot(request.region_id)
    if request.risk_score > scored.risk_score:
        scored = ScoredSnapshot(
            snapshot=scored.snapshot,
            risk_score=request.risk_score,
            confidence=scored.confidence,
            confidence_band=scored.confidence_band,
            drivers=scored.drivers,
            known_data_gaps=scored.known_data_gaps,
        )
    return InterventionResponse(
        region_id=scored.snapshot.region_id,
        interventions=build_interventions(scored),
        model_version=model.model_version,
        feature_snapshot_timestamp=scored.snapshot.observed_at,
    )


@app.get("/infer/regions")
async def infer_regions(limit: int = 50) -> dict[str, object]:
    logger.info("infer_regions request", extra={"limit": limit})
    try:
        model = load_baseline_model()
        snapshots = list_available_regions(limit=min(max(limit, 1), 200))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rows = []
    for snapshot in snapshots:
        scored = score_snapshot(model, snapshot)
        rows.append(
            {
                "region_id": snapshot.region_id,
                "risk_score": scored.risk_score,
                "confidence": scored.confidence,
                "confidence_band": scored.confidence_band,
                "primary_driver": scored.drivers[0].name if scored.drivers else "Unknown",
                "feature_snapshot_timestamp": snapshot.observed_at.isoformat(),
            }
        )

    rows.sort(key=lambda item: item["risk_score"], reverse=True)
    return {
        "regions": rows,
        "model_version": model.model_version,
        "feature_count": len(model.feature_names),
        "training_rows": model.training_rows,
    }


def _scored_snapshot(region_id: str):
    try:
        model = load_baseline_model()
        snapshot = load_feature_snapshot(region_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return model, score_snapshot(model, snapshot)
