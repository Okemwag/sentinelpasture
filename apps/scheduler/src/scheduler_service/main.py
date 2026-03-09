"""FastAPI scheduler service."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .jobs import default_plan

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from internal.clients.ai_infer_client import AIInferClient
from internal.clients.ai_train_client import AITrainClient
from internal.clients.feature_client import FeatureClient
from internal.clients.ingestion_client import IngestionClient
from internal.observability.logger import get_logger


app = FastAPI(
    title="Governance Intel Scheduler",
    version="0.1.0",
    description="Scheduler and orchestration service for governance-intel platform.",
)
logger = get_logger("scheduler.service")

INGESTION_BASE_URL = os.getenv("INGESTION_BASE_URL", "http://localhost:8300")
AI_INFER_BASE_URL = os.getenv("AI_INFER_BASE_URL", "http://localhost:8100")

ingestion_client = IngestionClient(base_url=INGESTION_BASE_URL)
feature_client = FeatureClient(base_url=INGESTION_BASE_URL)
train_client = AITrainClient(base_url=INGESTION_BASE_URL)
infer_client = AIInferClient(base_url=AI_INFER_BASE_URL)
logger.info(
    "scheduler service configured ingestion_base_url=%s ai_infer_base_url=%s",
    INGESTION_BASE_URL,
    AI_INFER_BASE_URL,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "scheduler"}


@app.get("/jobs")
async def jobs() -> dict[str, object]:
    plan = default_plan()
    return {"count": len(plan), "jobs": plan}


@app.post("/jobs/run/{job_name}")
async def run_job(job_name: str) -> dict[str, object]:
    started = datetime.utcnow().isoformat()
    logger.info("job run requested name=%s", job_name)
    try:
        if job_name == "ingest_daily":
            result = ingestion_client.run_pipeline(mode="full", train=True)
        elif job_name == "compute_features_daily":
            result = feature_client.refresh_features()
        elif job_name == "run_scoring_daily":
            regions = infer_client.list_regions(limit=25)
            result = {
                "status": "completed",
                "region_count": len(regions.get("regions", [])),
                "model_version": regions.get("model_version", "unknown"),
            }
        elif job_name == "retrain_monthly":
            result = train_client.retrain()
        elif job_name == "data_quality_reports":
            quality = ingestion_client.run_pipeline(mode="features", train=False)
            regions = infer_client.list_regions(limit=100)
            result = {
                "status": "completed",
                "quality_run": quality.get("status", "unknown"),
                "regions_scored": len(regions.get("regions", [])),
            }
        else:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_name}")

        finished = datetime.utcnow().isoformat()
        logger.info("job run completed name=%s", job_name)
        return {"job": job_name, "started_at": started, "finished_at": finished, "result": result}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("job run failed name=%s", job_name)
        raise HTTPException(status_code=502, detail=f"Job execution failed: {exc}") from exc
