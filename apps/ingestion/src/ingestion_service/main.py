"""FastAPI ingestion service."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from .config import load_settings
from .pipeline import Orchestrator


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = load_settings()
orchestrator = Orchestrator(settings)
logging.getLogger("ingestion.main").info(
    "ingestion service configured raw=%s processed=%s artifacts=%s synthetic_bootstrap_enabled=%s force=%s",
    settings.ai_data_raw_path,
    settings.ai_data_processed_path,
    settings.ai_data_artifacts_path,
    settings.synthetic_bootstrap_enabled,
    settings.synthetic_bootstrap_force,
)

app = FastAPI(title="Governance Intel Ingestion", version="0.1.0", description="Ingestion and normalization service for governance-intel platform.")


class RunPipelineRequest(BaseModel):
    mode: str = "full"
    train: bool = True


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "ingestion"}


@app.get("/pipeline")
async def pipeline() -> dict:
    return orchestrator.describe()


@app.post("/pipeline/run")
async def run_pipeline(request: RunPipelineRequest) -> dict:
    return orchestrator.run(mode=request.mode, train=request.train)
