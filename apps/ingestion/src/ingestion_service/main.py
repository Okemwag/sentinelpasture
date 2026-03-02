"""FastAPI ingestion service."""

from __future__ import annotations

from fastapi import FastAPI

from .config import load_settings
from .pipeline import Orchestrator


settings = load_settings()
orchestrator = Orchestrator(settings)

app = FastAPI(title="Governance Intel Ingestion", version="0.1.0", description="Ingestion and normalization service for governance-intel platform.")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "ingestion"}


@app.get("/pipeline")
async def pipeline() -> dict:
    return orchestrator.describe()

