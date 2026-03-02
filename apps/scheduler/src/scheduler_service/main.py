"""FastAPI scheduler service."""

from __future__ import annotations

from fastapi import FastAPI

from .jobs import default_plan


app = FastAPI(title="Governance Intel Scheduler", version="0.1.0", description="Scheduler and orchestration service for governance-intel platform.")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "scheduler"}


@app.get("/jobs")
async def jobs() -> dict[str, object]:
    plan = default_plan()
    return {"count": len(plan), "jobs": plan}

