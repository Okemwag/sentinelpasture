"""Specific ingest-daily job definition."""

from __future__ import annotations

from .plan import JobDefinition, ingest_daily_job as definition


def ingest_daily_job() -> JobDefinition:
    return definition()

