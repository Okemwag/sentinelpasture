"""Shared job definitions for the scheduler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JobDefinition:
    name: str
    schedule: str
    description: str


def default_plan() -> list[JobDefinition]:
    return [
        ingest_daily_job(),
        compute_features_daily_job(),
        run_scoring_daily_job(),
        retrain_monthly_job(),
        data_quality_reports_job(),
    ]


def ingest_daily_job() -> JobDefinition:
    return JobDefinition("ingest_daily", "0 2 * * *", "Trigger daily ingestion runs.")


def compute_features_daily_job() -> JobDefinition:
    return JobDefinition("compute_features_daily", "30 2 * * *", "Refresh feature tables.")


def run_scoring_daily_job() -> JobDefinition:
    return JobDefinition("run_scoring_daily", "0 3 * * *", "Run daily scoring.")


def retrain_monthly_job() -> JobDefinition:
    return JobDefinition("retrain_monthly", "0 4 1 * *", "Launch monthly retraining.")


def data_quality_reports_job() -> JobDefinition:
    return JobDefinition("data_quality_reports", "15 4 * * 1", "Produce data quality reports.")

