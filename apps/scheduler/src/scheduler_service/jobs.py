"""Registered scheduler jobs."""

from __future__ import annotations


def default_plan() -> list[dict[str, str]]:
    return [
        {"name": "ingest_daily", "schedule": "0 2 * * *", "description": "Trigger daily ingestion runs."},
        {"name": "compute_features_daily", "schedule": "30 2 * * *", "description": "Refresh feature tables."},
        {"name": "run_scoring_daily", "schedule": "0 3 * * *", "description": "Run daily scoring."},
        {"name": "retrain_monthly", "schedule": "0 4 1 * *", "description": "Launch monthly retraining."},
        {"name": "data_quality_reports", "schedule": "15 4 * * 1", "description": "Produce data quality reports."},
    ]

