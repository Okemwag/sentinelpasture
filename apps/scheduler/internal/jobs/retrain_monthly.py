"""Specific retraining job definition."""

from __future__ import annotations

from .plan import JobDefinition, retrain_monthly_job as definition


def retrain_monthly_job() -> JobDefinition:
    return definition()

