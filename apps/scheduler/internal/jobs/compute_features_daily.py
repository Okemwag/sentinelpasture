"""Specific feature-refresh job definition."""

from __future__ import annotations

from .plan import JobDefinition, compute_features_daily_job as definition


def compute_features_daily_job() -> JobDefinition:
    return definition()

