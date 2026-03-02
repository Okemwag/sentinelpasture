"""Specific scoring job definition."""

from __future__ import annotations

from .plan import JobDefinition, run_scoring_daily_job as definition


def run_scoring_daily_job() -> JobDefinition:
    return definition()

