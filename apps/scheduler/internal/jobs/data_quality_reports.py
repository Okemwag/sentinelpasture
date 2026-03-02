"""Specific data-quality reporting job definition."""

from __future__ import annotations

from .plan import JobDefinition, data_quality_reports_job as definition


def data_quality_reports_job() -> JobDefinition:
    return definition()

