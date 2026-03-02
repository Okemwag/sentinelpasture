"""Validation helpers for ingestion records."""

from __future__ import annotations


def validate_required_fields(record: dict, *required: str) -> dict:
    missing = [field for field in required if not record.get(field)]
    return {"valid": not missing, "missing": missing}

