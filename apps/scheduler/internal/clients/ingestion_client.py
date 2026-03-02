"""HTTP client stub for the ingestion service."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionClient:
    base_url: str = "http://localhost:8300"

