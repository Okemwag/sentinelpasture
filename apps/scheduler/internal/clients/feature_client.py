"""HTTP client stub for feature refresh orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureClient:
    base_url: str = "http://localhost:8300"

