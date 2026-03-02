"""HTTP client stub for AI retraining calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AITrainClient:
    base_url: str = "http://localhost:8100"

