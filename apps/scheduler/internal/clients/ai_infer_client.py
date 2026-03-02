"""HTTP client stub for AI inference calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AIInferClient:
    base_url: str = "http://localhost:8100"

