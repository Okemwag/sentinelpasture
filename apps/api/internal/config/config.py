"""Configuration loader for the API service."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    ai_inference_url: str = ""


def load() -> Settings:
    return Settings(
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        ai_inference_url=os.getenv("AI_INFERENCE_URL", ""),
    )
