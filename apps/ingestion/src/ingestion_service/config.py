"""Configuration for the ingestion service."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    environment: str = "development"


def load_settings() -> Settings:
    return Settings(environment=os.getenv("INGESTION_ENV", "development"))

