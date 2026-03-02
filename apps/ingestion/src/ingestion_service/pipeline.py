"""Ingestion pipeline description."""

from __future__ import annotations

from .config import Settings


class Orchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def describe(self) -> dict:
        return {
            "environment": self.settings.environment,
            "collectors": [
                "acled",
                "chirps",
                "ndvi_modis",
                "market_prices",
                "corridors",
                "manual_upload",
            ],
            "stages": [
                "collect",
                "validate",
                "deduplicate",
                "normalize",
                "persist_raw",
                "persist_processed",
            ],
        }

