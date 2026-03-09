"""Configuration for the ingestion service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    repo_root: Path = Path(".")
    ai_src_path: Path = Path(".")
    ai_data_raw_path: Path = Path(".")
    ai_data_processed_path: Path = Path(".")
    ai_data_artifacts_path: Path = Path(".")
    synthetic_bootstrap_enabled: bool = True
    synthetic_bootstrap_force: bool = False


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[4]
    ai_root = repo_root / "apps" / "ai"
    return Settings(
        environment=os.getenv("INGESTION_ENV", "development"),
        repo_root=repo_root,
        ai_src_path=ai_root / "src",
        ai_data_raw_path=ai_root / "data" / "raw",
        ai_data_processed_path=ai_root / "data" / "processed",
        ai_data_artifacts_path=ai_root / "data" / "artifacts",
        synthetic_bootstrap_enabled=_as_bool(os.getenv("DEMO_SYNTHETIC_BOOTSTRAP"), default=True),
        synthetic_bootstrap_force=_as_bool(os.getenv("DEMO_FORCE_SYNTHETIC_BOOTSTRAP"), default=False),
    )
