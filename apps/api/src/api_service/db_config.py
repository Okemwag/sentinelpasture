"""Database URL normalization and engine options."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


SQLITE_DEFAULT_URL = "sqlite:///./governance_intel.db"


def _normalize_database_url(raw_url: str | None) -> str:
    if not raw_url:
        return SQLITE_DEFAULT_URL
    if raw_url.startswith("postgres://"):
        return raw_url.replace("postgres://", "postgresql+psycopg://", 1)
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw_url


@dataclass(frozen=True)
class DatabaseConfig:
    url: str = SQLITE_DEFAULT_URL
    connect_args: dict[str, Any] = field(default_factory=dict)
    engine_kwargs: dict[str, Any] = field(default_factory=dict)


def load_database_config() -> DatabaseConfig:
    url = _normalize_database_url(os.getenv("DATABASE_URL"))
    if url.startswith("sqlite"):
        return DatabaseConfig(url=url, connect_args={"check_same_thread": False})
    return DatabaseConfig(url=url, engine_kwargs={"pool_pre_ping": True})

