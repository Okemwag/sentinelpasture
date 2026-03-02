"""Database setup for the API service."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


def _default_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///./governance_intel.db")


DATABASE_URL = _default_database_url()
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass

