"""Database setup for the API service."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .db_config import load_database_config


db_config = load_database_config()
engine = create_engine(
    db_config.url,
    future=True,
    connect_args=db_config.connect_args,
    **db_config.engine_kwargs,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass
