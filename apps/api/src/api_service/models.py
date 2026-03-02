"""ORM models for authentication and audit logging."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class UserRecord(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AuditEventRecord(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    actor_username: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    actor_role: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    resource: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    detail_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False, index=True)
