"""Audit event persistence helpers."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from .models import AuditEventRecord


def log_audit_event(
    db: Session,
    *,
    actor_username: str,
    actor_role: str,
    action: str,
    resource: str,
    outcome: str,
    detail: dict[str, Any] | None = None,
) -> AuditEventRecord:
    record = AuditEventRecord(
        actor_username=actor_username,
        actor_role=actor_role,
        action=action,
        resource=resource,
        outcome=outcome,
        detail_json=json.dumps(detail or {}, default=str),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_audit_events(db: Session, limit: int = 100) -> list[AuditEventRecord]:
    stmt = select(AuditEventRecord).order_by(desc(AuditEventRecord.created_at)).limit(limit)
    return list(db.scalars(stmt))
