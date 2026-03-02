"""Database-backed JWT authentication and role checks."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import UserRecord


@dataclass(frozen=True)
class User:
    id: int
    username: str
    email: str
    full_name: str
    role: str
    is_active: bool


class AuthService:
    token_ttl_hours = int(os.getenv("ACCESS_TOKEN_TTL_HOURS", "12"))

    def __init__(self) -> None:
        self._jwt_secret = os.getenv("JWT_SECRET", "local-dev-change-me")
        self._jwt_algorithm = "HS256"

    def login(self, db: Session, username: str, password: str) -> dict | None:
        record = db.scalar(select(UserRecord).where(UserRecord.username == username))
        if not record or not record.is_active or not self.verify_password(password, record.password_hash):
            return None
        record.last_login_at = datetime.utcnow()
        db.add(record)
        db.commit()
        user = self.to_user(record)
        return {
            "access_token": self.token_for(user),
            "token_type": "bearer",
            "expires_in": self.token_ttl_hours * 3600,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
            },
        }

    def user_from_auth_header(self, db: Session, header: str | None) -> User | None:
        if not header:
            return None
        token = header.replace("Bearer", "", 1).strip()
        payload = self.decode_token(token)
        if not payload:
            return None
        user_id = payload.get("sub")
        if user_id is None:
            return None
        record = db.get(UserRecord, int(user_id))
        if not record or not record.is_active:
            return None
        return self.to_user(record)

    def token_for(self, user: User) -> str:
        now = datetime.now(UTC)
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "role": user.role,
            "iat": now,
            "exp": now + timedelta(hours=self.token_ttl_hours),
        }
        return jwt.encode(payload, self._jwt_secret, algorithm=self._jwt_algorithm)

    def decode_token(self, token: str) -> dict | None:
        try:
            return jwt.decode(token, self._jwt_secret, algorithms=[self._jwt_algorithm])
        except jwt.PyJWTError:
            return None

    def hash_password(self, password: str, *, salt: str | None = None) -> str:
        local_salt = salt or secrets.token_hex(16)
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), local_salt.encode("utf-8"), 600_000)
        return f"{local_salt}${derived.hex()}"

    def verify_password(self, password: str, stored_hash: str) -> bool:
        if "$" not in stored_hash:
            return False
        salt, expected = stored_hash.split("$", 1)
        candidate = self.hash_password(password, salt=salt).split("$", 1)[1]
        return hmac.compare_digest(candidate, expected)

    def seed_defaults(self, db: Session) -> None:
        defaults = [
            ("admin", "admin@sentinelpasture.gov", "Platform Admin", "admin", os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123!")),
            ("analyst", "analyst@sentinelpasture.gov", "Platform Analyst", "analyst", os.getenv("DEFAULT_ANALYST_PASSWORD", "analyst123!")),
            ("operator", "operator@sentinelpasture.gov", "Platform Operator", "operator", os.getenv("DEFAULT_OPERATOR_PASSWORD", "operator123!")),
            ("viewer", "viewer@sentinelpasture.gov", "Platform Viewer", "viewer", os.getenv("DEFAULT_VIEWER_PASSWORD", "viewer123!")),
        ]
        for username, email, full_name, role, password in defaults:
            existing = db.scalar(select(UserRecord).where(UserRecord.username == username))
            if existing:
                continue
            db.add(
                UserRecord(
                    username=username,
                    email=email,
                    full_name=full_name,
                    role=role,
                    password_hash=self.hash_password(password),
                    is_active=True,
                )
            )
        db.commit()

    def to_user(self, record: UserRecord) -> User:
        return User(
            id=record.id,
            username=record.username,
            email=record.email,
            full_name=record.full_name,
            role=record.role,
            is_active=record.is_active,
        )


def role_allowed(user_role: str, *allowed: str) -> bool:
    ranks = {"viewer": 1, "analyst": 2, "operator": 3, "admin": 4}
    user_rank = ranks.get(user_role, 0)
    return any(user_rank >= ranks.get(role, 0) for role in allowed)
