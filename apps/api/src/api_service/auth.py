"""Lightweight demo auth and role checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    username: str
    full_name: str
    role: str


class AuthService:
    def __init__(self) -> None:
        self._users = {
            "admin": (User("admin", "Platform Admin", "admin"), "admin123"),
            "analyst": (User("analyst", "Platform Analyst", "analyst"), "analyst123"),
            "viewer": (User("viewer", "Platform Viewer", "viewer"), "viewer123"),
        }

    def login(self, username: str, password: str) -> dict | None:
        record = self._users.get(username)
        if not record or record[1] != password:
            return None
        user = record[0]
        return {
            "access_token": self.token_for(username),
            "token_type": "bearer",
            "user": {
                "username": user.username,
                "full_name": user.full_name,
                "role": user.role,
            },
        }

    def user_from_auth_header(self, header: str | None) -> User | None:
        if not header:
            return None
        token = header.replace("Bearer", "", 1).strip()
        for username, (user, _) in self._users.items():
            if token == self.token_for(username):
                return user
        return None

    def token_for(self, username: str) -> str:
        return hashlib.sha256(f"gov-intel-demo:{username}".encode("utf-8")).hexdigest()


def role_allowed(user_role: str, *allowed: str) -> bool:
    ranks = {"viewer": 1, "analyst": 2, "admin": 3}
    user_rank = ranks.get(user_role, 0)
    return any(user_rank >= ranks.get(role, 0) for role in allowed)

