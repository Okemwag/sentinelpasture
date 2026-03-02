"""Semantic deduplication helpers."""

from __future__ import annotations


def semantic_key(*parts: str) -> str:
    return "::".join(parts)

