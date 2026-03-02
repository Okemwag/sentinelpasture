"""Exact deduplication helpers."""

from __future__ import annotations


def exact_duplicate(seen: set[str], key: str) -> bool:
    if key in seen:
        return True
    seen.add(key)
    return False

