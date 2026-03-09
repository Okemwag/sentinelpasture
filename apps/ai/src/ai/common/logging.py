"""Logging helpers for the AI workspace."""

from __future__ import annotations

import logging
import os


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(name: str | None = None, *, level: str | int | None = None) -> logging.Logger:
    """Configure process logging once and return a named logger."""
    resolved_level = _resolve_level(level)
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(level=resolved_level, format=DEFAULT_LOG_FORMAT)
    else:
        root_logger.setLevel(resolved_level)
        for handler in root_logger.handlers:
            handler.setLevel(resolved_level)

    logger = logging.getLogger(name) if name else root_logger
    logger.setLevel(resolved_level)
    return logger


def _resolve_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    configured = os.getenv("GOVERNANCE_INTEL_LOG_LEVEL", "INFO")
    return getattr(logging, configured.upper(), logging.INFO)
