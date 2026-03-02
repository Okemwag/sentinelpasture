"""Simple logging helper for the scheduler service."""

from __future__ import annotations

import logging


def get_logger(name: str = "scheduler") -> logging.Logger:
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)

