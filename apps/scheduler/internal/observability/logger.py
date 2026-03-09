"""Simple logging helper for the scheduler service."""

from __future__ import annotations

import logging
import os


def get_logger(name: str = "scheduler") -> logging.Logger:
    level_name = os.getenv("GOVERNANCE_INTEL_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    else:
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
