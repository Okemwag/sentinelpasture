"""Compatibility wrapper for AI service logic."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from api_service.ai import AIGateway, AIGatewayError  # noqa: E402,F401

