"""Pydantic request models for the lightweight backend API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Signal(BaseModel):
    type: str
    source: str
    data: Dict[str, Any]
    location: Optional[Dict[str, float]] = None
    temporal: Optional[Dict[str, str]] = None


class ProcessSignalsRequest(BaseModel):
    signals: List[Signal]


class AnalyzeDriversRequest(BaseModel):
    data: Dict[str, Any]


class RecommendInterventionsRequest(BaseModel):
    region: str
    riskProfile: Dict[str, Any]

