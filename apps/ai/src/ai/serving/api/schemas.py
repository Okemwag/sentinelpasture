"""Contract-shaped request and response models for inference."""

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class RiskRequest(BaseModel):
    region_id: str
    at_time: datetime | None = None
    signals: List[Dict[str, Any]] = Field(default_factory=list)


class Driver(BaseModel):
    name: str
    contribution: float
    direction: str


class RiskResponse(BaseModel):
    region_id: str
    risk_score: float
    confidence_band: str
    confidence: float
    top_drivers: List[Driver]
    known_data_gaps: List[str]
    model_version: str
    feature_snapshot_timestamp: datetime


class ExplainRequest(BaseModel):
    region_id: str
    risk_score: float | None = None


class ExplainResponse(BaseModel):
    region_id: str
    summary: str
    top_drivers: List[Driver]
    uncertainty_notes: List[str]
    model_version: str
    feature_snapshot_timestamp: datetime


class InterventionRequest(BaseModel):
    region_id: str
    risk_score: float


class InterventionOption(BaseModel):
    category: str
    expected_impact: str
    time_to_effect: str
    confidence: str
    constraints_applied: List[str]


class InterventionResponse(BaseModel):
    region_id: str
    interventions: List[InterventionOption]
    model_version: str
    feature_snapshot_timestamp: datetime

