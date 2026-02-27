"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Enums
class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# Auth schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.VIEWER


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)


# Signal schemas
class SignalCreate(BaseModel):
    type: str
    source: str
    data: Dict[str, Any]
    location: Optional[Dict[str, float]] = None
    temporal: Optional[Dict[str, str]] = None


class SignalResponse(BaseModel):
    id: int
    signal_type: str
    source: str
    confidence: Optional[float]
    anomaly_score: Optional[float]
    severity: Optional[float]
    latitude: Optional[float]
    longitude: Optional[float]
    signal_timestamp: Optional[datetime]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# Assessment schemas
class ProcessSignalsRequest(BaseModel):
    signals: List[SignalCreate]
    region: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ThreatIndicatorResponse(BaseModel):
    id: int
    name: str
    indicator_type: str
    severity: float
    confidence: float
    description: Optional[str]
    patterns: Optional[Dict[str, Any]]
    mitigation_strategies: Optional[List[str]]
    
    model_config = ConfigDict(from_attributes=True)


class RiskAssessmentResponse(BaseModel):
    id: int
    assessment_id: str
    threat_level: float
    confidence: float
    pressure_index: Optional[float]
    region: Optional[str]
    risk_factors: Optional[List[str]]
    causal_factors: Optional[List[str]]
    signal_count: int
    indicator_count: int
    processing_time: Optional[float]
    model_version: Optional[str]
    created_at: datetime
    indicators: List[ThreatIndicatorResponse] = []
    
    model_config = ConfigDict(from_attributes=True)


class AssessmentListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    assessments: List[RiskAssessmentResponse]


# Regional data schemas
class RegionalDataCreate(BaseModel):
    region_code: str
    region_name: str
    risk_level: float
    stability_index: float
    trend: str
    metrics: Optional[Dict[str, Any]] = None
    drivers: Optional[List[Dict[str, Any]]] = None
    alerts: Optional[List[Dict[str, Any]]] = None
    data_date: datetime


class RegionalDataResponse(BaseModel):
    id: int
    region_code: str
    region_name: str
    risk_level: float
    stability_index: float
    trend: str
    metrics: Optional[Dict[str, Any]]
    drivers: Optional[List[Dict[str, Any]]]
    alerts: Optional[List[Dict[str, Any]]]
    data_date: datetime
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)


# Intervention schemas
class InterventionCreate(BaseModel):
    title: str
    category: str
    region: str
    description: Optional[str] = None
    expected_impact: str
    time_to_effect: str
    cost_band: str
    effectiveness_score: float
    confidence: float


class InterventionUpdate(BaseModel):
    status: Optional[str] = None
    actual_impact: Optional[Dict[str, Any]] = None
    lessons_learned: Optional[str] = None


class InterventionResponse(BaseModel):
    id: int
    title: str
    category: str
    region: str
    description: Optional[str]
    expected_impact: str
    time_to_effect: str
    cost_band: str
    effectiveness_score: float
    confidence: float
    status: str
    implemented_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# Alert schemas
class AlertCreate(BaseModel):
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    region: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AlertUpdate(BaseModel):
    status: Optional[str] = None


class AlertResponse(BaseModel):
    id: int
    alert_type: str
    severity: str
    title: str
    message: str
    region: Optional[str]
    source: Optional[str]
    status: str
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)


# Analytics schemas
class PredictionRequest(BaseModel):
    region: str = "national"
    timeframe: str = "7d"
    include_confidence: bool = True


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    confidence: float
    factors: List[str]
    metadata: Dict[str, Any]


class DriverAnalysisRequest(BaseModel):
    region: Optional[str] = None
    timeframe: Optional[str] = "30d"
    data: Optional[Dict[str, Any]] = None


class DriverAnalysisResponse(BaseModel):
    drivers: List[Dict[str, Any]]
    causal_relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# Data ingestion schemas
class DataIngestionCreate(BaseModel):
    source: str
    data_type: str
    data: List[Dict[str, Any]]


class DataIngestionResponse(BaseModel):
    id: int
    job_id: str
    source: str
    data_type: str
    status: str
    records_processed: int
    records_failed: int
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# System schemas
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    database: str
    ai_engine: str


class SystemStatusResponse(BaseModel):
    status: str
    active_assessments: int
    total_signals: int
    total_assessments: int
    database_status: str
    ai_engine_status: str
    timestamp: datetime
