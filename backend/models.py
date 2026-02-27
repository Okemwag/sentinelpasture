"""
Database models for Risk Intelligence Platform
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    Text, JSON, ForeignKey, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from .database import Base


class UserRole(str, enum.Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(SQLEnum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    assessments = relationship("RiskAssessment", back_populates="created_by_user")
    audit_logs = relationship("AuditLog", back_populates="user")


class RiskAssessment(Base):
    """Risk assessment results"""
    __tablename__ = "risk_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(String(100), unique=True, index=True, nullable=False)
    threat_level = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    pressure_index = Column(Float)
    region = Column(String(100), index=True)
    
    # JSON fields for complex data
    risk_factors = Column(JSON)
    causal_factors = Column(JSON)
    optimized_strategy = Column(JSON)
    
    # Metadata
    signal_count = Column(Integer, default=0)
    indicator_count = Column(Integer, default=0)
    processing_time = Column(Float)
    model_version = Column(String(50))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    created_by_user = relationship("User", back_populates="assessments")
    indicators = relationship("ThreatIndicator", back_populates="assessment")
    signals = relationship("Signal", back_populates="assessment")


class Signal(Base):
    """Processed signals"""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, index=True)
    signal_type = Column(String(100), index=True, nullable=False)
    source = Column(String(255), nullable=False)
    
    # Signal data
    raw_data = Column(JSON)
    processed_data = Column(JSON)
    embeddings = Column(JSON)  # Store as JSON array
    
    # Metrics
    confidence = Column(Float)
    anomaly_score = Column(Float)
    severity = Column(Float)
    
    # Location
    latitude = Column(Float)
    longitude = Column(Float)
    location_name = Column(String(255))
    
    # Temporal
    signal_timestamp = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    assessment_id = Column(Integer, ForeignKey("risk_assessments.id"))
    assessment = relationship("RiskAssessment", back_populates="signals")


class ThreatIndicator(Base):
    """Threat indicators identified by AI"""
    __tablename__ = "threat_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    indicator_type = Column(String(100), index=True)
    severity = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    
    description = Column(Text)
    patterns = Column(JSON)
    mitigation_strategies = Column(JSON)
    
    # Relationships
    assessment_id = Column(Integer, ForeignKey("risk_assessments.id"))
    assessment = relationship("RiskAssessment", back_populates="indicators")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RegionalData(Base):
    """Regional risk and stability data"""
    __tablename__ = "regional_data"
    
    id = Column(Integer, primary_key=True, index=True)
    region_code = Column(String(50), index=True, nullable=False)
    region_name = Column(String(255), nullable=False)
    
    # Risk metrics
    risk_level = Column(Float)
    stability_index = Column(Float)
    trend = Column(String(50))
    
    # Detailed metrics
    metrics = Column(JSON)
    drivers = Column(JSON)
    alerts = Column(JSON)
    
    # Temporal
    data_date = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Intervention(Base):
    """Intervention recommendations and tracking"""
    __tablename__ = "interventions"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    category = Column(String(100), index=True)
    region = Column(String(100), index=True)
    
    description = Column(Text)
    expected_impact = Column(String(50))
    time_to_effect = Column(String(50))
    cost_band = Column(String(100))
    
    # Effectiveness
    effectiveness_score = Column(Float)
    confidence = Column(Float)
    
    # Status tracking
    status = Column(String(50), default="recommended")  # recommended, planned, active, completed
    implemented_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    
    # Results
    actual_impact = Column(JSON)
    lessons_learned = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Alert(Base):
    """System alerts and notifications"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String(100), index=True, nullable=False)
    severity = Column(String(50), nullable=False)  # critical, high, medium, low
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    
    region = Column(String(100), index=True)
    source = Column(String(255))
    
    # Status
    status = Column(String(50), default="active")  # active, acknowledged, resolved
    acknowledged_at = Column(DateTime(timezone=True))
    resolved_at = Column(DateTime(timezone=True))
    
    # Metadata
    metadata = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class AuditLog(Base):
    """Audit log for tracking user actions"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(String(100))
    
    details = Column(JSON)
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")


class DataIngestion(Base):
    """Track data ingestion jobs"""
    __tablename__ = "data_ingestions"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, index=True)
    source = Column(String(255), nullable=False)
    data_type = Column(String(100))
    
    # Status
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    records_processed = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    
    # Results
    error_message = Column(Text)
    validation_results = Column(JSON)
    
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
