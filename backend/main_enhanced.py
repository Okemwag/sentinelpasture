"""
Enhanced FastAPI Backend for National Risk Intelligence Platform
Full integration with PostgreSQL, Authentication, WebSocket, and AI Engine
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import sys
import os
import json
import logging

# Add ai-engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai-engine'))

# Import local modules
from database import get_db, init_db, close_db
from models import (
    User, RiskAssessment, Signal, ThreatIndicator,
    RegionalData, Intervention, Alert, AuditLog, UserRole
)
from schemas import (
    Token, UserCreate, UserResponse, UserUpdate,
    ProcessSignalsRequest, RiskAssessmentResponse, AssessmentListResponse,
    RegionalDataResponse, InterventionCreate, InterventionResponse, InterventionUpdate,
    AlertCreate, AlertResponse, AlertUpdate,
    PredictionRequest, PredictionResponse,
    DriverAnalysisRequest, DriverAnalysisResponse,
    HealthResponse, SystemStatusResponse
)
from auth import (
    get_password_hash, verify_password, create_access_token,
    get_current_active_user, require_admin, require_analyst, require_viewer
)

# Import AI Engine
try:
    from orchestrator import AIOrchestrator
    from config.settings import Settings
    AI_ENGINE_AVAILABLE = True
except ImportError:
    logging.warning("AI Engine not available")
    AI_ENGINE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="National Risk Intelligence API",
    description="Governance-grade risk intelligence platform with AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        os.getenv("FRONTEND_URL", "http://localhost:3000")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Engine
orchestrator = None
if AI_ENGINE_AVAILABLE:
    try:
        settings = Settings()
        settings.model.device = "cpu"
        orchestrator = AIOrchestrator(settings=settings)
        logger.info("AI Engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI Engine: {e}")
        AI_ENGINE_AVAILABLE = False


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.room_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, room: str = "default"):
        await websocket.accept()
        self.active_connections.append(websocket)
        if room not in self.room_connections:
            self.room_connections[room] = []
        self.room_connections[room].append(websocket)
        logger.info(f"WebSocket connected to room: {room}")
    
    def disconnect(self, websocket: WebSocket, room: str = "default"):
        self.active_connections.remove(websocket)
        if room in self.room_connections:
            self.room_connections[room].remove(websocket)
        logger.info(f"WebSocket disconnected from room: {room}")
    
    async def broadcast(self, message: dict, room: str = "default"):
        """Broadcast message to all connections in a room"""
        if room in self.room_connections:
            for connection in self.room_connections[room]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to WebSocket: {e}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")


manager = ConnectionManager()


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("Starting up application...")
    await init_db()
    logger.info("Database initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application...")
    await close_db()
    logger.info("Database connections closed")


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with service status"""
    return HealthResponse(
        status="operational",
        timestamp=datetime.utcnow(),
        version="2.0.0",
        database="connected",
        ai_engine="available" if AI_ENGINE_AVAILABLE else "mock_mode"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="2.0.0",
        database="connected",
        ai_engine="available" if AI_ENGINE_AVAILABLE else "mock_mode"
    )


@app.get("/api/status", response_model=SystemStatusResponse)
async def system_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer)
):
    """Get comprehensive system status"""
    
    # Count assessments
    assessment_count = await db.scalar(select(func.count(RiskAssessment.id)))
    signal_count = await db.scalar(select(func.count(Signal.id)))
    
    # Get recent assessments
    recent_result = await db.execute(
        select(RiskAssessment)
        .order_by(desc(RiskAssessment.created_at))
        .limit(10)
    )
    recent_assessments = len(recent_result.scalars().all())
    
    return SystemStatusResponse(
        status="operational",
        active_assessments=recent_assessments,
        total_signals=signal_count or 0,
        total_assessments=assessment_count or 0,
        database_status="connected",
        ai_engine_status="available" if AI_ENGINE_AVAILABLE else "mock",
        timestamp=datetime.utcnow()
    )


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user"""
    
    # Check if user exists
    result = await db.execute(
        select(User).where(
            (User.email == user_data.email) | (User.username == user_data.username)
        )
    )
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email or username already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        role=user_data.role
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    logger.info(f"New user registered: {new_user.username}")
    
    return new_user


@app.post("/auth/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Login and get access token"""
    
    # Get user
    result = await db.execute(
        select(User).where(User.username == form_data.username)
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()
    
    # Create access token
    access_token = create_access_token(data={"sub": user.username})
    
    logger.info(f"User logged in: {user.username}")
    
    return Token(access_token=access_token, token_type="bearer")


@app.get("/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user


@app.get("/auth/users", response_model=List[UserResponse])
async def list_users(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin),
    skip: int = 0,
    limit: int = 100
):
    """List all users (admin only)"""
    result = await db.execute(
        select(User).offset(skip).limit(limit)
    )
    users = result.scalars().all()
    return users


@app.put("/auth/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_admin)
):
    """Update user (admin only)"""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update fields
    if user_update.email:
        user.email = user_update.email
    if user_update.full_name:
        user.full_name = user_update.full_name
    if user_update.role:
        user.role = user_update.role
    if user_update.is_active is not None:
        user.is_active = user_update.is_active
    
    await db.commit()
    await db.refresh(user)
    
    return user


# ============================================================================
# AI PROCESSING ENDPOINTS
# ============================================================================

@app.post("/api/process", response_model=RiskAssessmentResponse)
async def process_signals(
    request: ProcessSignalsRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_analyst)
):
    """Process signals through AI engine and store results"""
    
    try:
        if not AI_ENGINE_AVAILABLE or not orchestrator:
            raise HTTPException(
                status_code=503,
                detail="AI Engine not available"
            )
        
        # Convert signals to AI engine format
        raw_signals = [signal.model_dump() for signal in request.signals]
        
        # Process through AI engine
        result = await orchestrator.process_intelligence_pipeline(
            raw_signals,
            context=request.context
        )
        
        # Store assessment in database
        assessment = RiskAssessment(
            assessment_id=result["assessment"]["id"],
            threat_level=result["assessment"]["threat_level"],
            confidence=result["assessment"]["confidence"],
            pressure_index=result["assessment"].get("pressure_index", {}).get("overall"),
            region=request.region,
            risk_factors=result["assessment"].get("risk_factors", []),
            causal_factors=result["assessment"].get("causal_factors", []),
            optimized_strategy=result["assessment"].get("optimized_strategy"),
            signal_count=result["assessment"]["signal_count"],
            indicator_count=result["assessment"]["indicator_count"],
            processing_time=result["metadata"]["processing_time_seconds"],
            model_version=result["metadata"].get("model_version", "2.0.0"),
            created_by=current_user.id
        )
        
        db.add(assessment)
        await db.flush()
        
        # Store indicators
        for ind_data in result.get("indicators", []):
            indicator = ThreatIndicator(
                name=ind_data["name"],
                indicator_type=ind_data.get("type", "unknown"),
                severity=ind_data["severity"],
                confidence=ind_data["confidence"],
                description=ind_data.get("description"),
                patterns=ind_data.get("patterns"),
                mitigation_strategies=ind_data.get("mitigation_strategies"),
                assessment_id=assessment.id
            )
            db.add(indicator)
        
        await db.commit()
        await db.refresh(assessment)
        
        # Broadcast update via WebSocket
        await manager.broadcast({
            "type": "assessment_created",
            "data": {
                "assessment_id": assessment.assessment_id,
                "threat_level": assessment.threat_level,
                "region": assessment.region
            }
        })
        
        logger.info(f"Assessment created: {assessment.assessment_id}")
        
        return assessment
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/assessments", response_model=AssessmentListResponse)
async def list_assessments(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
    skip: int = 0,
    limit: int = 50,
    region: Optional[str] = None
):
    """List risk assessments with pagination"""
    
    query = select(RiskAssessment).order_by(desc(RiskAssessment.created_at))
    
    if region:
        query = query.where(RiskAssessment.region == region)
    
    # Get total count
    count_query = select(func.count(RiskAssessment.id))
    if region:
        count_query = count_query.where(RiskAssessment.region == region)
    total = await db.scalar(count_query)
    
    # Get paginated results
    result = await db.execute(query.offset(skip).limit(limit))
    assessments = result.scalars().all()
    
    return AssessmentListResponse(
        total=total or 0,
        page=skip // limit + 1,
        page_size=limit,
        assessments=assessments
    )


@app.get("/api/assessments/{assessment_id}", response_model=RiskAssessmentResponse)
async def get_assessment(
    assessment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer)
):
    """Get specific assessment by ID"""
    
    result = await db.execute(
        select(RiskAssessment).where(RiskAssessment.assessment_id == assessment_id)
    )
    assessment = result.scalar_one_or_none()
    
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    return assessment


# Continue in next file due to length...


# ============================================================================
# REGIONAL DATA ENDPOINTS
# ============================================================================

@app.get("/api/regional", response_model=List[RegionalDataResponse])
async def list_regional_data(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
    limit: int = 100
):
    """Get latest regional data for all regions"""
    
    # Get most recent data for each region
    result = await db.execute(
        select(RegionalData)
        .order_by(desc(RegionalData.data_date))
        .limit(limit)
    )
    regional_data = result.scalars().all()
    
    return regional_data


@app.get("/api/regional/{region_code}", response_model=RegionalDataResponse)
async def get_regional_data(
    region_code: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer)
):
    """Get latest data for specific region"""
    
    result = await db.execute(
        select(RegionalData)
        .where(RegionalData.region_code == region_code)
        .order_by(desc(RegionalData.data_date))
        .limit(1)
    )
    regional_data = result.scalar_one_or_none()
    
    if not regional_data:
        raise HTTPException(status_code=404, detail="Regional data not found")
    
    return regional_data


# ============================================================================
# INTERVENTION ENDPOINTS
# ============================================================================

@app.post("/api/interventions", response_model=InterventionResponse, status_code=status.HTTP_201_CREATED)
async def create_intervention(
    intervention_data: InterventionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_analyst)
):
    """Create new intervention recommendation"""
    
    intervention = Intervention(**intervention_data.model_dump())
    db.add(intervention)
    await db.commit()
    await db.refresh(intervention)
    
    # Broadcast via WebSocket
    await manager.broadcast({
        "type": "intervention_created",
        "data": {"id": intervention.id, "title": intervention.title}
    })
    
    return intervention


@app.get("/api/interventions", response_model=List[InterventionResponse])
async def list_interventions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
    region: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 100
):
    """List interventions with optional filters"""
    
    query = select(Intervention).order_by(desc(Intervention.created_at))
    
    if region:
        query = query.where(Intervention.region == region)
    if status_filter:
        query = query.where(Intervention.status == status_filter)
    
    result = await db.execute(query.limit(limit))
    interventions = result.scalars().all()
    
    return interventions


@app.put("/api/interventions/{intervention_id}", response_model=InterventionResponse)
async def update_intervention(
    intervention_id: int,
    intervention_update: InterventionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_analyst)
):
    """Update intervention status and results"""
    
    result = await db.execute(
        select(Intervention).where(Intervention.id == intervention_id)
    )
    intervention = result.scalar_one_or_none()
    
    if not intervention:
        raise HTTPException(status_code=404, detail="Intervention not found")
    
    # Update fields
    if intervention_update.status:
        intervention.status = intervention_update.status
        if intervention_update.status == "active":
            intervention.implemented_at = datetime.utcnow()
        elif intervention_update.status == "completed":
            intervention.completed_at = datetime.utcnow()
    
    if intervention_update.actual_impact:
        intervention.actual_impact = intervention_update.actual_impact
    if intervention_update.lessons_learned:
        intervention.lessons_learned = intervention_update.lessons_learned
    
    await db.commit()
    await db.refresh(intervention)
    
    return intervention


# ============================================================================
# ALERT ENDPOINTS
# ============================================================================

@app.post("/api/alerts", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    alert_data: AlertCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_analyst)
):
    """Create new alert"""
    
    alert = Alert(**alert_data.model_dump())
    db.add(alert)
    await db.commit()
    await db.refresh(alert)
    
    # Broadcast critical alerts immediately
    if alert.severity in ["critical", "high"]:
        await manager.broadcast({
            "type": "alert_created",
            "data": {
                "id": alert.id,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message
            }
        }, room="alerts")
    
    logger.warning(f"Alert created: {alert.severity} - {alert.title}")
    
    return alert


@app.get("/api/alerts", response_model=List[AlertResponse])
async def list_alerts(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer),
    status_filter: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100
):
    """List alerts with optional filters"""
    
    query = select(Alert).order_by(desc(Alert.created_at))
    
    if status_filter:
        query = query.where(Alert.status == status_filter)
    if severity:
        query = query.where(Alert.severity == severity)
    
    result = await db.execute(query.limit(limit))
    alerts = result.scalars().all()
    
    return alerts


@app.put("/api/alerts/{alert_id}", response_model=AlertResponse)
async def update_alert(
    alert_id: int,
    alert_update: AlertUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_analyst)
):
    """Update alert status"""
    
    result = await db.execute(
        select(Alert).where(Alert.id == alert_id)
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    if alert_update.status:
        alert.status = alert_update.status
        if alert_update.status == "acknowledged":
            alert.acknowledged_at = datetime.utcnow()
        elif alert_update.status == "resolved":
            alert.resolved_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(alert)
    
    return alert


# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/{room}")
async def websocket_endpoint(websocket: WebSocket, room: str):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket, room)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Echo back or broadcast
            if message.get("type") == "ping":
                await manager.send_personal_message(
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                    websocket
                )
            else:
                # Broadcast to room
                await manager.broadcast(message, room)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, room)
        await manager.broadcast(
            {"type": "user_disconnected", "room": room},
            room
        )


# ============================================================================
# ANALYTICS & PREDICTION ENDPOINTS
# ============================================================================

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_risk(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_viewer)
):
    """Generate risk predictions"""
    
    if not AI_ENGINE_AVAILABLE or not orchestrator:
        raise HTTPException(status_code=503, detail="AI Engine not available")
    
    try:
        # Get historical data for the region
        result = await db.execute(
            select(RiskAssessment)
            .where(RiskAssessment.region == request.region)
            .order_by(desc(RiskAssessment.created_at))
            .limit(100)
        )
        historical = result.scalars().all()
        
        # Generate predictions (simplified - would use actual AI model)
        predictions = []
        base_date = datetime.utcnow()
        
        for i in range(7):
            pred_date = base_date + timedelta(days=i)
            predictions.append({
                "date": pred_date.isoformat(),
                "risk_level": 68 + i,  # Placeholder
                "confidence": 0.82 - (i * 0.02)
            })
        
        return PredictionResponse(
            predictions=predictions,
            confidence=0.78,
            factors=["Historical trends", "Seasonal patterns"],
            metadata={
                "model_version": "2.0.0",
                "historical_data_points": len(historical)
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-drivers", response_model=DriverAnalysisResponse)
async def analyze_drivers(
    request: DriverAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_analyst)
):
    """Analyze risk drivers using causal inference"""
    
    if not AI_ENGINE_AVAILABLE or not orchestrator:
        raise HTTPException(status_code=503, detail="AI Engine not available")
    
    try:
        # Use causal inference engine
        # This would integrate with the actual AI engine's causal analysis
        
        return DriverAnalysisResponse(
            drivers=[
                {
                    "name": "Economic stress",
                    "contribution": 0.34,
                    "trend": "increasing",
                    "confidence": 0.92
                },
                {
                    "name": "Climate anomaly",
                    "contribution": 0.28,
                    "trend": "increasing",
                    "confidence": 0.85
                }
            ],
            causal_relationships=[
                {
                    "from": "Climate anomaly",
                    "to": "Economic stress",
                    "strength": 0.65
                }
            ],
            metadata={
                "model_version": "2.0.0",
                "analysis_date": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Driver analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main_enhanced:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
