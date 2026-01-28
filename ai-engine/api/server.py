"""
FastAPI Server for AI Engine
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import uvicorn

from ..orchestrator import AIOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Intelligence Engine API",
    description="Advanced AI system for governance and security intelligence",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Orchestrator
orchestrator = AIOrchestrator()

# Pydantic models

class SignalInput(BaseModel):
    """Input signal model"""
    type: str = Field(..., description="Signal type")
    source: str = Field(..., description="Signal source")
    data: Dict[str, Any] = Field(..., description="Signal data")
    location: Dict[str, Any] = Field(..., description="Location information")
    temporal: Optional[Dict[str, Any]] = Field(
        None, description="Temporal context"
    )

class AnalysisRequest(BaseModel):
    """Analysis request model"""
    signals: List[SignalInput] = Field(..., description="List of signals to analyze")
    context: Optional[Dict[str, Any]] = Field(None, description="Decision context")

class AnalysisResponse(BaseModel):
    """Analysis response model"""
    assessment: Dict[str, Any]
    indicators: List[Dict[str, Any]]
    insights: Dict[str, Any]
    metadata: Dict[str, Any]

class SystemStatus(BaseModel):
    """System status model"""
    status: str
    active_assessments: int
    historical_data_points: int
    monitoring_active: bool
    subsystems: Dict[str, str]
    performance: Dict[str, Any]
    timestamp: str

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AI Intelligence Engine",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    try:
        status = orchestrator.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_signals(request: AnalysisRequest):
    """
    Analyze signals through intelligence pipeline
    
    Process multiple signals through the complete AI pipeline including:
    - Signal processing and normalization
    - Threat detection and pattern recognition
    - Risk assessment
    - Insight generation
    """
    try:
        logger.info(f"Received analysis request with {len(request.signals)} signals")
        
        # Convert Pydantic models to dicts
        raw_signals = []
        for signal in request.signals:
            raw_signal = {
                "type": signal.type,
                "source": signal.source,
                "data": signal.data,
                "location": signal.location,
                "temporal": signal.temporal or {"timestamp": datetime.now().isoformat()}
            }
            raw_signals.append(raw_signal)
        
        # Process through pipeline
        result = await orchestrator.process_intelligence_pipeline(
            raw_signals,
            request.context
        )
        
        logger.info(f"Analysis complete: {result['metadata']['processing_time_seconds']:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def analyze_batch(
    requests: List[AnalysisRequest],
    background_tasks: BackgroundTasks
):
    """
    Batch analysis endpoint
    
    Process multiple analysis requests in parallel
    """
    try:
        logger.info(f"Received batch request with {len(requests)} analyses")
        
        results = []
        for req in requests:
            raw_signals = [
                {
                    "type": s.type,
                    "source": s.source,
                    "data": s.data,
                    "location": s.location,
                    "temporal": s.temporal or {"timestamp": datetime.now().isoformat()}
                }
                for s in req.signals
            ]
            
            result = await orchestrator.process_intelligence_pipeline(
                raw_signals,
                req.context
            )
            results.append(result)
        
        return {
            "batch_size": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assessment/{assessment_id}")
async def get_assessment(assessment_id: str):
    """Get specific assessment by ID"""
    try:
        report = await orchestrator.export_assessment_report(assessment_id)
        return report
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving assessment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assessments")
async def list_assessments():
    """List all active assessments"""
    try:
        assessments = list(orchestrator.active_assessments.keys())
        return {
            "count": len(assessments),
            "assessments": assessments,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing assessments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate")
async def simulate_scenario(scenario: Dict[str, Any], context: Dict[str, Any]):
    """
    Simulate a scenario
    
    Run a what-if scenario simulation through the AI pipeline
    """
    try:
        logger.info(f"Simulating scenario: {scenario.get('name', 'unnamed')}")
        
        result = await orchestrator.simulate_scenario(scenario, context)
        
        return result
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get system metrics for monitoring"""
    try:
        status = orchestrator.get_system_status()
        
        # Format for Prometheus or similar
        metrics = {
            "signals_processed_total": status["performance"]["signals_processed"],
            "anomalies_detected_total": status["performance"]["anomalies_detected"],
            "active_assessments": status["active_assessments"],
            "subsystem_status": status["subsystems"],
            "timestamp": status["timestamp"]
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/signals/process")
async def process_single_signal(signal: SignalInput):
    """
    Process a single signal
    
    Quick endpoint for processing individual signals
    """
    try:
        raw_signal = {
            "type": signal.type,
            "source": signal.source,
            "data": signal.data,
            "location": signal.location,
            "temporal": signal.temporal or {"timestamp": datetime.now().isoformat()}
        }
        
        processed = await orchestrator.signal_processor.ingest_signal(
            signal_type=raw_signal["type"],
            source=raw_signal["source"],
            raw_data=raw_signal["data"],
            location=raw_signal["location"],
            temporal=raw_signal["temporal"]
        )
        
        return processed
        
    except Exception as e:
        logger.error(f"Signal processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("AI Engine API starting up...")
    logger.info("All subsystems initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("AI Engine API shutting down...")
    orchestrator.stop_monitoring()
    logger.info("Shutdown complete")

# Main entry point

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "aiengine.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    start_server(reload=True)
