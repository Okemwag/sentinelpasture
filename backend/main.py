"""
FastAPI Backend for National Risk Intelligence Platform
Connects Next.js frontend with Python AI Engine
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
from datetime import datetime

# Add ai-engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai-engine'))

try:
    from orchestrator import AIOrchestrator
    from config import Settings
    AI_ENGINE_AVAILABLE = True
except ImportError:
    print("Warning: AI Engine not available, using mock responses")
    AI_ENGINE_AVAILABLE = False

app = FastAPI(
    title="National Risk Intelligence API",
    description="Backend API for governance-grade risk intelligence platform",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Engine (if available)
orchestrator = None
if AI_ENGINE_AVAILABLE:
    try:
        settings = Settings()
        settings.model.device = "cpu"
        orchestrator = AIOrchestrator(settings=settings)
        print("AI Engine initialized successfully")
    except Exception as e:
        print(f"Failed to initialize AI Engine: {e}")
        AI_ENGINE_AVAILABLE = False


# Request/Response Models
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


# Health check
@app.get("/")
async def root():
    return {
        "status": "operational",
        "service": "National Risk Intelligence API",
        "version": "1.0.0",
        "ai_engine_status": "available" if AI_ENGINE_AVAILABLE else "mock_mode",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ai_engine": AI_ENGINE_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }


# AI Engine Endpoints
@app.post("/api/process")
async def process_signals(request: ProcessSignalsRequest):
    """Process signals through AI engine for threat assessment"""
    try:
        if AI_ENGINE_AVAILABLE and orchestrator:
            # Convert to format expected by AI engine
            raw_signals = [signal.dict() for signal in request.signals]
            result = await orchestrator.process_intelligence_pipeline(raw_signals)
            
            return {
                "assessment": {
                    "threat_level": result.get("assessment", {}).get("threat_level", 0),
                    "confidence": result.get("assessment", {}).get("confidence", 0),
                    "risk_factors": result.get("assessment", {}).get("risk_factors", [])
                },
                "indicators": result.get("indicators", []),
                "recommendations": result.get("recommendations", []),
                "metadata": {
                    "model_version": "v2.4.1",
                    "processing_time": result.get("processing_time", 0)
                }
            }
        else:
            # Mock response
            return get_mock_assessment()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/api/predict")
async def predict_risk(region: str = "national", timeframe: str = "7d"):
    """Generate risk predictions for a region"""
    try:
        if AI_ENGINE_AVAILABLE and orchestrator:
            # Use AI engine for predictions
            # This would call the appropriate prediction methods
            pass
        
        # Mock response for now
        return get_mock_prediction(region, timeframe)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/analyze-drivers")
async def analyze_drivers(request: AnalyzeDriversRequest):
    """Analyze risk drivers using causal inference"""
    try:
        if AI_ENGINE_AVAILABLE and orchestrator:
            # Use causal inference engine
            pass
        
        # Mock response
        return get_mock_driver_analysis()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/recommend-interventions")
async def recommend_interventions(request: RecommendInterventionsRequest):
    """Recommend interventions based on risk profile"""
    try:
        if AI_ENGINE_AVAILABLE and orchestrator:
            # Use recommendation engine
            pass
        
        # Mock response
        return get_mock_interventions(request.region)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


# Mock data functions
def get_mock_assessment():
    return {
        "assessment": {
            "threat_level": 67,
            "confidence": 0.85,
            "risk_factors": ["Economic stress", "Climate anomaly", "Social tension"]
        },
        "indicators": [
            {
                "type": "economic",
                "severity": 0.72,
                "description": "Unemployment rate increasing in urban areas"
            },
            {
                "type": "environmental",
                "severity": 0.65,
                "description": "Drought conditions affecting agricultural regions"
            }
        ],
        "recommendations": [
            {
                "action": "Economic stabilization program",
                "priority": 1,
                "estimated_impact": "High"
            }
        ],
        "metadata": {
            "model_version": "v2.4.1",
            "processing_time": 0.234
        }
    }


def get_mock_prediction(region: str, timeframe: str):
    return {
        "predictions": [
            {"date": "2026-02-20", "risk_level": 68, "confidence": 0.82},
            {"date": "2026-02-21", "risk_level": 69, "confidence": 0.80},
            {"date": "2026-02-22", "risk_level": 70, "confidence": 0.78},
        ],
        "confidence": 0.78,
        "factors": ["Economic indicators trending upward", "Seasonal patterns"]
    }


def get_mock_driver_analysis():
    return {
        "drivers": [
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
        "causal_relationships": [
            {
                "from": "Climate anomaly",
                "to": "Economic stress",
                "strength": 0.65
            }
        ]
    }


def get_mock_interventions(region: str):
    return {
        "interventions": [
            {
                "category": "Economic stabilization program",
                "expectedImpact": "High",
                "timeToEffect": "Medium",
                "costBand": "KES 800M - 1.2B",
                "confidence": "High",
                "effectiveness_score": 0.85
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
