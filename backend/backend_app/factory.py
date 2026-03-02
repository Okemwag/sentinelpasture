"""Application factory for the lightweight backend API."""

from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    AnalyzeDriversRequest,
    ProcessSignalsRequest,
    RecommendInterventionsRequest,
)
from .services.ai_runtime import LocalAIRuntime


def create_app() -> FastAPI:
    app = FastAPI(
        title="National Risk Intelligence API",
        description="Backend API for governance-grade risk intelligence platform",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ai_runtime = LocalAIRuntime()

    @app.get("/")
    async def root():
        return {
            "status": "operational",
            "service": "National Risk Intelligence API",
            "version": "1.0.0",
            "ai_engine_status": ai_runtime.engine_name,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "ai_engine": ai_runtime.engine_name,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @app.post("/api/process")
    async def process_signals(request: ProcessSignalsRequest):
        try:
            raw_signals = [signal.model_dump() for signal in request.signals]
            return await ai_runtime.process_signals(raw_signals)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    @app.get("/api/predict")
    async def predict_risk(region: str = "national", timeframe: str = "7d"):
        try:
            return await ai_runtime.predict_risk(region, timeframe)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    @app.post("/api/analyze-drivers")
    async def analyze_drivers(request: AnalyzeDriversRequest):
        try:
            return await ai_runtime.analyze_drivers(request.data)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    @app.post("/api/recommend-interventions")
    async def recommend_interventions(request: RecommendInterventionsRequest):
        try:
            return await ai_runtime.recommend_interventions(request.region, request.riskProfile)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc

    return app


app = create_app()

