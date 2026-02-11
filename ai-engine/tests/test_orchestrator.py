"""
Tests for AI Orchestrator
"""

import pytest
from datetime import datetime

from ai_engine import AIOrchestrator, Settings


@pytest.fixture
def orchestrator():
    """Create orchestrator instance"""
    settings = Settings()
    settings.model.device = "cpu"
    return AIOrchestrator(settings=settings)


@pytest.fixture
def sample_signals():
    """Create sample signals"""
    return [
        {
            "type": "social_media",
            "source": "verified_media",
            "data": {"text": "Test signal 1", "likes": 100},
            "location": {"latitude": 40.7128, "longitude": -74.0060},
            "temporal": {"timestamp": datetime.now().isoformat()}
        },
        {
            "type": "economic",
            "source": "official_government",
            "data": {"gdp": 1000, "inflation": 2.5},
            "location": {"latitude": 40.7128, "longitude": -74.0060},
            "temporal": {"timestamp": datetime.now().isoformat()}
        }
    ]


@pytest.mark.asyncio
async def test_process_pipeline(orchestrator, sample_signals):
    """Test intelligence pipeline processing"""
    result = await orchestrator.process_intelligence_pipeline(sample_signals)
    
    assert result is not None
    assert "assessment" in result
    assert "indicators" in result
    assert "insights" in result
    assert "metadata" in result


@pytest.mark.asyncio
async def test_assessment_structure(orchestrator, sample_signals):
    """Test assessment structure"""
    result = await orchestrator.process_intelligence_pipeline(sample_signals)
    assessment = result["assessment"]
    
    assert "id" in assessment
    assert "threat_level" in assessment
    assert "confidence" in assessment
    assert 0 <= assessment["threat_level"] <= 6
    assert 0 <= assessment["confidence"] <= 1


def test_system_status(orchestrator):
    """Test system status retrieval"""
    status = orchestrator.get_system_status()
    
    assert isinstance(status, dict)
    assert "status" in status
    assert "subsystems" in status
    assert status["status"] == "operational"
