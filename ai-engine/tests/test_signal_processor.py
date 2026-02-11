"""
Tests for Signal Processor
"""

import pytest
import numpy as np
from datetime import datetime

from ai_engine.core import SignalProcessor
from ai_engine.exceptions import SignalProcessingError


@pytest.fixture
def signal_processor():
    """Create signal processor instance"""
    return SignalProcessor(embedding_model="all-MiniLM-L6-v2")


@pytest.fixture
def sample_signal():
    """Create sample signal"""
    return {
        "type": "social_media",
        "source": "verified_media",
        "data": {
            "text": "Test signal content",
            "likes": 100,
            "shares": 50
        },
        "location": {
            "latitude": 40.7128,
            "longitude": -74.0060,
            "region": "New York"
        },
        "temporal": {
            "timestamp": datetime.now().isoformat()
        }
    }


@pytest.mark.asyncio
async def test_ingest_signal(signal_processor, sample_signal):
    """Test signal ingestion"""
    result = await signal_processor.ingest_signal(
        signal_type=sample_signal["type"],
        source=sample_signal["source"],
        raw_data=sample_signal["data"],
        location=sample_signal["location"],
        temporal=sample_signal["temporal"]
    )
    
    assert result is not None
    assert "id" in result
    assert "embeddings" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1


@pytest.mark.asyncio
async def test_signal_normalization(signal_processor, sample_signal):
    """Test signal normalization"""
    result = await signal_processor.ingest_signal(
        signal_type=sample_signal["type"],
        source=sample_signal["source"],
        raw_data=sample_signal["data"],
        location=sample_signal["location"],
        temporal=sample_signal["temporal"]
    )
    
    assert "processed_data" in result
    assert "features" in result


def test_get_stats(signal_processor):
    """Test statistics retrieval"""
    stats = signal_processor.get_stats()
    
    assert isinstance(stats, dict)
    assert "total_processed" in stats
    assert "anomalies_detected" in stats
