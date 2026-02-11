"""
AI Engine - Advanced Intelligence System

A powerful, state-of-the-art AI engine combining multiple advanced architectures
and techniques for comprehensive threat analysis, decision support, and predictive intelligence.

Usage:
    from ai_engine import AIOrchestrator, Settings
    
    # Create orchestrator with custom settings
    settings = Settings()
    orchestrator = AIOrchestrator(settings=settings)
    
    # Process signals
    result = await orchestrator.process_intelligence_pipeline(raw_signals)
"""

from .orchestrator import AIOrchestrator
from .config import Settings, get_settings
from .factory import ComponentFactory, get_factory
from .exceptions import (
    AIEngineError,
    SignalProcessingError,
    ThreatAnalysisError,
    ModelError,
    ConfigurationError,
    ValidationError
)
from .utils import setup_logging, get_logger

__version__ = '1.0.0'
__author__ = 'AI Engine Team'

__all__ = [
    'AIOrchestrator',
    'Settings',
    'get_settings',
    'ComponentFactory',
    'get_factory',
    'AIEngineError',
    'SignalProcessingError',
    'ThreatAnalysisError',
    'ModelError',
    'ConfigurationError',
    'ValidationError',
    'setup_logging',
    'get_logger',
]
