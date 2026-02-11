"""
Custom Exceptions for AI Engine
"""


class AIEngineError(Exception):
    """Base exception for AI Engine"""
    pass


class SignalProcessingError(AIEngineError):
    """Error during signal processing"""
    pass


class ThreatAnalysisError(AIEngineError):
    """Error during threat analysis"""
    pass


class ModelError(AIEngineError):
    """Error related to model operations"""
    pass


class ConfigurationError(AIEngineError):
    """Error in configuration"""
    pass


class ValidationError(AIEngineError):
    """Error in input validation"""
    pass


class OptimizationError(AIEngineError):
    """Error during optimization"""
    pass


class InferenceError(AIEngineError):
    """Error during model inference"""
    pass


class DataError(AIEngineError):
    """Error related to data"""
    pass
