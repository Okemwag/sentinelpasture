"""
Core AI Engine Components

This package contains the core components of the AI engine including
signal processing, threat analysis, and advanced learning systems.
"""

from .signal_processor import SignalProcessor
from .threat_analyzer import ThreatAnalyzer, ThreatPatternRecognizer, EscalationPredictor
from .advanced_learning import (
    MetaLearner,
    ReinforcementLearningAgent,
    AdaptiveLearningSystem,
    EnsembleLearner,
    ConceptDriftDetector
)
from .causal_inference import CausalInferenceEngine, StructuralCausalModel
from .quantum_optimizer import QuantumInspiredOptimizer, QuantumNeuralNetwork
from .state_space_models import MambaBlock, S4Layer, HybridSSMTransformer
from .neural_architecture_search import NeuralArchitectureSearch
from .multimodal_fusion import MultimodalFusionEngine, TensorFusion, HierarchicalFusion
from .explainable_ai import ExplainableAI, SaliencyMapper

__all__ = [
    'SignalProcessor',
    'ThreatAnalyzer',
    'ThreatPatternRecognizer',
    'EscalationPredictor',
    'MetaLearner',
    'ReinforcementLearningAgent',
    'AdaptiveLearningSystem',
    'EnsembleLearner',
    'ConceptDriftDetector',
    'CausalInferenceEngine',
    'StructuralCausalModel',
    'QuantumInspiredOptimizer',
    'QuantumNeuralNetwork',
    'MambaBlock',
    'S4Layer',
    'HybridSSMTransformer',
    'NeuralArchitectureSearch',
    'MultimodalFusionEngine',
    'TensorFusion',
    'HierarchicalFusion',
    'ExplainableAI',
    'SaliencyMapper',
]

__version__ = '1.0.0'
