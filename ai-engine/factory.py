"""
Factory Pattern for Component Creation

Provides centralized creation of AI engine components with proper
dependency injection and configuration.
"""

from typing import Optional
import torch

from .config import Settings, get_settings
from .core import (
    SignalProcessor,
    ThreatAnalyzer,
    MetaLearner,
    ReinforcementLearningAgent,
    AdaptiveLearningSystem,
    EnsembleLearner,
    CausalInferenceEngine,
    QuantumInspiredOptimizer,
    HybridSSMTransformer,
    NeuralArchitectureSearch,
    MultimodalFusionEngine,
    ExplainableAI
)
from .utils import get_logger

logger = get_logger(__name__)


class ComponentFactory:
    """Factory for creating AI engine components"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        logger.info("ComponentFactory initialized")
    
    def create_signal_processor(self) -> SignalProcessor:
        """Create signal processor"""
        return SignalProcessor(
            embedding_model=self.settings.model.embedding_model
        )
    
    def create_threat_analyzer(self) -> ThreatAnalyzer:
        """Create threat analyzer"""
        return ThreatAnalyzer(
            device=self.settings.model.device
        )
    
    def create_meta_learner(
        self, 
        input_dim: int = 448, 
        hidden_dim: int = 256
    ) -> MetaLearner:
        """Create meta-learner"""
        if not self.settings.learning.meta_learning_enabled:
            logger.warning("Meta-learning is disabled in settings")
        
        return MetaLearner(input_dim=input_dim, hidden_dim=hidden_dim)
    
    def create_rl_agent(
        self,
        state_dim: int = 448,
        action_dim: int = 10
    ) -> ReinforcementLearningAgent:
        """Create RL agent"""
        if not self.settings.learning.reinforcement_learning_enabled:
            logger.warning("Reinforcement learning is disabled in settings")
        
        return ReinforcementLearningAgent(
            state_dim=state_dim,
            action_dim=action_dim
        )
    
    def create_adaptive_system(
        self,
        input_dim: int = 448,
        output_dim: int = 8
    ) -> AdaptiveLearningSystem:
        """Create adaptive learning system"""
        if not self.settings.learning.adaptive_learning_enabled:
            logger.warning("Adaptive learning is disabled in settings")
        
        return AdaptiveLearningSystem(
            input_dim=input_dim,
            output_dim=output_dim
        )
    
    def create_ensemble(
        self,
        input_dim: int = 448,
        output_dim: int = 8
    ) -> EnsembleLearner:
        """Create ensemble learner"""
        return EnsembleLearner(
            n_models=self.settings.learning.ensemble_size,
            input_dim=input_dim,
            output_dim=output_dim
        )
    
    def create_causal_engine(self) -> CausalInferenceEngine:
        """Create causal inference engine"""
        return CausalInferenceEngine()
    
    def create_quantum_optimizer(self) -> QuantumInspiredOptimizer:
        """Create quantum optimizer"""
        if not self.settings.optimization.quantum_enabled:
            logger.warning("Quantum optimization is disabled in settings")
        
        return QuantumInspiredOptimizer(
            n_qubits=self.settings.optimization.n_qubits,
            n_iterations=self.settings.optimization.n_iterations
        )
    
    def create_ssm_model(
        self,
        d_model: int = 512,
        n_layers: int = 6
    ) -> HybridSSMTransformer:
        """Create state space model"""
        return HybridSSMTransformer(
            d_model=d_model,
            n_layers=n_layers
        )
    
    def create_nas(
        self,
        input_dim: int = 448,
        output_dim: int = 8
    ) -> NeuralArchitectureSearch:
        """Create neural architecture search"""
        return NeuralArchitectureSearch(
            input_dim=input_dim,
            output_dim=output_dim
        )
    
    def create_multimodal_fusion(
        self,
        modality_dims: Optional[dict] = None
    ) -> MultimodalFusionEngine:
        """Create multimodal fusion engine"""
        if modality_dims is None:
            modality_dims = {
                "text": 384,
                "numeric": 64,
                "spatial": 32
            }
        
        return MultimodalFusionEngine(
            modality_dims=modality_dims,
            fusion_dim=512
        )
    
    def create_explainer(self, model: torch.nn.Module) -> ExplainableAI:
        """Create explainability system"""
        return ExplainableAI(model)


# Global factory instance
_factory: Optional[ComponentFactory] = None


def get_factory() -> ComponentFactory:
    """Get global factory instance"""
    global _factory
    
    if _factory is None:
        _factory = ComponentFactory()
    
    return _factory


def set_factory(factory: ComponentFactory) -> None:
    """Set global factory instance"""
    global _factory
    _factory = factory
