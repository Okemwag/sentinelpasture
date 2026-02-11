"""
Abstract Interfaces and Protocols

Defines contracts for core components to enable dependency injection
and loose coupling.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch


class ISignalProcessor(ABC):
    """Interface for signal processors"""
    
    @abstractmethod
    async def ingest_signal(
        self,
        signal_type: str,
        source: str,
        raw_data: Dict[str, Any],
        location: Dict[str, Any],
        temporal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a raw signal"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        pass


class IThreatAnalyzer(ABC):
    """Interface for threat analyzers"""
    
    @abstractmethod
    async def analyze_signals(
        self, signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze signals for threats"""
        pass


class ILearningSystem(ABC):
    """Interface for learning systems"""
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction"""
        pass
    
    @abstractmethod
    def update(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Update model with new data"""
        pass


class IOptimizer(ABC):
    """Interface for optimizers"""
    
    @abstractmethod
    def optimize(
        self,
        objective_function: Any,
        bounds: List[Any]
    ) -> Dict[str, Any]:
        """Optimize objective function"""
        pass


class IExplainer(ABC):
    """Interface for explainability systems"""
    
    @abstractmethod
    def explain_prediction(
        self,
        input_data: torch.Tensor,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Explain a prediction"""
        pass


class ICausalEngine(ABC):
    """Interface for causal inference"""
    
    @abstractmethod
    def discover_causal_structure(
        self,
        data: Any,
        variable_names: List[str]
    ) -> Dict[str, List[str]]:
        """Discover causal structure"""
        pass
    
    @abstractmethod
    def estimate_intervention_effect(
        self,
        data: Any,
        treatment_var: str,
        outcome_var: str,
        confounders: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Estimate intervention effect"""
        pass
