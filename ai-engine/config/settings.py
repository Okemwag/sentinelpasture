"""
Settings and Configuration Management
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os
import json
from pathlib import Path


@dataclass
class ModelSettings:
    """Model-specific settings"""
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    max_sequence_length: int = 512


@dataclass
class LearningSettings:
    """Learning system settings"""
    meta_learning_enabled: bool = True
    reinforcement_learning_enabled: bool = True
    adaptive_learning_enabled: bool = True
    ensemble_size: int = 5
    learning_rate: float = 1e-4


@dataclass
class OptimizationSettings:
    """Optimization settings"""
    quantum_enabled: bool = True
    n_qubits: int = 10
    n_iterations: int = 1000
    cooling_rate: float = 0.995


@dataclass
class ProcessingSettings:
    """Signal processing settings"""
    max_signal_buffer_size: int = 10000
    anomaly_threshold: float = 0.7
    confidence_threshold: float = 0.5
    batch_processing_enabled: bool = True


@dataclass
class Settings:
    """Main settings container"""
    model: ModelSettings = field(default_factory=ModelSettings)
    learning: LearningSettings = field(default_factory=LearningSettings)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    
    debug: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'Settings':
        """Create settings from dictionary"""
        return cls(
            model=ModelSettings(**config.get('model', {})),
            learning=LearningSettings(**config.get('learning', {})),
            optimization=OptimizationSettings(**config.get('optimization', {})),
            processing=ProcessingSettings(**config.get('processing', {})),
            debug=config.get('debug', False),
            log_level=config.get('log_level', 'INFO')
        )
    
    @classmethod
    def from_file(cls, path: str) -> 'Settings':
        """Load settings from JSON file"""
        with open(path, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'model': {
                'embedding_model': self.model.embedding_model,
                'device': self.model.device,
                'batch_size': self.model.batch_size,
                'max_sequence_length': self.model.max_sequence_length,
            },
            'learning': {
                'meta_learning_enabled': self.learning.meta_learning_enabled,
                'reinforcement_learning_enabled': self.learning.reinforcement_learning_enabled,
                'adaptive_learning_enabled': self.learning.adaptive_learning_enabled,
                'ensemble_size': self.learning.ensemble_size,
                'learning_rate': self.learning.learning_rate,
            },
            'optimization': {
                'quantum_enabled': self.optimization.quantum_enabled,
                'n_qubits': self.optimization.n_qubits,
                'n_iterations': self.optimization.n_iterations,
                'cooling_rate': self.optimization.cooling_rate,
            },
            'processing': {
                'max_signal_buffer_size': self.processing.max_signal_buffer_size,
                'anomaly_threshold': self.processing.anomaly_threshold,
                'confidence_threshold': self.processing.confidence_threshold,
                'batch_processing_enabled': self.processing.batch_processing_enabled,
            },
            'debug': self.debug,
            'log_level': self.log_level,
        }
    
    def save(self, path: str) -> None:
        """Save settings to JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    
    if _settings is None:
        # Try to load from environment variable
        config_path = os.getenv('AI_ENGINE_CONFIG')
        
        if config_path and os.path.exists(config_path):
            _settings = Settings.from_file(config_path)
        else:
            _settings = Settings()
    
    return _settings


def set_settings(settings: Settings) -> None:
    """Set global settings instance"""
    global _settings
    _settings = settings
