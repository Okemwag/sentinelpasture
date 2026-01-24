"""
Configuration configuration for the GPT model.
This module defines the hyperparameters and structural settings for the transformer.
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    ModelConfig holds the hyperparameters for the GPT model.
    """
    block_size: int = 1024  # Maximum context length (time_steps)
    vocab_size: int = 50257 # GPT-2 vocabulary size (default)
    n_layer: int = 12       # Number of transformer blocks
    n_head: int = 12        # Number of attention heads
    n_embd: int = 768       # Embedding dimension
    dropout: float = 0.1    # Dropout probability
    bias: bool = True       # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # Optimization settings
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    def __post_init__(self):
        """
        Validate configuration constraints.
        """
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
