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
    vocab_size: int = 100277 # tiktoken cl100k_base vocab size + padding
    n_layer: int = 12       # Number of transformer blocks
    n_head: int = 12        # Number of attention heads
    n_embd: int = 768       # Embedding dimension
    n_kv_head: int = 4      # Number of key/value heads (GQA)
    dropout: float = 0.1    # Dropout probability
    bias: bool = False      # False: a bit better and faster (usually True for LoRA/adapters but we are scratching)
    
    # Modern Arch params
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

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
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
