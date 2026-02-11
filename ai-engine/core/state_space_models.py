"""
State Space Models - Advanced sequence modeling with Mamba/S4 architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MambaBlock(nn.Module):
    """
    Mamba: Linear-Time Sequence Modeling with Selective State Spaces
    Efficient alternative to Transformers for long sequences
    """
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=3, padding=1, groups=self.d_inner
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Initialize A (state transition matrix)
        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Initialize D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D) where B=batch, L=length, D=d_model
        """
        B, L, D = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Convolution (time-mixing)
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = F.silu(x)
        
        # SSM
        y = self.selective_scan(x)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output

    def selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective SSM scan - the core of Mamba
        """
        B, L, D = x.shape
        
        # Compute delta (time step)
        delta = F.softplus(self.dt_proj(x))  # (B, L, d_inner)
        
        # Compute B and C (input and output matrices)
        BC = self.x_proj(x)  # (B, L, 2*d_state)
        B, C = BC.chunk(2, dim=-1)  # Each (B, L, d_state)
        
        # Get A
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretize A and B
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # Scan
        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        
        for i in range(L):
            h = deltaA[:, i] * h + deltaB[:, i] * x[:, i].unsqueeze(-1)
            y = torch.sum(h * C[:, i].unsqueeze(1), dim=-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        
        # Add skip connection
        y = y + x * self.D
        
        return y


class S4Layer(nn.Module):
    """
    Structured State Space (S4) layer
    Efficient for very long sequences
    """
    
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # Learnable parameters
        self.A = nn.Parameter(torch.randn(d_model, d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state, 1))
        self.C = nn.Parameter(torch.randn(d_model, 1, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        
        # Apply S4 kernel
        y = self._apply_ssm(x)
        
        # Residual and norm
        return self.norm(x + y)
    
    def _apply_ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply state space model"""
        B, L, D = x.shape
        
        # Initialize state
        h = torch.zeros(B, D, self.d_state, 1, device=x.device)
        
        outputs = []
        for t in range(L):
            # State update: h_t = A @ h_{t-1} + B @ x_t
            h = torch.matmul(self.A, h) + self.B * x[:, t].unsqueeze(-1).unsqueeze(-1)
            
            # Output: y_t = C @ h_t + D @ x_t
            y = torch.matmul(self.C, h).squeeze(-1).squeeze(-1) + self.D * x[:, t]
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


class HybridSSMTransformer(nn.Module):
    """
    Hybrid model combining SSM efficiency with Transformer expressiveness
    """
    
    def __init__(
        self, 
        d_model: int = 512, 
        n_layers: int = 6,
        n_heads: int = 8,
        use_mamba: bool = True
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            HybridLayer(d_model, n_heads, use_mamba) 
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class HybridLayer(nn.Module):
    """Single hybrid layer with SSM and attention"""
    
    def __init__(self, d_model: int, n_heads: int, use_mamba: bool = True):
        super().__init__()
        
        # SSM for efficient long-range dependencies
        if use_mamba:
            self.ssm = MambaBlock(d_model)
        else:
            self.ssm = S4Layer(d_model)
        
        # Local attention for fine-grained patterns
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SSM path
        x = x + self.ssm(self.norm1(x))
        
        # Attention path
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm2(x)
        
        # FFN
        x = x + self.ffn(self.norm3(x))
        
        return x
