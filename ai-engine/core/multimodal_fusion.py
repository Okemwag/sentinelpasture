"""
Multimodal Fusion - Advanced fusion of heterogeneous data sources
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
import numpy as np


class MultimodalFusionEngine(nn.Module):
    """
    Advanced multimodal fusion using cross-attention and tensor fusion
    """
    
    def __init__(
        self, 
        modality_dims: Dict[str, int],
        fusion_dim: int = 512,
        n_heads: int = 8
    ):
        super().__init__()
        
        self.modality_dims = modality_dims
        self.fusion_dim = fusion_dim
        
        # Modality-specific encoders
        self.encoders = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            for modality, dim in modality_dims.items()
        })
        
        # Cross-modal attention
        self.cross_attention = nn.ModuleDict({
            f"{m1}_to_{m2}": nn.MultiheadAttention(
                fusion_dim, n_heads, batch_first=True
            )
            for m1 in modality_dims.keys()
            for m2 in modality_dims.keys()
            if m1 != m2
        })
        
        # Tensor fusion
        self.tensor_fusion = TensorFusion(
            len(modality_dims), fusion_dim
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, fusion_dim)
        )
    
    def forward(
        self, 
        modality_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse multiple modalities
        
        Args:
            modality_inputs: Dict mapping modality names to tensors
        
        Returns:
            Fused representation
        """
        # Encode each modality
        encoded = {}
        for modality, x in modality_inputs.items():
            if modality in self.encoders:
                encoded[modality] = self.encoders[modality](x)
        
        # Cross-modal attention
        attended = {}
        for modality in encoded.keys():
            attended[modality] = encoded[modality]
            
            for other_modality in encoded.keys():
                if modality != other_modality:
                    key = f"{modality}_to_{other_modality}"
                    if key in self.cross_attention:
                        attn_out, _ = self.cross_attention[key](
                            encoded[modality].unsqueeze(1),
                            encoded[other_modality].unsqueeze(1),
                            encoded[other_modality].unsqueeze(1)
                        )
                        attended[modality] = attended[modality] + attn_out.squeeze(1)
        
        # Tensor fusion
        modality_list = [attended[m] for m in sorted(attended.keys())]
        fused = self.tensor_fusion(modality_list)
        
        # Output projection
        output = self.output_proj(fused)
        
        return output


class TensorFusion(nn.Module):
    """
    Tensor fusion for combining multiple modalities
    """
    
    def __init__(self, n_modalities: int, dim: int):
        super().__init__()
        
        self.n_modalities = n_modalities
        self.dim = dim
        
        # Fusion weights
        self.fusion_weights = nn.Parameter(
            torch.randn(n_modalities, dim, dim)
        )
    
    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform tensor fusion
        
        Args:
            modalities: List of modality tensors, each (B, D)
        
        Returns:
            Fused tensor (B, D)
        """
        # Start with first modality
        fused = modalities[0]
        
        # Iteratively fuse with other modalities
        for i, modality in enumerate(modalities[1:], 1):
            # Outer product and weighted sum
            outer = torch.einsum('bi,bj->bij', fused, modality)
            fused = torch.einsum('bij,ij->bi', outer, self.fusion_weights[i])
        
        return fused


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion for multi-scale information
    """
    
    def __init__(self, input_dim: int, n_levels: int = 3):
        super().__init__()
        
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // (2 ** i)),
                nn.ReLU(),
                nn.LayerNorm(input_dim // (2 ** i))
            )
            for i in range(n_levels)
        ])
        
        total_dim = sum(input_dim // (2 ** i) for i in range(n_levels))
        self.fusion = nn.Linear(total_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical fusion
        """
        level_outputs = [level(x) for level in self.levels]
        concatenated = torch.cat(level_outputs, dim=-1)
        return self.fusion(concatenated)
