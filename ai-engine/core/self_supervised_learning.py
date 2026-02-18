"""
Self-Supervised Learning - Learn powerful representations without labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ContrastiveLearning(nn.Module):
    """
    Contrastive learning (SimCLR-style) for self-supervised representation learning
    """
    
    def __init__(self, encoder: nn.Module, projection_dim: int = 128, temperature: float = 0.5):
        super().__init__()
        
        self.encoder = encoder
        self.temperature = temperature
        
        # Projection head
        encoder_dim = self._get_encoder_dim()
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def _get_encoder_dim(self) -> int:
        """Get encoder output dimension"""
        dummy_input = torch.randn(1, 448)
        with torch.no_grad():
            output = self.encoder(dummy_input)
        return output.shape[-1]
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with two augmented views
        """
        # Encode
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Project
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Contrastive loss
        loss = self.nt_xent_loss(z1, z2)
        
        return loss
    
    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Normalized Temperature-scaled Cross Entropy Loss
        """
        batch_size = z1.shape[0]
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)  # (2*batch_size, projection_dim)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2*batch_size, 2*batch_size)
        
        # Create labels (positive pairs)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(z.device)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder for self-supervised learning
    """
    
    def __init__(
        self,
        input_dim: int = 448,
        hidden_dim: int = 256,
        mask_ratio: float = 0.75
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with random masking
        
        Returns:
            loss, reconstructed, mask
        """
        batch_size, seq_len = x.shape
        
        # Random masking
        mask = self.random_masking(batch_size, seq_len)
        
        # Apply mask
        x_masked = x.clone()
        x_masked[mask] = 0
        
        # Encode
        encoded = self.encoder(x_masked)
        
        # Replace masked positions with mask token
        encoded[mask] = self.mask_token
        
        # Decode
        reconstructed = self.decoder(encoded)
        
        # Compute loss only on masked positions
        loss = F.mse_loss(reconstructed[mask], x[mask])
        
        return loss, reconstructed, mask
    
    def random_masking(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Generate random mask"""
        n_masked = int(seq_len * self.mask_ratio)
        
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        for i in range(batch_size):
            masked_indices = torch.randperm(seq_len)[:n_masked]
            mask[i, masked_indices] = True
        
        return mask


class MomentumContrastLearning(nn.Module):
    """
    Momentum Contrast (MoCo) for self-supervised learning
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07
    ):
        super().__init__()
        
        self.K = K
        self.m = m
        self.T = T
        
        # Query encoder
        self.encoder_q = encoder
        
        # Key encoder (momentum)
        self.encoder_k = self._create_momentum_encoder(encoder)
        
        # Projection heads
        encoder_dim = self._get_encoder_dim()
        self.projection_q = nn.Linear(encoder_dim, dim)
        self.projection_k = nn.Linear(encoder_dim, dim)
        
        # Queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def _create_momentum_encoder(self, encoder: nn.Module) -> nn.Module:
        """Create momentum encoder"""
        import copy
        encoder_k = copy.deepcopy(encoder)
        
        # No gradients for momentum encoder
        for param in encoder_k.parameters():
            param.requires_grad = False
        
        return encoder_k
    
    def _get_encoder_dim(self) -> int:
        """Get encoder output dimension"""
        dummy_input = torch.randn(1, 448)
        with torch.no_grad():
            output = self.encoder_q(dummy_input)
        return output.shape[-1]
    
    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of key encoder"""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update queue"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace oldest keys
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        
        self.queue_ptr[0] = ptr
    
    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x_q: Query samples
            x_k: Key samples
        """
        # Query features
        q = self.encoder_q(x_q)
        q = self.projection_q(q)
        q = F.normalize(q, dim=1)
        
        # Key features (no gradient)
        with torch.no_grad():
            self._momentum_update()
            
            k = self.encoder_k(x_k)
            k = self.projection_k(k)
            k = F.normalize(k, dim=1)
        
        # Positive logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        
        # Labels (positive is first)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        
        # Loss
        loss = F.cross_entropy(logits, labels)
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return loss


class SelfSupervisedLearningSystem:
    """
    Comprehensive self-supervised learning system
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        method: str = "contrastive",
        device: str = "cpu"
    ):
        self.encoder = encoder
        self.method = method
        self.device = device
        
        # Initialize method
        if method == "contrastive":
            self.model = ContrastiveLearning(encoder).to(device)
        elif method == "mae":
            self.model = MaskedAutoencoderViT().to(device)
        elif method == "moco":
            self.model = MomentumContrastLearning(encoder).to(device)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        logger.info(f"Self-supervised learning initialized with method: {method}")
    
    def train_epoch(self, dataloader: Any) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            if self.method == "contrastive":
                x1, x2 = self._augment_batch(batch)
                loss = self.model(x1, x2)
            elif self.method == "mae":
                x = batch.to(self.device)
                loss, _, _ = self.model(x)
            elif self.method == "moco":
                x_q, x_k = self._augment_batch(batch)
                loss = self.model(x_q, x_k)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _augment_batch(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create two augmented views of batch"""
        # Simple augmentation: add noise
        x1 = batch + torch.randn_like(batch) * 0.1
        x2 = batch + torch.randn_like(batch) * 0.1
        
        return x1.to(self.device), x2.to(self.device)
    
    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """Get learned representations"""
        self.model.eval()
        with torch.no_grad():
            if self.method in ["contrastive", "moco"]:
                representations = self.encoder(x.to(self.device))
            else:
                representations = self.model.encoder(x.to(self.device))
        
        return representations
    
    def fine_tune(
        self,
        downstream_task: nn.Module,
        train_loader: Any,
        epochs: int = 10
    ) -> List[float]:
        """Fine-tune on downstream task"""
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.Adam(downstream_task.parameters(), lr=1e-4)
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Get representations
                with torch.no_grad():
                    features = self.get_representations(x)
                
                # Downstream task
                output = downstream_task(features)
                loss = F.cross_entropy(output, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            losses.append(avg_loss)
            
            logger.info(f"Fine-tune epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")
        
        return losses
