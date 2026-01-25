"""
Core GPT Model Architecture.
Implements a casual decoder-only Transformer from scratch using PyTorch.
Upgraded to Modern Architecture:
- RMSNorm
- SwiGLU
- Rotary Embeddings (RoPE)
- Grouped Query Attention (GQA)
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    freqs_cis: complex tensor of shape (T, d_head/2)
    x: complex tensor of shape (B, T, H, d_head/2)
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    xq: (B, T, H_q, D)
    xk: (B, T, H_k, D)
    freqs_cis: (T, D/2)
    """
    # View as complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast frequencies
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_head
        
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # Reshape for broadcast: (B, T, H, D)
        xq = xq.view(B, T, self.n_head, self.head_dim)
        xk = xk.view(B, T, self.n_kv_head, self.head_dim)
        xv = xv.view(B, T, self.n_kv_head, self.head_dim)
        
        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Grouped Query Attention handling
        # Repeat k/v heads if n_kv_head < n_head
        if self.n_rep > 1:
            # (B, T, H_k, D) -> (B, T, H_k, 1, D) -> (B, T, H_k, Rep, D) -> (B, T, H_q, D)
            xk = xk[:, :, :, None, :].expand(B, T, self.n_kv_head, self.n_rep, self.head_dim).reshape(B, T, self.n_head, self.head_dim)
            xv = xv[:, :, :, None, :].expand(B, T, self.n_kv_head, self.n_rep, self.head_dim).reshape(B, T, self.n_head, self.head_dim)
            
        # Transpose for attention: (B, H, T, D)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            # Manual implementation
            att = (xq @ xk.transpose(-2, -1)) * (1.0 / math.sqrt(xk.size(-1)))
            # Causal mask
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ xv
            
        # Reassemble
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.resid_dropout(self.wo(y))

class MLP(nn.Module):
    """
    SwiGLU MLP.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        # SwiGLU typically uses 2/3 * 4d to maintain parameter count parity or similar,
        # but here we stick to 4x expansion for simplicity or follow Llama's convention?
        # Llama 2: hidden_dim = 4 * n_embd, then int(2 * hidden_dim / 3), then multiple of 256.
        # Let's keep it simple: 4 * n_embd.
        # SwiGLU needs 3 projections.
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.feed_forward = MLP(config)

    def forward(self, x, freqs_cis):
        h = x + self.attn(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Share weights
        self.tok_embeddings.weight = self.output.weight
        
        # Precompute RoPE frequencies
        self.freqs_cis = None
        
        self.apply(self._init_weights)
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0, device='cpu'):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def forward(self, idx, targets=None):
        B, T = idx.size()
        device = idx.device
        
        # Lazy init frequencies if needed or shape changed
        if self.freqs_cis is None or self.freqs_cis.shape[0] < T:
            self.freqs_cis = self.precompute_freqs_cis(
                self.config.n_embd // self.config.n_head, 
                self.config.block_size * 2, # ample buffer
                self.config.rope_theta,
                device
            )
            
        freqs_cis = self.freqs_cis[:T].to(device)
        
        h = self.tok_embeddings(idx)
        
        for layer in self.layers:
            h = layer(h, freqs_cis)
            
        h = self.norm(h)
        
        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.output(h[:, [-1], :])
            loss = None
            
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        extra_args = dict(fused=True) if device_type == 'cuda' and hasattr(torch.optim, 'AdamW') else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
