"""
Data loading and Tokenization utilities.
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Optional

class Tokenizer:
    """
    Abstract Tokenizer interface.
    """
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError
    
    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError
    
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

class CharTokenizer(Tokenizer):
    """
    Simple character-level tokenizer.
    Good for testing and small datasets (Shakespeare).
    """
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size_val = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
    
    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.itos[i] for i in tokens])

    @property
    def vocab_size(self) -> int:
        return self.vocab_size_val

class GPTDataset:
    """
    Simple memory-mapped dataset for training.
    """
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
        
    def get_batch(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a small batch of data of inputs x and targets y.
        """
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        
        if device == 'cuda':
            # pinned memory move to gpu async could be optimized here
            x = x.to(device)
            y = y.to(device)
        else:
             x = x.to(device)
             y = y.to(device)
             
        return x, y

def prepare_data(input_file_path: str, split: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor, Tokenizer]:
    """
    Reads a text file, tokenizes it (char level for now), and splits into train/val.
    """
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    n = int(split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Data loaded from {input_file_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Train size: {len(train_data):,}")
    print(f"Val size: {len(val_data):,}")
    
    return train_data, val_data, tokenizer
