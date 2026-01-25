"""
Data loading and Tokenization utilities.
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Optional

import tiktoken

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

class TiktokenTokenizer(Tokenizer):
    """
    OpenAI's BPE tokenizer (cl100k_base).
    Used by GPT-4, Llama 3 (compatible range), etc.
    """
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.enc = tiktoken.get_encoding(encoding_name)
    
    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special={'<|endoftext|>'})
    
    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

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
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        else:
             x = x.to(device)
             y = y.to(device)
             
        return x, y

def prepare_data(input_file_path: str, split: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor, Tokenizer]:
    """
    Reads a text file, tokenizes it using tiktoken, and splits into train/val.
    """
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    tokenizer = TiktokenTokenizer()
    print(f"Tokenizing {len(text)} characters...")
    tokens = tokenizer.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)
    
    n = int(split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Data loaded from {input_file_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Total tokens: {len(data):,}")
    print(f"Train size: {len(train_data):,}")
    print(f"Val size: {len(val_data):,}")
    
    return train_data, val_data, tokenizer
