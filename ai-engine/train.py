"""
Training script for the Custom GPT model.
"""

import os
import time
import math
import torch
from model import GPT
from config import ModelConfig
from data import prepare_data, GPTDataset

def train():
    # -----------------------------------------------------------------------------
    # Hyperparameters & Config
    out_dir = 'out'
    eval_interval = 200
    log_interval = 10
    eval_iters = 50
    eval_only = False # if True, script exits after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'scratch' # 'scratch' or 'resume'
    
    # Data configuration (Using dummy shakespeare file path for now)
    dataset_path = 'input.txt' 
    
    # Default Config
    config = ModelConfig()
    
    # Override for smaller training (fast iteration)
    config.n_layer = 4
    config.n_head = 4
    config.n_embd = 128
    config.block_size = 64
    config.batch_size = 16 # Not in config class but used here
    batch_size = 32
    
    learning_rate = 1e-3
    max_iters = 5000
    lr_decay_iters = 5000
    min_lr = 1e-4 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # -----------------------------------------------------------------------------
    
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337)
    
    # Data Setup
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. using dummy text.")
        with open(dataset_path, 'w') as f:
            f.write("Hello world! This is a test dataset for the GPT model. " * 100)
    
    train_data, val_data, tokenizer = prepare_data(dataset_path)
    config.vocab_size = tokenizer.vocab_size
    train_dataset = GPTDataset(train_data, config.block_size)
    val_dataset = GPTDataset(val_data, config.block_size)
    
    # Model Setup
    print(f"Initializing model with config: {config}")
    if init_from == 'scratch':
        model = GPT(config)
    
    model.to(device)
    
    # Optimizer
    optimizer = model.configure_optimizers(config.weight_decay, learning_rate, (config.beta1, config.beta2), device)
    
    # Logging
    iter_num = 0
    best_val_loss = 1e9
    
    # Loss estimation
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, d in [('train', train_dataset), ('val', val_dataset)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = d.get_batch(batch_size, device)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # LR Scheduler
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        warmup_iters = 200
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    print("Starting training...")
    t0 = time.time()
    
    while iter_num < max_iters:
        
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Evaluation and Checkpointing
        if iter_num % eval_interval == 0 and iter_num > 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                    
        # Forward Backward Update
        X, Y = train_dataset.get_batch(batch_size, device)
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        
        # Logging
        if iter_num % log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
            
        iter_num += 1

if __name__ == '__main__':
    train()
