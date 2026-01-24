"""
Generation script for the Custom GPT model.
Loads a checkpoint and generates text based on a prompt.
"""

import os
import torch
from model import GPT, ModelConfig
from data import CharTokenizer

def generate():
    # -----------------------------------------------------------------------------
    out_dir = 'out'
    start = "\n" # or "<|endoftext|>" or etc. Can also specify a string, e.g. "To be or not to be"
    num_samples = 3 # number of samples to draw
    max_new_tokens = 500 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have -inf probability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
    # -----------------------------------------------------------------------------

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}!")
        return

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Initialize model
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model config: {config}")
    
    # Setup tokenizer (needs to match training data! assumes char level for now or simplistic)
    # Ideally should load the tokenizer from a file saved during training
    # For this demo, we re-create the char tokenizer from input.txt if it exists
    # Or just warn.
    if os.path.exists('input.txt'):
         with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
            tokenizer = CharTokenizer(text)
    else:
        # Fallback for demo
        tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n")
        
    # Encode start text
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    
    start_ids = tokenizer.encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # Run generation
    with torch.no_grad():
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print("---------------")
            print(tokenizer.decode(y[0].tolist()))
            print("---------------")

if __name__ == '__main__':
    generate()
