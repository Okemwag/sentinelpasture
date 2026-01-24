# AI Engine - Custom GPT

This directory contains a "from scratch" implementation of a GPT-style Transformer language model using PyTorch.

## Structure
- `model.py`: The `GPT` model definition, including `CausalSelfAttention`, `MLP`, and `Block`.
- `train.py`: The training loop, data loading, and checkpointing logic.
- `data.py`: Tokenizer (`CharTokenizer`) and `GPTDataset` implementation.
- `config.py`: `ModelConfig` dataclass for hyperparameters.
- `generate.py`: Inference script to generate text from a trained checkpoint.

## Installation

Requires PyTorch.

```bash
pip install torch numpy
```

## Usage

### 1. Training
Run the training script to train the model on `input.txt`. If `input.txt` does not exist, it will create a dummy dataset.

```bash
python train.py
```

This will save checkpoints to the `out/` directory.

### 2. Generation
Run the generation script to sample text from the trained model.

```bash
python generate.py
```
