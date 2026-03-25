#!/usr/bin/env python3
"""Evaluate the latest checkpoint and generate samples."""
import torch
from pathlib import Path
from src.config import ModelConfig
from src.model import MiniGPT
from src.tokenizer import CharTokenizer
from src.generate import generate
from src.evaluate import compute_loss, compute_perplexity
from src.dataset import TextDataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Find latest checkpoint  
checkpoint_dir = Path("checkpoints")
checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))

# Sort by step number (extract from filename)
checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
latest_ckpt = checkpoints[-1]

print(f"Loading: {latest_ckpt.name}\n")

# Load checkpoint
checkpoint = torch.load(latest_ckpt, weights_only=False)
train_step = checkpoint['step']
train_loss = checkpoint['loss']

# Set up model
model_config = ModelConfig(
    block_size=256,
    vocab_size=65,
    n_embd=128,
    n_layer=4,
    n_head=4,
    dropout=0.1,
)

# Load tokenizer
tok = CharTokenizer()
text = Path("data/raw/input.txt").read_text()
tok.build_vocab(text)

# Create and load model
model = MiniGPT(model_config).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

print("=" * 70)
print("CHECKPOINT METRICS")
print("=" * 70)
print(f"Step: {train_step}")
print(f"Training loss: {train_loss:.4f}\n")

print("=" * 70)
print("GENERATED SAMPLES (temperature=0.8, top_k=40)")
print("=" * 70)

prompts = ["ROMEO:", "The king", "To be"]
for prompt in prompts:
    print(f"\n'{prompt}'")
    print("-" * 50)
    generated = generate(model, tok, prompt, max_new_tokens=100, temperature=0.8, top_k=40, device=device)
    print(generated)

# Validation evaluation
print("\n" + "=" * 70)
print("VALIDATION METRICS")
print("=" * 70)

val_ids = torch.load(Path("data/processed/val.pt"))
val_dataset = TextDataset(val_ids, model_config.block_size)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Evaluating on {len(val_dataset)} validation samples...")
val_loss = compute_loss(model, val_loader, device, max_batches=100)
val_ppl = compute_perplexity(val_loss)

print(f"Validation loss: {val_loss:.4f}")
print(f"Validation perplexity: {val_ppl:.2f}")

print("\n✓ Done!\n")
