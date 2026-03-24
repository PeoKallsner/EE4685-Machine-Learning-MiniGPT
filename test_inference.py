#!/usr/bin/env python3
"""Test generation and evaluation with a trained checkpoint."""
import torch
from pathlib import Path
from src.config import ModelConfig
from src.model import MiniGPT
from src.tokenizer import CharTokenizer
from src.generate import generate
from src.evaluate import compute_loss, compute_perplexity
from src.dataset import TextDataset
from torch.utils.data import DataLoader

# Load checkpoint
checkpoint_path = Path("checkpoints/checkpoint_step_500.pt")
if not checkpoint_path.exists():
    print(f"❌ Checkpoint not found at {checkpoint_path}")
    exit(1)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}\n")

# Load checkpoint to get training step and loss
checkpoint = torch.load(checkpoint_path, weights_only=False)
train_step = checkpoint['step']
train_loss = checkpoint['loss']
print(f"Loaded checkpoint from step {train_step}")
print(f"Training loss at checkpoint: {train_loss:.4f}\n")

# Set up model config
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
print(f"Vocabulary size: {tok.vocab_size}\n")

# Create and load model
model = MiniGPT(model_config).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

print("=" * 70)
print("SAMPLE TEXT GENERATION")
print("=" * 70)

# Generate from different prompts
prompts = [
    "ROMEO:",
    "The king",
    "To be",
]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    print("-" * 50)
    generated = generate(model, tok, prompt, max_new_tokens=150, temperature=0.8, top_k=40, device=device)
    print(generated[:300])  # Show first 300 chars
    print("[...]" if len(generated) > 300 else "")

# Evaluate on validation set
print("\n" + "=" * 70)
print("VALIDATION EVALUATION")
print("=" * 70)

val_ids = torch.load(Path("data/processed/val.pt"))
val_dataset = TextDataset(val_ids, model_config.block_size)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"\nValidation set size: {len(val_dataset)} samples")
print("Computing loss...")

val_loss = compute_loss(model, val_loader, device, max_batches=50)
val_ppl = compute_perplexity(val_loss)

print(f"Validation loss: {val_loss:.4f}")
print(f"Validation perplexity: {val_ppl:.2f}")

print("\n" + "=" * 70)
print("✓ Generation and evaluation complete!")
print("=" * 70)
