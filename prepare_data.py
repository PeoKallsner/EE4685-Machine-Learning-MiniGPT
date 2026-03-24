#!/usr/bin/env python3
"""Prepare data splits from raw text for training."""
from pathlib import Path
import torch
from src.tokenizer import CharTokenizer
from src.dataset import TextDataset

# Read raw text
raw_path = Path("data/raw/input.txt")
text = raw_path.read_text()
print(f"Text loaded: {len(text):,} characters")
print(f"Preview: {text[:100]!r}\n")

# Create tokenizer and build vocabulary
tokenizer = CharTokenizer()
tokenizer.build_vocab(text)
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Sample chars: {list(tokenizer.char_to_id.keys())[:20]}")
print(f"<UNK> token at ID: {tokenizer.unk_id}\n")

# Prepare splits (80/10/10)
print("Preparing data splits...")
TextDataset.prepare_splits(
    raw_text_path="data/raw/input.txt",
    processed_dir="data/processed",
    tokenizer=tokenizer,
    train_ratio=0.8,
    val_ratio=0.1
)

# Check what was created
processed_dir = Path("data/processed")
print("\nCreated files:")
for f in sorted(processed_dir.glob("*.pt")):
    data = torch.load(f, weights_only=True)
    print(f"  {f.name}: shape {data.shape}, dtype {data.dtype}")

print("\n✓ Data prepared! Ready to train.")
