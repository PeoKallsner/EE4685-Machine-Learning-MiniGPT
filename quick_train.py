#!/usr/bin/env python3
"""Quick training script for MiniGPT on Shakespeare data."""
import sys
from pathlib import Path
import torch
from dataclasses import dataclass
from src.config import Config, ModelConfig, TrainingConfig, DataConfig, LoggingConfig
from src.train import train

# Create configuration
data_config = DataConfig(
    raw_dir="data/raw",
    processed_dir="data/processed",
)

model_config = ModelConfig(
    block_size=256,
    vocab_size=65,
    n_embd=128,
    n_layer=4,
    n_head=4,
    dropout=0.1,
)

training_config = TrainingConfig(
    batch_size=32,
    max_epochs=2,
    learning_rate=3e-4,
    weight_decay=0.1,
    grad_clip=1.0,
    warmup_steps=0,
    eval_interval=100,
    save_interval=500,
    checkpoint_dir="checkpoints",
    seed=42,
    device="auto",
)

logging_config = LoggingConfig(
    log_interval=100,
)

config = Config(
    data=data_config,
    model=model_config,
    training=training_config,
    logging=logging_config,
)

print("=" * 70)
print("MINIGPT TRAINING - SHAKESPEARE CORPUS")
print("=" * 70)
print(f"\nConfig:")
print(f"  Model: {model_config.n_layer} layers, {model_config.n_head} heads, {model_config.n_embd} embed dim")
print(f"  Vocab size: {model_config.vocab_size}")
print(f"  Block size: {model_config.block_size}")
print(f"\nTraining:")
print(f"  Epochs: {training_config.max_epochs}")
print(f"  Batch size: {training_config.batch_size}")
print(f"  Learning rate: {training_config.learning_rate}")
print(f"  Eval interval: {training_config.eval_interval}")
print(f"  Save interval: {training_config.save_interval}")
print("=" * 70)
print()

try:
    train(config)
    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)
except Exception as e:
    print(f"\n✗ Training failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
