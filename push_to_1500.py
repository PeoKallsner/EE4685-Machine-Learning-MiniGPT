#!/usr/bin/env python3
"""Final push to 1500 steps."""
from pathlib import Path
import torch
from src.config import Config, ModelConfig, TrainingConfig, DataConfig, LoggingConfig
from src.train import train

config = Config(
    data=DataConfig(raw_dir="data/raw", processed_dir="data/processed"),
    model=ModelConfig(block_size=256, vocab_size=65, n_embd=128, n_layer=4, n_head=4, dropout=0.1),
    training=TrainingConfig(
        batch_size=32, max_epochs=5, learning_rate=3e-4, weight_decay=0.1,
        grad_clip=1.0, eval_interval=50, save_interval=250,
        checkpoint_dir="checkpoints", seed=42, device="auto",
    ),
    logging=LoggingConfig(log_interval=100),
)

print("=" * 70)
print("TRAINING TO 1500 STEPS")
print("=" * 70)

train(config)

# Show result
ckpts = sorted(Path("checkpoints").glob("checkpoint_step_*.pt"), key=lambda x: int(x.stem.split("_")[-1]))
print(f"\nFinal checkpoint: {ckpts[-1].name}")
final = torch.load(ckpts[-1], weights_only=False)
print(f"Step: {final['step']}, Loss: {final['loss']:.4f}\n")
