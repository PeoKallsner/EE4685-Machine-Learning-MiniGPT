#!/usr/bin/env python3
"""Final training push to reach 1500-2000 steps."""
import sys
from pathlib import Path
import torch
from src.config import Config, ModelConfig, TrainingConfig, DataConfig, LoggingConfig
from src.train import train

# Aggressive training for final push
config = Config(
    data=DataConfig(raw_dir="data/raw", processed_dir="data/processed"),
    model=ModelConfig(
        block_size=256, vocab_size=65, n_embd=128, n_layer=4, n_head=4, dropout=0.1
    ),
    training=TrainingConfig(
        batch_size=32, max_epochs=10, learning_rate=3e-4, weight_decay=0.1,
        grad_clip=1.0, eval_interval=50, save_interval=250,  # More frequent evals
        checkpoint_dir="checkpoints", seed=42, device="auto",
    ),
    logging=LoggingConfig(log_interval=100),
)

print("=" * 70)
print("FINAL TRAINING PUSH (steps 1000 → ~1500+)")
print("=" * 70)
print(f"\nEval interval: {config.training.eval_interval} steps")
print(f"Early stopping patience: 200 steps")
print("Target: Val PPL ~8-9\n")

try:
    train(config)
    print("\n✓ Training complete!")
except KeyboardInterrupt:
    print("\n⏸ Training paused")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Show final checkpoint
ckpts = sorted(Path("checkpoints").glob("checkpoint_step_*.pt"), key=lambda x: int(x.stem.split("_")[-1]))
if ckpts:
    final = torch.load(ckpts[-1], weights_only=False)
    print(f"\nFinal checkpoint: {ckpts[-1].name}")
    print(f"  Step: {final['step']}, Loss: {final['loss']:.4f}")
