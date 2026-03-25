#!/usr/bin/env python3
"""Continue training with early stopping when validation loss plateaus."""
import sys
from pathlib import Path
import torch
from src.config import Config, ModelConfig, TrainingConfig, DataConfig, LoggingConfig
from src.train import train, load_checkpoint
from src.model import MiniGPT

# Extended training config with early stopping
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
    max_epochs=10,  # Enough for ~3000 total steps
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
print("MINIGPT TRAINING - CONTINUED (WITH EARLY STOPPING)")
print("=" * 70)

# Check latest checkpoint
checkpoint_dir = Path("checkpoints")
checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))

if checkpoints:
    latest_ckpt = checkpoints[-1]
    ckpt_state = torch.load(latest_ckpt, weights_only=False)
    print(f"\nLatest checkpoint: {latest_ckpt.name}")
    print(f"  Step: {ckpt_state['step']}")
    print(f"  Loss: {ckpt_state['loss']:.4f}")
else:
    print("\n❌ No checkpoints found!")
    sys.exit(1)

print(f"\nTraining configuration:")
print(f"  Model: {model_config.n_layer} layers, {model_config.n_head} heads, {model_config.n_embd} embed dim")
print(f"  Epochs: {training_config.max_epochs}")
print(f"  Batch size: {training_config.batch_size}")
print(f"  Learning rate: {training_config.learning_rate}")
print(f"\nTarget:")
print(f"  Validation perplexity: ~8-9")
print(f"  Total steps: 2500-3500 (currently ~{ckpt_state['step']})")
print(f"  Early stopping: 200 steps without improvement")
print("=" * 70)
print()

try:
    train(config)
    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print("=" * 70)
    
    # Show final checkpoint
    new_checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    new_checkpoints.sort(key=lambda x: int(x.stem.split("_")[-1]))
    if new_checkpoints and new_checkpoints[-1] != latest_ckpt:
        final_ckpt = new_checkpoints[-1]
        final_state = torch.load(final_ckpt, weights_only=False)
        print(f"\nFinal checkpoint: {final_ckpt.name}")
        print(f"  Final step: {final_state['step']}")
        print(f"  Final loss: {final_state['loss']:.4f}")
        
except Exception as e:
    print(f"\n✗ Training failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
