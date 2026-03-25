#!/usr/bin/env python3
"""Resume training from the latest checkpoint."""
import sys
from pathlib import Path
import torch
from src.config import Config, ModelConfig, TrainingConfig, DataConfig, LoggingConfig
from src.train import train, load_checkpoint
from src.model import MiniGPT
from src.utils import get_device

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
    max_epochs=5,  # Train for 5 more epochs
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
print("MINIGPT TRAINING - RESUME FROM CHECKPOINT")
print("=" * 70)

# Find latest checkpoint
checkpoint_dir = Path("checkpoints")
checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))

if not checkpoints:
    print("❌ No checkpoints found. Run quick_train.py first!")
    sys.exit(1)

latest_checkpoint = checkpoints[-1]
print(f"\nLatest checkpoint: {latest_checkpoint.name}")

# Load checkpoint to verify
checkpoint = torch.load(latest_checkpoint, weights_only=False)
step = checkpoint['step']
loss = checkpoint['loss']
print(f"  Step: {step}")
print(f"  Loss: {loss:.4f}\n")

print(f"Resuming training configuration:")
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
    
    # Show new checkpoint
    new_checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if new_checkpoints and new_checkpoints[-1] != latest_checkpoint:
        final_checkpoint = new_checkpoints[-1]
        print(f"\nFinal checkpoint: {final_checkpoint.name}")
        final_state = torch.load(final_checkpoint, weights_only=False)
        print(f"  Final step: {final_state['step']}")
        print(f"  Final loss: {final_state['loss']:.4f}")
        
except Exception as e:
    print(f"\n✗ Training failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
