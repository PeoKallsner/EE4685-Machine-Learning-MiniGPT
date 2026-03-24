"""
src/train.py
------------
Training loop for MiniGPT.

This module contains the :func:`train` function which runs the main training
loop: forward pass → loss computation → backprop → optimiser step.

It also contains helper utilities for:
- Setting up the optimiser (AdamW).
- Loading and saving checkpoints.
- Running evaluation on the validation set at regular intervals.

Usage example::

    config = load_config("configs/default_config.yaml",
                         "configs/model_config.yaml")
    train(config)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config
from src.model import MiniGPT


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    path: str | Path,
) -> None:
    """Save model weights and optimiser state to a ``.pt`` checkpoint file.

    Args:
        model: The MiniGPT model instance.
        optimizer: The optimiser whose state should be saved.
        step: Current training step number (used for resuming).
        loss: Most recent training loss value.
        path: File path to write the checkpoint to.

    TODO:
        - Build a state dict containing model weights, optimiser state,
          step number, and loss.
        - Use ``torch.save`` to write it to *path*.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: MiniGPT,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """Load a checkpoint and restore model (and optionally optimiser) state.

    Args:
        path: Path to the ``.pt`` checkpoint file.
        model: The model instance to load weights into.
        optimizer: If provided, its state is also restored.

    Returns:
        The training step at which the checkpoint was saved.

    TODO:
        - Use ``torch.load`` to read the checkpoint.
        - Call ``model.load_state_dict`` with the saved weights.
        - If *optimizer* is given, restore its state too.
        - Return the saved step number.
    """
    checkpoint = torch.load(path, weights_only=False)
    
    model.load_state_dict(checkpoint["model"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint["step"]


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------


def build_optimizer(model: MiniGPT, config: Config) -> torch.optim.Optimizer:
    """Create an AdamW optimiser with weight decay.

    Parameters in the model that should **not** be decayed (biases, layer
    norm weights) are placed in a separate parameter group with
    ``weight_decay=0``.

    Args:
        model: The MiniGPT model.
        config: Top-level config (uses ``training.learning_rate`` and
            ``training.weight_decay``).

    Returns:
        A configured :class:`torch.optim.AdamW` optimiser.

    TODO:
        - Separate parameters into "decay" and "no-decay" groups.
        - Instantiate ``torch.optim.AdamW`` with the two groups.
    """
    # Separate parameters into decay and no-decay groups
    decay = set()
    no_decay = set()
    
    for name, param in model.named_parameters():
        # Biases and layer norm weights should not be decayed
        if "bias" in name or "LayerNorm" in name or "ln" in name or "norm" in name:
            no_decay.add(name)
        else:
            decay.add(name)
    
    # Create parameter groups
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay],
            "weight_decay": config.training.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n in no_decay],
            "weight_decay": 0.0,
        },
    ]
    
    # Create AdamW optimizer
    optimizer = torch.optim.AdamW(param_groups, lr=config.training.learning_rate)
    
    return optimizer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(config: Config) -> None:
    """Run the full MiniGPT training loop.

    This is the main entry point for training.  It:

    1. Sets the random seed for reproducibility.
    2. Loads the processed dataset and creates DataLoaders.
    3. Builds the model and moves it to the target device.
    4. Creates the optimiser (and optionally a learning-rate scheduler).
    5. Iterates over mini-batches, computing loss and updating weights.
    6. Periodically evaluates on the validation set.
    7. Saves checkpoints at regular intervals.

    Args:
        config: The merged training + model configuration.

    TODO:
        - Load train and val datasets (``src.dataset.TextDataset``).
        - Instantiate ``MiniGPT`` from ``config.model``.
        - Call :func:`build_optimizer`.
        - Write the training loop with gradient clipping and logging.
        - Call :func:`evaluate` at ``config.training.eval_interval`` steps.
        - Call :func:`save_checkpoint` at ``config.training.save_interval``
          steps.
    """
    from src.utils import set_seed, get_device
    from src.dataset import TextDataset
    
    # Set random seed for reproducibility
    set_seed(config.training.seed)
    
    # Get device
    device = get_device(config.training.device)
    print(f"Using device: {device}")
    
    # Load datasets
    processed_dir = Path(config.data.processed_dir)
    train_ids = torch.load(processed_dir / "train.pt")
    val_ids = torch.load(processed_dir / "val.pt")
    
    train_dataset = TextDataset(train_ids, config.model.block_size)
    val_dataset = TextDataset(val_ids, config.model.block_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    
    # Build model
    model = MiniGPT(config.model).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Build optimizer
    optimizer = build_optimizer(model, config)
    print(f"Optimizer: {optimizer}")
    
    # Training loop
    model.train()
    step = 0
    
    for epoch in range(config.training.max_epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.grad_clip
            )
            
            # Optimizer step
            optimizer.step()
            
            step += 1
            
            # Logging
            if step % config.logging.log_interval == 0:
                print(
                    f"Epoch {epoch + 1}/{config.training.max_epochs} | "
                    f"Step {step} | Loss {loss.item():.4f}"
                )
            
            # Evaluation
            if step % config.training.eval_interval == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x = val_x.to(device)
                        val_y = val_y.to(device)
                        _, batch_loss = model(val_x, val_y)
                        val_loss += batch_loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                print(f"Step {step} | Val Loss {avg_val_loss:.4f}")
                model.train()
            
            # Checkpointing
            if step % config.training.save_interval == 0:
                ckpt_path = (
                    Path(config.training.checkpoint_dir) 
                    / f"checkpoint_step_{step}.pt"
                )
                save_checkpoint(model, optimizer, step, loss.item(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
    
    print("Training complete!")
