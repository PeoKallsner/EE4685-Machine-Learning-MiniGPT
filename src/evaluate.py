"""
src/evaluate.py
---------------
Evaluation utilities for MiniGPT.

The primary metric for language models is **perplexity**, which measures how
"surprised" the model is by the text.  Lower perplexity = better model.

    Perplexity = exp( average cross-entropy loss )

This module provides:
- :func:`compute_loss` — average cross-entropy loss on a dataset split.
- :func:`compute_perplexity` — perplexity derived from the loss.

Usage example::

    loss = compute_loss(model, val_loader, device)
    ppl  = compute_perplexity(loss)
    print(f"Validation perplexity: {ppl:.2f}")
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import MiniGPT


def compute_loss(
    model: MiniGPT,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Compute the average cross-entropy loss over a dataset split.

    The model is set to evaluation mode (``model.eval()``) before iterating
    over the data loader and restored to training mode afterwards.

    Args:
        model: The MiniGPT model to evaluate.
        data_loader: DataLoader yielding ``(input_ids, targets)`` batches.
        device: The device to run evaluation on.
        max_batches: If provided, only evaluate on this many batches (useful
            for fast approximate evaluation during training).

    Returns:
        The mean cross-entropy loss as a Python float.

    TODO:
        - Set model to eval mode (``model.eval()``).
        - Iterate over *data_loader* within ``torch.no_grad()``.
        - Accumulate the loss from each batch.
        - Restore training mode (``model.train()``) before returning.
    """
    # Set model to eval mode
    model.eval()
    
    total_loss = 0.0
    total_batches = 0
    
    # Iterate over data_loader within torch.no_grad()
    with torch.no_grad():
        for batch_idx, (input_ids, targets) in enumerate(data_loader):
            # Stop if max_batches reached
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            # Move to device
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            # Forward pass (returns logits and loss)
            logits, loss = model(input_ids, targets=targets)
            
            # Accumulate loss
            total_loss += loss.item()
            total_batches += 1
    
    # Restore training mode
    model.train()
    
    # Return average loss
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss


def compute_perplexity(loss: float) -> float:
    """Convert an average cross-entropy loss to perplexity.

    Args:
        loss: Mean cross-entropy loss (nats, i.e. natural-log base).

    Returns:
        Perplexity: ``exp(loss)``.

    TODO: return ``math.exp(loss)``.
    """
    return math.exp(loss)
