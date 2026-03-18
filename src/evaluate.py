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
    # TODO: implement
    raise NotImplementedError


def compute_perplexity(loss: float) -> float:
    """Convert an average cross-entropy loss to perplexity.

    Args:
        loss: Mean cross-entropy loss (nats, i.e. natural-log base).

    Returns:
        Perplexity: ``exp(loss)``.

    TODO: return ``math.exp(loss)``.
    """
    # TODO: implement
    raise NotImplementedError
