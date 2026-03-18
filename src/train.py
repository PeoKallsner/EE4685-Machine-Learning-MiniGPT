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
    # TODO: implement
    raise NotImplementedError


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
    # TODO: implement
    raise NotImplementedError


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
    # TODO: implement
    raise NotImplementedError


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
    # TODO: implement
    raise NotImplementedError
