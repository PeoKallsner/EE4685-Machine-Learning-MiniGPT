"""
src/utils.py
------------
Miscellaneous helper utilities for MiniGPT.

This module collects small, reusable functions that don't belong to a
specific larger module.

Contents
~~~~~~~~
- :func:`set_seed` — set random seeds for reproducibility.
- :func:`get_device` — auto-detect the best available device.
- :func:`count_parameters` — count trainable model parameters.
- :func:`format_number` — pretty-print large integers (e.g. 1.3M, 42K).
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.

    TODO:
        - Call ``random.seed(seed)``.
        - Call ``np.random.seed(seed)``.
        - Call ``torch.manual_seed(seed)``.
        - If CUDA is available, call ``torch.cuda.manual_seed_all(seed)``.
    """
    # TODO: implement
    raise NotImplementedError


def get_device(device_str: str = "auto") -> torch.device:
    """Return the best available :class:`torch.device`.

    Args:
        device_str: One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
            ``"auto"`` selects CUDA if available, then MPS (Apple Silicon),
            then falls back to CPU.

    Returns:
        A :class:`torch.device` instance.

    TODO:
        - If ``device_str == "auto"``: check ``torch.cuda.is_available()``
          first, then ``torch.backends.mps.is_available()``.
        - Otherwise return ``torch.device(device_str)``.
    """
    # TODO: implement
    raise NotImplementedError


def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in *model*.

    Args:
        model: Any PyTorch model.

    Returns:
        Integer count of trainable parameters.

    TODO: sum ``p.numel()`` for all ``p`` where ``p.requires_grad``.
    """
    # TODO: implement
    raise NotImplementedError


def format_number(n: int) -> str:
    """Return a human-readable string for a large integer.

    Examples::

        format_number(1_300_000)  # "1.30M"
        format_number(42_000)     # "42.00K"
        format_number(512)        # "512"

    Args:
        n: A non-negative integer.

    Returns:
        A formatted string with a SI-style suffix (K, M, B).

    TODO: implement with threshold checks for billions, millions, thousands.
    """
    # TODO: implement
    raise NotImplementedError
