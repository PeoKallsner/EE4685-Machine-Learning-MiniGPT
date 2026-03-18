"""
src/config.py
-------------
Defines the project configuration as a Python dataclass.

The dataclass is populated from a YAML file (see ``configs/``) via the
:func:`load_config` helper.  Using a typed dataclass gives us IDE
auto-complete and makes it easy to validate settings early.

Usage example::

    config = load_config("configs/default_config.yaml",
                         "configs/model_config.yaml")
    print(config.model.n_layer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Settings related to data loading and storage."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    train_file: str = "train.txt"
    val_file: str = "val.txt"
    test_file: str = "test.txt"
    val_split: float = 0.1

    # TODO: add any additional data-related fields here


@dataclass
class ModelConfig:
    """Hyperparameters that define the MiniGPT architecture."""

    vocab_size: Optional[int] = None  # Set automatically from tokeniser
    block_size: int = 256
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1
    ffn_multiplier: int = 4
    bias: bool = False

    # TODO: add any additional architecture hyperparameters here


@dataclass
class TrainingConfig:
    """Hyperparameters used during the training loop."""

    batch_size: int = 32
    max_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    device: str = "auto"

    # TODO: add scheduler type, etc.


@dataclass
class LoggingConfig:
    """Settings for logging training progress."""

    log_dir: str = "logs"
    log_interval: int = 100

    # TODO: add W&B / TensorBoard integration flags


@dataclass
class Config:
    """Top-level configuration object combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(*yaml_paths: str | Path) -> Config:
    """Load and merge one or more YAML config files into a :class:`Config`.

    Later files override keys from earlier files, allowing a model-specific
    config to override values set in the default config.

    Args:
        *yaml_paths: Paths to YAML configuration files.  At least one path
            must be provided.

    Returns:
        A fully populated :class:`Config` dataclass instance.

    Example::

        config = load_config("configs/default_config.yaml",
                             "configs/model_config.yaml")
    """
    # TODO: implement YAML merging and dataclass population
    raise NotImplementedError("load_config is not yet implemented")
