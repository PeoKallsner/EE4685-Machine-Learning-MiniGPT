"""
src/main.py
-----------
Entry point for the MiniGPT project.

Run this script from the repository root to either train the model or
generate text from a trained checkpoint.

Usage
~~~~~
Train::

    python src/main.py --mode train \\
        --config configs/default_config.yaml configs/model_config.yaml

Generate::

    python src/main.py --mode generate \\
        --config configs/default_config.yaml configs/model_config.yaml \\
        --checkpoint checkpoints/best.pt \\
        --prompt "Once upon a time" \\
        --max_new_tokens 200 \\
        --temperature 0.8

Command-line arguments are parsed with :mod:`argparse` and merged into the
:class:`~src.config.Config` dataclass via :func:`~src.config.load_config`.
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.generate import generate
from src.tokenizer import CharTokenizer
from src.train import train
from src.utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        An :class:`argparse.Namespace` with the parsed arguments.

    TODO:
        - Add ``--mode`` argument: ``"train"`` or ``"generate"``.
        - Add ``--config`` argument (nargs='+') for YAML config file paths.
        - Add ``--checkpoint`` argument for loading a saved model.
        - Add ``--prompt``, ``--max_new_tokens``, ``--temperature``,
          ``--top_k`` arguments for generation mode.
    """
    parser = argparse.ArgumentParser(
        description="MiniGPT — train or generate text with a tiny GPT model."
    )

    parser.add_argument(
        "--mode",
        choices=["train", "generate"],
        required=True,
        help="Whether to train the model or generate text.",
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/default_config.yaml", "configs/model_config.yaml"],
        help="One or more YAML config file paths (later files override earlier ones).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint to load (for generation or resuming training).",
    )

    # Generation-specific arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Seed text for generation mode.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="If set, only sample from the top-k most likely tokens.",
    )

    # TODO: add any additional arguments (e.g. --seed override)

    return parser.parse_args()


def main() -> None:
    """Main entry point: parse args, load config, and dispatch to train/generate.

    TODO:
        1. Call :func:`parse_args` to get the command-line arguments.
        2. Call :func:`~src.config.load_config` with the given YAML paths.
        3. Call :func:`~src.utils.set_seed` with ``config.training.seed``.
        4. If ``args.mode == "train"``, call :func:`~src.train.train`.
        5. If ``args.mode == "generate"``:
           a. Load the tokeniser vocabulary from disk.
           b. Build the model and load the checkpoint.
           c. Call :func:`~src.generate.generate` and print the output.
    """
    args = parse_args()

    # TODO: implement — load config and dispatch
    raise NotImplementedError


if __name__ == "__main__":
    main()
