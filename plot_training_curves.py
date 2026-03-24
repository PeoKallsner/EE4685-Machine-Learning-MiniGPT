"""
plot_training_curves.py
-----------------------
Plot training and validation loss curves from checkpoint logs.

This script extracts and visualizes training metrics from checkpoint files
to show loss/perplexity evolution during training.

Usage::

    python plot_training_curves.py --checkpoint_dir checkpoints/ --output plots/curves.png
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def extract_training_metrics(checkpoint_dir: str | Path) -> Tuple[List[int], List[float]]:
    """Extract training steps and losses from checkpoint files.

    Args:
        checkpoint_dir: Directory containing checkpoint files named 
            'checkpoint_step_*.pt'.

    Returns:
        Tuple of (steps, losses) where steps are integers and losses are floats.
    """
    checkpoint_dir = Path(checkpoint_dir)
    metrics = {}
    
    # Load all checkpoints and extract step numbers and losses
    for ckpt_file in sorted(checkpoint_dir.glob("checkpoint_step_*.pt")):
        step = int(ckpt_file.stem.split("_")[-1])
        checkpoint = torch.load(ckpt_file, weights_only=False)
        loss = checkpoint["loss"]
        metrics[step] = loss
    
    if not metrics:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    steps = sorted(metrics.keys())
    losses = [metrics[s] for s in steps]
    
    return steps, losses


def plot_training_curves(
    steps: List[int],
    train_losses: List[float],
    val_losses: List[float] | None = None,
    output_path: str | Path = "training_curves.png",
    title: str = "Training Loss",
) -> None:
    """Plot training curves.

    Args:
        steps: List of training step numbers.
        train_losses: List of training loss values.
        val_losses: Optional list of validation loss values.
        output_path: Path to save the plot.
        title: Plot title.
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(steps, train_losses, linewidth=2, label="Training Loss", marker="o")
    
    if val_losses is not None:
        plt.plot(steps, val_losses, linewidth=2, label="Validation Loss", marker="s")
    
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_perplexity(
    steps: List[int],
    losses: List[float],
    output_path: str | Path = "perplexity_curve.png",
) -> None:
    """Plot perplexity (exp(loss)) curves.

    Args:
        steps: List of training step numbers.
        losses: List of loss values.
        output_path: Path to save the plot.
    """
    perplexities = [np.exp(loss) for loss in losses]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, perplexities, linewidth=2, marker="o", color="orangered")
    
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title("Model Perplexity During Training", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Perplexity plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves from checkpoint logs"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/training_curves.png",
        help="Output path for the plot",
    )
    parser.add_argument(
        "--perplexity",
        action="store_true",
        help="Also plot perplexity curve",
    )
    
    args = parser.parse_args()
    
    print(f"Loading checkpoints from {args.checkpoint_dir}...")
    steps, losses = extract_training_metrics(args.checkpoint_dir)
    
    print(f"Found {len(steps)} checkpoints")
    print(f"Loss range: {min(losses):.4f} - {max(losses):.4f}")
    
    print(f"Plotting training curves to {args.output}...")
    plot_training_curves(steps, losses, output_path=args.output)
    
    if args.perplexity:
        perp_path = Path(args.output).parent / "perplexity_curve.png"
        print(f"Plotting perplexity curve to {perp_path}...")
        plot_perplexity(steps, losses, output_path=perp_path)
    
    print("✓ Done!")


if __name__ == "__main__":
    main()
