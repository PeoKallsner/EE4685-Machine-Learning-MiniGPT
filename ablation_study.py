"""
ablation_study.py
-----------------
Run multiple training configurations for hyperparameter ablation studies.

This script trains MiniGPT with different hyperparameter settings and
compares the results to understand the impact of each parameter.

Examples of ablations:
- Model size (n_embd, n_layer, n_head)
- Learning rate
- Batch size
- Dropout rates
- Weight decay

Usage::

    python ablation_study.py --configs ablation_configs.yaml --output ablation_results.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from src.config import Config, ModelConfig, TrainingConfig
from src.train import train, build_optimizer
from src.model import MiniGPT
from src.dataset import TextDataset
from src.utils import set_seed, get_device, format_number


def create_ablation_configs() -> List[Dict]:
    """Generate a set of configurations for ablation study.

    Returns:
        List of configuration dictionaries, each defining one experiment.
    """
    base_config = {
        "vocab_size": 65,
        "block_size": 256,
        "n_embd": 128,
        "n_layer": 4,
        "n_head": 4,
        "dropout": 0.1,
        "ffn_multiplier": 4,
        "learning_rate": 3e-4,
        "batch_size": 32,
        "weight_decay": 0.1,
        "max_epochs": 2,  # Short for quick ablation
    }
    
    configs = []
    
    # Ablation 1: Model size (embedding dimension)
    for n_embd in [64, 128, 256]:
        config = base_config.copy()
        config["n_embd"] = n_embd
        config["name"] = f"embd_{n_embd}"
        configs.append(config)
    
    # Ablation 2: Number of layers
    for n_layer in [2, 4, 6]:
        config = base_config.copy()
        config["n_layer"] = n_layer
        config["name"] = f"layer_{n_layer}"
        configs.append(config)
    
    # Ablation 3: Learning rate
    for lr in [1e-4, 3e-4, 1e-3]:
        config = base_config.copy()
        config["learning_rate"] = lr
        config["name"] = f"lr_{lr:.0e}"
        configs.append(config)
    
    # Ablation 4: Dropout
    for dropout in [0.0, 0.1, 0.2]:
        config = base_config.copy()
        config["dropout"] = dropout
        config["name"] = f"dropout_{dropout}"
        configs.append(config)
    
    # Ablation 5: Batch size
    for batch_size in [16, 32, 64]:
        config = base_config.copy()
        config["batch_size"] = batch_size
        config["name"] = f"batch_{batch_size}"
        configs.append(config)
    
    return configs


def run_single_experiment(
    config_dict: Dict,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    device: torch.device,
) -> Dict:
    """Run a single training experiment and return metrics.

    Args:
        config_dict: Configuration dictionary with hyperparameters.
        train_ids: Training token IDs.
        val_ids: Validation token IDs.
        device: Device to train on.

    Returns:
        Dictionary with experiment results (name, hyperparams, final_loss, etc).
    """
    from torch.utils.data import DataLoader
    
    config_name = config_dict.pop("name")
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_ids, config_dict["block_size"])
    val_dataset = TextDataset(val_ids, config_dict["block_size"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_dict["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config_dict["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    
    # Build model config
    model_config = ModelConfig(
        vocab_size=config_dict["vocab_size"],
        block_size=config_dict["block_size"],
        n_embd=config_dict["n_embd"],
        n_layer=config_dict["n_layer"],
        n_head=config_dict["n_head"],
        dropout=config_dict["dropout"],
        ffn_multiplier=config_dict.get("ffn_multiplier", 4),
    )
    
    # Create model and move to device
    model = MiniGPT(model_config).to(device)
    
    # Build optimizer
    class DummyConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    training_config = DummyConfig(learning_rate=config_dict["learning_rate"], weight_decay=config_dict["weight_decay"])
    dummy_config = DummyConfig(training=training_config)
    
    optimizer = build_optimizer(model, dummy_config)
    
    # Simple training loop
    model.train()
    best_val_loss = float("inf")
    final_train_loss = 0.0
    
    for epoch in range(config_dict["max_epochs"]):
        train_loss_sum = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_sum += loss.item()
            final_train_loss = loss.item()
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                _, loss = model(x, y)
                val_loss_sum += loss.item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        best_val_loss = min(best_val_loss, avg_val_loss)
        
        print(
            f"  Epoch {epoch + 1}: train_loss={train_loss_sum / len(train_loader):.4f}, "
            f"val_loss={avg_val_loss:.4f}"
        )
        
        model.train()
    
    return {
        "config_name": config_name,
        "n_embd": config_dict["n_embd"],
        "n_layer": config_dict["n_layer"],
        "n_head": config_dict["n_head"],
        "dropout": config_dict["dropout"],
        "learning_rate": config_dict["learning_rate"],
        "batch_size": config_dict["batch_size"],
        "weight_decay": config_dict["weight_decay"],
        "final_train_loss": final_train_loss,
        "best_val_loss": best_val_loss,
        "val_perplexity": float(2 ** best_val_loss),
        "num_params": model.count_parameters(),
    }


def run_ablation_study(
    processed_dir: str | Path = "data/processed",
    output_file: str | Path = "ablation_results.csv",
) -> None:
    """Run full ablation study with multiple configurations.

    Args:
        processed_dir: Directory containing train.pt and val.pt files.
        output_file: Path to save results CSV.
    """
    processed_dir = Path(processed_dir)
    
    # Load datasets
    print("Loading datasets...")
    train_ids = torch.load(processed_dir / "train.pt")
    val_ids = torch.load(processed_dir / "val.pt")
    print(f"Train size: {len(train_ids):,}, Val size: {len(val_ids):,}")
    
    # Set seed and device
    set_seed(42)
    device = get_device("auto")
    print(f"Using device: {device}")
    
    # Get configurations
    configs = create_ablation_configs()
    print(f"Running {len(configs)} ablation experiments...")
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running experiment: {config['name']}")
        result = run_single_experiment(config.copy(), train_ids, val_ids, device)
        results.append(result)
        print(f"  → Best Val Loss: {result['best_val_loss']:.4f}")
        print(f"  → Val Perplexity: {result['val_perplexity']:.2f}")
    
    # Save results
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    
    sorted_results = sorted(results, key=lambda x: x["best_val_loss"])
    
    print("\nTop 5 configurations by validation loss:")
    for i, result in enumerate(sorted_results[:5], 1):
        print(
            f"{i}. {result['config_name']:20s} | "
            f"Loss: {result['best_val_loss']:.4f} | "
            f"Perplexity: {result['val_perplexity']:.2f} | "
            f"Params: {format_number(result['num_params'])}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter ablation study"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Directory with train.pt and val.pt files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ablation_results.csv",
        help="Output CSV file for results",
    )
    
    args = parser.parse_args()
    
    run_ablation_study(args.processed_dir, args.output)


if __name__ == "__main__":
    main()
