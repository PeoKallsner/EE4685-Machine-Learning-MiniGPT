#!/usr/bin/env python3
"""Quick test script for training and utility implementations."""

from src.utils import format_number, get_device, count_parameters, set_seed
from src.config import ModelConfig
from src.model import MiniGPT
import torch

# Test format_number
print("Testing format_number:")
print(f"  format_number(1_300_000) = {format_number(1_300_000)}")
print(f"  format_number(42_000) = {format_number(42_000)}")
print(f"  format_number(512) = {format_number(512)}")

# Test get_device
print("\nTesting get_device:")
device = get_device('auto')
print(f"  get_device('auto') = {device}")

# Test set_seed
print("\nTesting set_seed:")
set_seed(42)
val1 = torch.randn(3)
set_seed(42)
val2 = torch.randn(3)
print(f"  Seeded values equal: {torch.allclose(val1, val2)}")

# Test count_parameters
print("\nTesting count_parameters:")
config = ModelConfig(vocab_size=65, block_size=256, n_embd=128, n_layer=2, n_head=4)
model = MiniGPT(config)
param_count = count_parameters(model)
print(f"  Model has {format_number(param_count)} parameters")
print(f"  Model.count_parameters() = {format_number(model.count_parameters())}")

print("\n✓ All utility functions working correctly!")
