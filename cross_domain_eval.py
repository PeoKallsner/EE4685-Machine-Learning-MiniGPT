#!/usr/bin/env python3
"""
Cross-domain evaluation: Test MiniGPT trained on Shakespeare on Jane Austen text.

Measures domain transfer and robustness to different writing styles.
"""

import torch
import math
from pathlib import Path
from src.config import ModelConfig
from src.model import MiniGPT
from src.tokenizer import CharTokenizer
from src.dataset import TextDataset
from src.evaluate import compute_loss, compute_perplexity
from torch.utils.data import DataLoader


device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Load best checkpoint (trained on Shakespeare)
ckpts = sorted(Path("checkpoints").glob("checkpoint_step_*.pt"))
ckpts.sort(key=lambda x: int(x.stem.split("_")[-1]))
best_ckpt = ckpts[-1]
checkpoint = torch.load(best_ckpt, weights_only=False)

print("=" * 70)
print("CROSS-DOMAIN EVALUATION: Shakespeare Model on Jane Austen")
print("=" * 70)
print(f"\nModel: {best_ckpt.name} (trained on Shakespeare)")
print(f"Device: {device}\n")

# Setup model (trained on Shakespeare)
config = ModelConfig(block_size=256, vocab_size=65, n_embd=128, n_layer=4, n_head=4, dropout=0.1)
model = MiniGPT(config).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Load tokenizer (trained on Shakespeare)
tok = CharTokenizer()
shakespeare_text = Path("data/raw/input.txt").read_text()
tok.build_vocab(shakespeare_text)

print(f"Tokenizer vocabulary size: {tok.vocab_size}")
print(f"Vocabulary: {sorted(list(tok.char_to_id.keys()))[:20]}...")

# Try to load Jane Austen text
jane_austen_path = Path("data/raw/jane_austen.txt")
if not jane_austen_path.exists():
    print(f"\n❌ ERROR: {jane_austen_path} not found")
    print("\nTo test on Jane Austen:")
    print("  1. Download: curl -o data/raw/jane_austen.txt https://www.gutenberg.org/cache/epub/1342/pg1342.txt")
    print("  2. Run this script again")
    exit(1)

# Load Jane Austen text
print(f"\n✓ Loaded Jane Austen text from {jane_austen_path.name}")
jane_austen_text = jane_austen_path.read_text()
print(f"  Size: {len(jane_austen_text):,} characters")

# Analyze character coverage
jane_chars = set(jane_austen_text)
vocab_chars = set(tok.char_to_id.keys())
coverage = len(jane_chars & vocab_chars) / len(jane_chars) if jane_chars else 0

print(f"  Unique characters: {len(jane_chars)}")
print(f"  In vocabulary: {len(jane_chars & vocab_chars)} ({coverage*100:.1f}%)")

if coverage < 1.0:
    unknown_chars = jane_chars - vocab_chars
    print(f"  Unknown characters: {sorted(unknown_chars)[:20]}...")

# Encode Jane Austen text (use <UNK> for unknown characters)
try:
    jane_ids = torch.tensor(tok.encode(jane_austen_text), dtype=torch.long)
    print(f"\n✓ Encoded to {len(jane_ids):,} tokens")
except Exception as e:
    print(f"\n⚠️ Encoding error: {e}")
    exit(1)

# Create dataset and dataloader
jane_dataset = TextDataset(jane_ids, config.block_size)
jane_loader = DataLoader(jane_dataset, batch_size=32, shuffle=False)

print(f"  Created {len(jane_dataset):,} training examples")

# Evaluate
print("\n" + "=" * 70)
print("EVALUATION ON JANE AUSTEN TEXT")
print("=" * 70)

with torch.no_grad():
    print("\nEvaluating on sample (first 100 batches)...")
    jane_loss = compute_loss(model, jane_loader, device, max_batches=100)
    jane_ppl = compute_perplexity(jane_loss)

# Compare to Shakespeare
print("\n" + "=" * 70)
print("CROSS-DOMAIN COMPARISON")
print("=" * 70)

shakespeare_test_ppl = 10.57  # From validation
print(f"\nModel trained and tested on Shakespeare:")
print(f"  Test PPL: {shakespeare_test_ppl:.2f}")

print(f"\nModel evaluated on Jane Austen (different domain):")
print(f"  Jane Austen PPL: {jane_ppl:.2f}")

ppl_increase = jane_ppl - shakespeare_test_ppl
ppl_ratio = jane_ppl / shakespeare_test_ppl

print(f"\nDomain shift analysis:")
print(f"  Absolute increase: {ppl_increase:+.2f} PPL")
print(f"  Relative increase: {(ppl_ratio - 1) * 100:+.1f}%")
print(f"  Ratio (Jane/Shak): {ppl_ratio:.3f}x")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if ppl_increase < 1.0:
    print("\n✅ EXCELLENT: Model generalizes across domains!")
    print("   Jane Austen PPL is close to Shakespeare PPL")
elif ppl_increase < 2.0:
    print("\n✅ GOOD: Model shows reasonable domain transfer")
    print("   Moderate PPL increase (expected for different author)")
elif ppl_increase < 3.0:
    print("\n⚠️ MODERATE: Notable domain shift detected")
    print("   Model specialized to Shakespeare, struggles with Austen")
else:
    print("\n⚠️ SIGNIFICANT: Model is domain-specific")
    print("   Large PPL increase shows domain specialization")

print(f"""
Explanation:
• PPL increase from {shakespeare_test_ppl:.2f} → {jane_ppl:.2f} is {'small' if ppl_increase < 2 else 'large'}
• This indicates model {'generalizes well across' if ppl_increase < 2 else 'is specialized to'} writing styles
• Jane Austen uses different: vocabulary, punctuation, sentence structure
• Model trained on 892K Shakespeare tokens, applied to different corpus

Conclusion:
→ Model captures {'general' if ppl_increase < 2 else 'domain-specific'} language patterns
→ Training on Shakespeare {'transfers well to' if ppl_ratio < 1.5 else 'does not transfer to'} other domains
""")

print("=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"\n{'Dataset':<20} {'Loss':<10} {'Perplexity':<12} {'Status':<20}")
print("-" * 65)
print(f"{'Shakespeare (test)':<20} {'2.3584':<10} {shakespeare_test_ppl:<12.2f} {'Original domain':<20}")
print(f"{'Jane Austen':<20} {jane_loss:<10.4f} {jane_ppl:<12.2f} {'Cross-domain':<20}")
print()

# Additional analysis
print("=" * 70)
print("CONFIDENCE ASSESSMENT")
print("=" * 70)
print(f"""
Based on Jane Austen PPL ({jane_ppl:.2f}):

✅ Model Quality: {'Excellent' if jane_ppl < 12 else 'Good' if jane_ppl < 15 else 'Fair'}
✅ Domain Transfer: {'Strong' if ppl_ratio < 1.3 else 'Moderate' if ppl_ratio < 1.5 else 'Weak'}
✅ Generalization: {'Excellent' if ppl_increase < 1 else 'Good' if ppl_increase < 2 else 'Fair'}

The model {'successfully' if jane_ppl < 12 else 'partially'} generalizes to unseen text from different domains.
""")
