# Architecture Notes — MiniGPT

This document describes the design decisions behind the MiniGPT
architecture and provides a brief explanation of each component.

---

## Overview

MiniGPT is a **decoder-only transformer** — the same high-level architecture
used by GPT-2 and GPT-3.  It is trained with a **language modelling
objective**: given a sequence of tokens, predict the next token at every
position.

---

## Architecture Diagram

```
Input token IDs  (B, T)
       │
   TokenEmbedding  (vocab_size → n_embd)
       +
   PositionalEmbedding  (block_size → n_embd)
       │
   Dropout
       │
   ╔══════════════════════════════╗
   ║  TransformerBlock × n_layer  ║
   ║  ┌──────────────────────┐   ║
   ║  │ LayerNorm (Pre-LN)   │   ║
   ║  │ MultiHeadSelfAttention│  ║
   ║  │ Residual connection  │   ║
   ║  ├──────────────────────┤   ║
   ║  │ LayerNorm (Pre-LN)   │   ║
   ║  │ FeedForward (MLP)    │   ║
   ║  │ Residual connection  │   ║
   ║  └──────────────────────┘   ║
   ╚══════════════════════════════╝
       │
   Final LayerNorm
       │
   Linear LM head  (n_embd → vocab_size)
       │
   Logits  (B, T, vocab_size)
```

---

## Component Details

### Token Embedding
- Learnable lookup table: `nn.Embedding(vocab_size, n_embd)`
- Maps each integer token ID to a dense vector of dimension `n_embd`.

### Positional Embedding
- Learnable absolute positional embeddings: `nn.Embedding(block_size, n_embd)`
- Added (not concatenated) to the token embeddings.
- Alternative: sinusoidal or rotary position embeddings (RoPE).
  - TODO: consider implementing RoPE for better length generalisation.

### Multi-Head Self-Attention (Causal)
- Linear projections for Q, K, V: three `nn.Linear(n_embd, n_embd)` layers.
- Attention score: `A = softmax( Q K^T / sqrt(head_dim) ) * causal_mask`
- The causal mask is a lower-triangular matrix of ones; future positions
  receive `-inf` before the softmax, giving attention weight ≈ 0.
- Output projection: `nn.Linear(n_embd, n_embd)`.
- Dropout on attention weights and residual output.

### Feed-Forward Network (MLP)
- Two linear layers with a GELU activation in between.
- Hidden dimension = `n_embd * ffn_multiplier` (default ×4).
- Matches the original transformer and GPT designs.

### Pre-LN (Pre-Layer Normalisation)
- Layer norm is applied **before** the attention and FFN sub-layers
  (not after, as in the original Vaswani et al. paper).
- Pre-LN training is more stable at larger scales.
  Reference: Xiong et al. (2020), "On Layer Normalization in the Transformer
  Architecture". https://arxiv.org/abs/2002.04745

### Residual Connections
- Each sub-layer (attention, FFN) is wrapped with a residual connection:
  `x = x + sublayer(LayerNorm(x))`
- Allows gradients to flow directly through the network depth.

### Language Model Head
- A single `nn.Linear(n_embd, vocab_size, bias=False)` layer.
- Produces un-normalised log-probabilities (logits) for each token.
- **Weight tying** (optional): sharing weights between the token embedding
  matrix and the LM head can reduce parameters and improve performance.
  - TODO: investigate weight tying.

---

## Design Choices and Trade-offs

| Choice | Rationale |
|--------|-----------|
| Character-level tokeniser | Simple, no external dependencies; vocabulary ≤ 100 |
| Learnable positional embeddings | Easy to implement; sufficient for short contexts |
| Pre-LN | More stable training vs. Post-LN |
| No bias in linear layers | Slightly fewer params; GPT-2 / NanoGPT convention |
| AdamW with weight decay | Standard for transformer training |

---

## Hyperparameter Sensitivity

<!-- TODO: fill in after experiments -->

| Hyperparameter | Tested values | Best value | Notes |
|---------------|--------------|-----------|-------|
| n_layer | — | — | More layers → lower perplexity (diminishing returns) |
| n_head | — | — | Must divide n_embd |
| n_embd | — | — | Larger = more capacity, slower training |
| block_size | — | — | Longer context = more memory |
| learning_rate | — | — | 3e-4 is a good default |
| dropout | — | — | 0.0 for small data, 0.1+ for larger |

---

## References

- Vaswani et al. (2017), "Attention Is All You Need"
  https://arxiv.org/abs/1706.03762
- Radford et al. (2019), "Language Models are Unsupervised Multitask Learners" (GPT-2)
  https://openai.com/research/better-language-models
- Karpathy, A. (2022), "nanoGPT"
  https://github.com/karpathy/nanoGPT
