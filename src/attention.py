"""
src/attention.py
----------------
Scaled dot-product multi-head self-attention with a causal mask.

This is the most important building block of the transformer.  The causal
mask ensures that position *i* can only attend to positions ≤ *i*, which is
required for an autoregressive (decoder-only) language model.

Mathematical overview
~~~~~~~~~~~~~~~~~~~~~
Given an input tensor X of shape ``(B, T, C)`` (batch, time, channels):

1. Project X into queries Q, keys K, and values V using learned linear layers.
2. Compute attention scores: ``A = softmax( Q @ K.T / sqrt(d_k) ) * mask``
3. Compute the attended output: ``out = A @ V``
4. Concatenate all heads and apply a final output projection.

Reference:
    Vaswani et al. (2017), "Attention Is All You Need"
    https://arxiv.org/abs/1706.03762

Usage example::

    attn = MultiHeadSelfAttention(n_embd=128, n_head=4, block_size=256,
                                  dropout=0.1)
    x = torch.randn(2, 64, 128)   # batch=2, seq_len=64, embd=128
    out = attn(x)                  # shape: (2, 64, 128)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention module.

    Args:
        n_embd: Embedding (model) dimension.  Must be divisible by *n_head*.
        n_head: Number of attention heads.
        block_size: Maximum sequence length; used to build the causal mask.
        dropout: Dropout probability applied to the attention weights.
        bias: Whether to use bias in the linear projection layers.

    Attributes:
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        head_dim: Dimension per head (``n_embd // n_head``).

    TODO:
        - Add the Q, K, V projection linear layers.
        - Add the output projection linear layer.
        - Register the causal mask as a buffer (``register_buffer``).
        - Add a dropout layer.
        - Implement the forward pass.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        assert n_embd % n_head == 0, (
            f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        )

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Q, K, V projection layers
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        # Output projection layer
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Register causal mask buffer (lower triangular matrix)
        # Positions can attend to themselves and past positions
        causal_mask = torch.tril(torch.ones(1, 1, block_size, block_size))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal multi-head self-attention.

        Args:
            x: Input tensor of shape ``(B, T, C)`` where B is batch size,
               T is sequence length, and C is the embedding dimension.

        Returns:
            Output tensor of the same shape ``(B, T, C)``.

        TODO:
            1. Compute Q, K, V by projecting *x* with the learned layers.
            2. Reshape Q, K, V to separate the heads:
               ``(B, n_head, T, head_dim)``.
            3. Compute scaled dot-product attention scores.
            4. Apply the causal mask (set future positions to -inf).
            5. Softmax + dropout on the attention weights.
            6. Weighted sum of values.
            7. Reshape and apply the output projection.
        """
        B, T, C = x.shape
        
        # Project input to Q, K, V
        Q = self.q_proj(x)  # (B, T, C)
        K = self.k_proj(x)  # (B, T, C)
        V = self.v_proj(x)  # (B, T, C)
        
        # Reshape to separate heads: (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention scores
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply causal mask: prevent attending to future positions
        # Only use the part of the mask that corresponds to the current sequence length
        # causal_mask has 1s where attention is allowed, 0s where it's not
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)
        
        # Weighted sum of values
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        out = attn_weights @ V
        
        # Reshape back to (B, T, C)
        out = out.transpose(1, 2).contiguous()  # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim)
        out = out.view(B, T, C)  # (B, T, n_head, head_dim) -> (B, T, C)
        
        # Apply output projection
        out = self.out_proj(out)
        
        # Apply residual dropout
        out = self.resid_dropout(out)
        
        return out
