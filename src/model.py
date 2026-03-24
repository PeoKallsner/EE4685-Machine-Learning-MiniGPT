"""
src/model.py
------------
Full MiniGPT decoder-only transformer language model.

Architecture overview
~~~~~~~~~~~~~~~~~~~~~
::

    Input IDs  (B, T)
        │
    TokenEmbedding + PositionalEmbedding   → (B, T, C)
        │
    Dropout
        │
    ┌──────────────────────────────┐
    │  TransformerBlock × n_layer  │
    │  ┌──────────────────────┐    │
    │  │  LayerNorm           │    │
    │  │  MultiHeadSelfAttn   │    │
    │  │  Residual connection │    │
    │  ├──────────────────────┤    │
    │  │  LayerNorm           │    │
    │  │  FeedForward (MLP)   │    │
    │  │  Residual connection │    │
    │  └──────────────────────┘    │
    └──────────────────────────────┘
        │
    Final LayerNorm
        │
    Linear head  → logits (B, T, vocab_size)

Usage example::

    from src.config import ModelConfig
    config = ModelConfig(vocab_size=65, block_size=256, n_embd=128,
                         n_layer=4, n_head=4)
    model = MiniGPT(config)
    logits, loss = model(input_ids, targets)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention import MultiHeadSelfAttention
from src.config import ModelConfig


# ---------------------------------------------------------------------------
# Feed-forward (MLP) block
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside each transformer block.

    Structure: Linear → GELU → Linear → Dropout

    Args:
        n_embd: Input / output embedding dimension.
        ffn_multiplier: The hidden layer has size ``n_embd * ffn_multiplier``.
        dropout: Dropout probability.
        bias: Whether to add bias to the linear layers.

    TODO:
        - Define the two linear layers and the dropout layer.
        - Implement the forward pass.
    """

    def __init__(
        self,
        n_embd: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # Define the feed-forward network: Linear → GELU → Linear → Dropout
        hidden_dim = n_embd * ffn_multiplier
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Args:
            x: Tensor of shape ``(B, T, C)``.

        Returns:
            Tensor of the same shape ``(B, T, C)``.

        TODO: pass *x* through ``self.net``.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Single transformer block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """A single decoder transformer block.

    Combines multi-head self-attention and a feed-forward network, each
    preceded by layer normalisation and followed by a residual connection
    (Pre-LN style).

    Args:
        config: Model configuration (uses ``n_embd``, ``n_head``,
            ``block_size``, ``dropout``, ``bias``).

    TODO:
        - Define ``self.ln1``, ``self.attn``, ``self.ln2``, ``self.ffn``.
        - Implement the forward pass with residual connections.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # Layer normalization before attention
        self.ln1 = nn.LayerNorm(config.n_embd)
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
            bias=config.bias,
        )
        # Layer normalization before feed-forward
        self.ln2 = nn.LayerNorm(config.n_embd)
        # Position-wise feed-forward network
        self.ffn = FeedForward(
            n_embd=config.n_embd,
            ffn_multiplier=config.ffn_multiplier,
            dropout=config.dropout,
            bias=config.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one transformer block to *x*.

        Args:
            x: Tensor of shape ``(B, T, C)``.

        Returns:
            Tensor of the same shape ``(B, T, C)``.

        TODO:
            - x = x + self.attn(self.ln1(x))   (attention + residual)
            - x = x + self.ffn(self.ln2(x))    (FFN + residual)
        """
        # Attention with residual connection (Pre-LN style)
        x = x + self.attn(self.ln1(x))
        # Feed-forward with residual connection (Pre-LN style)
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full MiniGPT model
# ---------------------------------------------------------------------------


class MiniGPT(nn.Module):
    """Decoder-only transformer language model (GPT-style).

    Args:
        config: A :class:`~src.config.ModelConfig` instance specifying all
            architectural hyperparameters.

    Attributes:
        config: The model configuration.

    TODO:
        - Define token embedding table (``nn.Embedding``).
        - Define positional embedding table (``nn.Embedding``).
        - Define an ``nn.ModuleList`` of :class:`TransformerBlock` layers.
        - Define the final layer norm.
        - Define the language-model head (``nn.Linear``).
        - Implement :meth:`forward`.
        - Implement :meth:`count_parameters`.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        
        # Token embedding table
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Positional embedding table
        self.pos_embed = nn.Embedding(config.block_size, config.n_embd)
        
        # Dropout after embeddings
        self.dropout = nn.Dropout(config.dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        
        # Final layer normalization
        self.ln_final = nn.LayerNorm(config.n_embd)
        
        # Language model head (project to vocabulary)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run a forward pass through MiniGPT.

        Args:
            input_ids: Integer tensor of shape ``(B, T)`` containing token
                IDs.  ``T`` must be ≤ ``config.block_size``.
            targets: Optional integer tensor of shape ``(B, T)`` containing
                the ground-truth next tokens.  When provided, the
                cross-entropy loss is computed and returned.

        Returns:
            A tuple ``(logits, loss)`` where:

            - ``logits`` has shape ``(B, T, vocab_size)``
            - ``loss`` is the scalar cross-entropy loss, or ``None`` if
              *targets* was not provided.

        TODO:
            1. Compute token embeddings + positional embeddings.
            2. Pass through all transformer blocks.
            3. Apply the final layer norm.
            4. Project to logit space via the LM head.
            5. If *targets* is given, compute cross-entropy loss.
        """
        B, T = input_ids.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )
        
        # Get token embeddings
        tok_emb = self.token_embed(input_ids)  # (B, T, C)
        
        # Get positional embeddings
        pos = torch.arange(T, dtype=torch.long, device=input_ids.device)  # (T,)
        pos_emb = self.pos_embed(pos)  # (T, C)
        
        # Combine token and positional embeddings
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)  # (B, T, C)
        
        # Apply final layer norm
        x = self.ln_final(x)  # (B, T, C)
        
        # Project to vocabulary space
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy loss (only care about next-token prediction)
            logits_flat = logits.view(-1, self.config.vocab_size)  # (B*T, vocab_size)
            targets_flat = targets.view(-1)  # (B*T,)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters.

        TODO: sum ``p.numel()`` for all parameters where ``p.requires_grad``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
