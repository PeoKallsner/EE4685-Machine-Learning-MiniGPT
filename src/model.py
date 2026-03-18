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
        # TODO: define self.net as an nn.Sequential with the layers above

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Args:
            x: Tensor of shape ``(B, T, C)``.

        Returns:
            Tensor of the same shape ``(B, T, C)``.

        TODO: pass *x* through ``self.net``.
        """
        # TODO: implement
        raise NotImplementedError


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
        # TODO: define layer norm, attention, and feed-forward sub-modules

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
        # TODO: implement
        raise NotImplementedError


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
        # TODO: define all sub-modules

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
        # TODO: implement
        raise NotImplementedError

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters.

        TODO: sum ``p.numel()`` for all parameters where ``p.requires_grad``.
        """
        # TODO: implement
        raise NotImplementedError
