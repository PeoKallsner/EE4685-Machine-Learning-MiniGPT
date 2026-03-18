"""
src/generate.py
---------------
Autoregressive text generation with MiniGPT.

Once the model is trained, :func:`generate` produces new text by
repeatedly:

1. Passing the current context (token IDs) through the model.
2. Obtaining a probability distribution over the vocabulary from the
   **last** token's logits.
3. Sampling the next token (with temperature scaling and optional top-k
   filtering).
4. Appending the sampled token to the context.
5. Repeating until the desired number of new tokens has been generated.

Sampling strategies
~~~~~~~~~~~~~~~~~~~
- **Greedy** (``temperature=0``): always pick the most likely next token.
- **Temperature sampling**: divide logits by ``temperature`` before softmax.
  Higher temperature → more diverse / random output.
- **Top-k sampling**: restrict sampling to the *k* most probable tokens.

Usage example::

    generated_ids = generate(
        model, tokenizer, prompt="Once upon a time",
        max_new_tokens=200, temperature=0.8, top_k=40,
    )
    print(tokenizer.decode(generated_ids))
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from src.model import MiniGPT
from src.tokenizer import CharTokenizer


def generate(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> str:
    """Generate text from a *prompt* using an autoregressive sampling loop.

    Args:
        model: A trained MiniGPT instance.
        tokenizer: The tokeniser used to encode *prompt* and decode output.
        prompt: The initial text that seeds the generation.
        max_new_tokens: How many tokens to generate beyond the prompt.
        temperature: Softmax temperature.  ``1.0`` = normal distribution;
            values < 1 make the distribution sharper (more deterministic);
            values > 1 make it flatter (more random).
        top_k: If given, only sample from the *k* highest-probability tokens.
        device: The device to run generation on.  Defaults to the device of
            the model parameters.

    Returns:
        The generated text (prompt + newly generated tokens) as a string.

    TODO:
        1. Encode *prompt* to token IDs with ``tokenizer.encode``.
        2. Move IDs to *device* and reshape to ``(1, T)``.
        3. Set model to eval mode and wrap in ``torch.no_grad()``.
        4. Loop ``max_new_tokens`` times:
           a. Truncate context to ``model.config.block_size`` if needed.
           b. Forward pass → logits of shape ``(1, T, vocab_size)``.
           c. Take the last time step: logits ``(1, vocab_size)``.
           d. Apply temperature scaling.
           e. Apply top-k filtering if ``top_k`` is given.
           f. Softmax → probabilities.
           g. Sample with ``torch.multinomial`` (or argmax if greedy).
           h. Append sampled token ID to the context.
        5. Decode the full sequence (including prompt) with ``tokenizer.decode``.
        6. Return the decoded string.
    """
    # TODO: implement
    raise NotImplementedError
