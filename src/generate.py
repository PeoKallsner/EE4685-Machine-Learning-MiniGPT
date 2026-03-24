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
    # 1. Encode prompt to token IDs
    prompt_ids = tokenizer.encode(prompt)
    
    # 2. Move IDs to device and reshape to (1, T)
    if device is None:
        device = next(model.parameters()).device
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T)
    
    # 3. Set model to eval mode and wrap in torch.no_grad()
    model.eval()
    with torch.no_grad():
        # 4. Loop max_new_tokens times
        for _ in range(max_new_tokens):
            # 4a. Truncate context to model.config.block_size if needed
            context_truncated = context[:, -model.config.block_size:]
            
            # 4b. Forward pass → logits of shape (1, T, vocab_size)
            logits, _ = model(context_truncated)  # (1, T, vocab_size), loss=None
            
            # 4c. Take the last time step: logits (1, vocab_size)
            logits_next = logits[0, -1, :]  # (vocab_size,)
            
            # 4d. Apply temperature scaling
            if temperature != 0.0:
                logits_next = logits_next / temperature
            
            # 4e. Apply top-k filtering if top_k is given
            if top_k is not None:
                # Get top-k values and indices
                top_k_logits, top_k_indices = torch.topk(logits_next, top_k)
                # Create a tensor of -inf for all positions
                logits_filtered = torch.full_like(logits_next, float('-inf'))
                # Fill in the top-k positions
                logits_filtered[top_k_indices] = top_k_logits
                logits_next = logits_filtered
            
            # 4f. Softmax → probabilities
            probs = F.softmax(logits_next, dim=-1)
            
            # 4g. Sample with torch.multinomial (or argmax if temperature=0)
            if temperature == 0.0:
                # Greedy: pick the most likely token
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                # Multinomial sampling
                next_token = torch.multinomial(probs, num_samples=1)  # (1,)
            
            # 4h. Append sampled token ID to the context
            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)  # (1, T+1)
    
    # 5. Decode the full sequence (including prompt) with tokenizer.decode
    generated_ids = context[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    # 6. Return the decoded string
    return generated_text
