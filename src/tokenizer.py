"""
src/tokenizer.py
----------------
Character-level tokeniser for MiniGPT.

A tokeniser converts raw text into sequences of integer token IDs that the
model can process, and converts sequences of IDs back into human-readable
text.

For simplicity we start with a **character-level** tokeniser: every unique
character in the training corpus becomes a token.  This gives a small
vocabulary (typically < 100 tokens for English text) and requires no
external libraries.

Possible extension: replace with a BPE (Byte-Pair Encoding) tokeniser
such as ``tiktoken`` or ``sentencepiece`` for better compression.

Usage example::

    tokenizer = CharTokenizer()
    tokenizer.build_vocab("Hello, world!")
    ids = tokenizer.encode("Hello")     # [7, 3, 11, 11, 14]  (example)
    text = tokenizer.decode(ids)        # "Hello"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


class CharTokenizer:
    """A simple character-level tokeniser.

    Attributes:
        char_to_id: Mapping from character to integer token ID.
        id_to_char: Reverse mapping from integer token ID to character.
        vocab_size: Number of unique tokens in the vocabulary.

    TODO:
        - Implement :meth:`build_vocab` to scan a text and create mappings.
        - Implement :meth:`encode` to convert a string to a list of IDs.
        - Implement :meth:`decode` to convert a list of IDs back to a string.
        - Implement :meth:`save` and :meth:`load` for vocabulary persistence.
    """

    def __init__(self) -> None:
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self.vocab_size: int = 0
        self.unk_token: str = "<UNK>"
        self.unk_id: int = -1  # Will be set during build_vocab

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def build_vocab(self, text: str) -> None:
        """Scan *text* and build the character vocabulary.

        After calling this method, :attr:`char_to_id`, :attr:`id_to_char`,
        and :attr:`vocab_size` will be populated.

        Args:
            text: The full training corpus as a single string.

        TODO:
            - Extract the sorted set of unique characters from *text*.
            - Assign an integer ID to each character.
            - Populate ``self.char_to_id`` and ``self.id_to_char``.
            - Set ``self.vocab_size``.
        """
        # Extract unique characters and sort them for consistency
        unique_chars = sorted(set(text))
        
        # Create mappings from character to ID and vice versa
        self.char_to_id = {char: idx for idx, char in enumerate(unique_chars)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        
        # Set the vocabulary size (not including <UNK>)
        self.vocab_size = len(unique_chars)
        
        # Set the <UNK> token ID as the next available ID
        self.unk_id = self.vocab_size

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Convert a string into a list of integer token IDs.

        Args:
            text: Input string to encode.

        Returns:
            A list of integer token IDs, one per character.

        Raises:
            ValueError: If the vocabulary has not been built yet.

        TODO:
            - Look up each character in ``self.char_to_id``.
            - Handle unknown characters gracefully (skip or map to <UNK>).
        """
        if not self.char_to_id:
            raise ValueError("Vocabulary has not been built. Call build_vocab() first.")
        
        # Convert each character to its token ID, map unknown characters to <UNK>
        ids = []
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                ids.append(self.unk_id)
        
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert a list of integer token IDs back into a string.

        Args:
            ids: Sequence of integer token IDs to decode.

        Returns:
            The reconstructed string.

        Raises:
            ValueError: If the vocabulary has not been built yet.

        TODO:
            - Look up each ID in ``self.id_to_char``.
            - Join the characters and return the result.
        """
        if not self.id_to_char:
            raise ValueError("Vocabulary has not been built. Call build_vocab() first.")
        
        # Convert each token ID back to its character
        # Skip <UNK> tokens as they represent unknown characters
        chars = []
        for token_id in ids:
            if token_id in self.id_to_char:
                chars.append(self.id_to_char[token_id])
            elif token_id != self.unk_id:
                # If it's not in the mapping and not <UNK>, skip it
                pass
        
        return "".join(chars)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the vocabulary to a JSON file.

        Args:
            path: Destination file path (e.g. ``data/processed/vocab.json``).

        TODO: implement using ``json.dump``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        vocab_data = {
            "char_to_id": self.char_to_id,
            "id_to_char": self.id_to_char,
            "vocab_size": self.vocab_size,
        }
        
        with open(path, "w") as f:
            json.dump(vocab_data, f, indent=2)

    def load(self, path: str | Path) -> None:
        """Load a previously saved vocabulary from a JSON file.

        Args:
            path: Path to the JSON vocabulary file.

        TODO: implement using ``json.load``.
        """
        path = Path(path)
        
        with open(path, "r") as f:
            vocab_data = json.load(f)
        
        # Restore the vocabulary mappings
        # Note: JSON keys are strings, so we need to convert id_to_char keys back to integers
        self.char_to_id = vocab_data["char_to_id"]
        self.id_to_char = {int(k): v for k, v in vocab_data["id_to_char"].items()}
        self.vocab_size = vocab_data["vocab_size"]
        
        # Restore the <UNK> token ID based on the vocab size
        self.unk_id = self.vocab_size

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
