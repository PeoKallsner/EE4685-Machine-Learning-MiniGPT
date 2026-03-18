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

        # TODO: add special tokens (e.g. <PAD>, <UNK>, <BOS>, <EOS>) if needed

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
        # TODO: implement
        raise NotImplementedError

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
        # TODO: implement
        raise NotImplementedError

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
        # TODO: implement
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the vocabulary to a JSON file.

        Args:
            path: Destination file path (e.g. ``data/processed/vocab.json``).

        TODO: implement using ``json.dump``.
        """
        # TODO: implement
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        """Load a previously saved vocabulary from a JSON file.

        Args:
            path: Path to the JSON vocabulary file.

        TODO: implement using ``json.load``.
        """
        # TODO: implement
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
