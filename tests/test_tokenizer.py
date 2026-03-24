"""
tests/test_tokenizer.py
-----------------------
Unit tests for :class:`src.tokenizer.CharTokenizer`.

These tests verify that the tokeniser correctly:
- Builds a vocabulary from a string.
- Encodes text to integer IDs.
- Decodes IDs back to the original text (round-trip).
- Reports the correct vocabulary size.
- Saves and loads the vocabulary without loss.

All tests are currently *stubs* — they will raise ``NotImplementedError``
until the tokeniser is implemented.  Fill in the ``# TODO`` sections as you
implement ``src/tokenizer.py``.
"""

import pytest

from src.tokenizer import CharTokenizer


@pytest.fixture
def tokenizer() -> CharTokenizer:
    """Return a :class:`CharTokenizer` with a small vocabulary.

    NOTE: This fixture raises NotImplementedError until build_vocab is
    implemented.  Tests that depend on it are marked with
    ``@pytest.mark.skip`` so they are skipped before the fixture runs.
    """
    t = CharTokenizer()
    t.build_vocab("Hello, World!")
    return t


class TestBuildVocab:
    """Tests for :meth:`CharTokenizer.build_vocab`."""

    def test_vocab_is_non_empty(self, tokenizer: CharTokenizer) -> None:
        """Vocabulary size should be > 0 after building from non-empty text."""
        assert tokenizer.vocab_size > 0

    def test_all_chars_in_vocab(self, tokenizer: CharTokenizer) -> None:
        """Every character in the training text must be in char_to_id."""
        for ch in "Hello, World!":
            assert ch in tokenizer.char_to_id

    def test_vocab_size_matches_unique_chars(self) -> None:
        """vocab_size must equal the number of unique characters in the text.

        TODO: remove ``pytest.skip`` once build_vocab is implemented.
        """
        text = "abcabc"
        t = CharTokenizer()
        t.build_vocab(text)
        assert t.vocab_size == len(set(text))


class TestEncode:
    """Tests for :meth:`CharTokenizer.encode`."""

    def test_encode_returns_list_of_ints(self, tokenizer: CharTokenizer) -> None:
        """encode should return a list of integers."""
        ids = tokenizer.encode("Hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_length_matches_input(self, tokenizer: CharTokenizer) -> None:
        """Encoded length should equal the number of characters in the input."""
        text = "Hi!"
        assert len(tokenizer.encode(text)) == len(text)


class TestDecode:
    """Tests for :meth:`CharTokenizer.decode`."""

    def test_roundtrip(self, tokenizer: CharTokenizer) -> None:
        """decode(encode(text)) should return the original text."""
        text = "Hello"
        assert tokenizer.decode(tokenizer.encode(text)) == text


class TestPersistence:
    """Tests for :meth:`CharTokenizer.save` and :meth:`CharTokenizer.load`."""

    def test_save_and_load_roundtrip(
        self, tokenizer: CharTokenizer, tmp_path
    ) -> None:
        """A tokeniser saved and reloaded should encode text identically."""
        vocab_path = tmp_path / "vocab.json"
        tokenizer.save(vocab_path)

        t2 = CharTokenizer()
        t2.load(vocab_path)
        assert t2.encode("Hello") == tokenizer.encode("Hello")
