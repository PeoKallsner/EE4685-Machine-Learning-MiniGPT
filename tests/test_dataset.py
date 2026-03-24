"""
tests/test_dataset.py
---------------------
Unit tests for :class:`src.dataset.TextDataset`.

These tests verify that the dataset:
- Returns the correct number of items.
- Returns tensors of the expected shape.
- Produces targets shifted by one position relative to inputs.

All tests are stubs until ``src/dataset.py`` is implemented.
"""

import pytest

torch = pytest.importorskip("torch")

from src.dataset import TextDataset


@pytest.fixture
def token_ids() -> list:
    """A simple sequence of token IDs for testing."""
    return list(range(100))  # IDs 0..99


class TestTextDatasetLen:
    """Tests for :meth:`TextDataset.__len__`."""

    def test_len_correct(self, token_ids: list) -> None:
        """Dataset length should equal len(token_ids) - block_size.

        TODO: remove ``pytest.skip`` once TextDataset.__init__ is implemented.
        """
        block_size = 10
        ds = TextDataset(token_ids, block_size)
        assert len(ds) == len(token_ids) - block_size


class TestTextDatasetGetItem:
    """Tests for :meth:`TextDataset.__getitem__`."""

    def test_shapes(self, token_ids: list) -> None:
        """x and y should both have shape (block_size,).

        TODO: remove ``pytest.skip`` once TextDataset.__getitem__ is implemented.
        """
        block_size = 10
        ds = TextDataset(token_ids, block_size)
        x, y = ds[0]
        assert x.shape == (block_size,)
        assert y.shape == (block_size,)

    def test_target_is_shifted_input(self, token_ids: list) -> None:
        """y[i] should equal x[i + 1] for all positions.

        TODO: remove ``pytest.skip`` once TextDataset.__getitem__ is implemented.
        """
        block_size = 10
        ds = TextDataset(token_ids, block_size)
        x, y = ds[0]
        # The first (block_size - 1) elements of y should match x[1:]
        assert torch.equal(x[1:], y[:-1])
