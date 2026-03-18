"""
src/dataset.py
--------------
PyTorch :class:`~torch.utils.data.Dataset` for character-level language
modelling.

The dataset takes a sequence of token IDs (produced by the tokeniser) and
returns overlapping ``(input, target)`` pairs of length ``block_size``.

For language modelling the target at position *i* is the token at position
*i + 1* — i.e., the model always tries to predict the **next** token.

Example::

    token_ids = tokenizer.encode(raw_text)   # long list of integers
    dataset = TextDataset(token_ids, block_size=256)
    x, y = dataset[0]  # x.shape == y.shape == (256,)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """A sliding-window dataset for next-token prediction.

    Each item returned is a pair ``(x, y)`` of integer tensors with shape
    ``(block_size,)``.  The target ``y`` is ``x`` shifted one position to
    the right.

    Args:
        token_ids: A flat list (or 1-D tensor) of integer token IDs
            representing the entire corpus.
        block_size: Context window length.

    Attributes:
        data: 1-D :class:`torch.Tensor` of all token IDs.
        block_size: Context window length.

    TODO:
        - Implement :meth:`__len__` to return the number of possible windows.
        - Implement :meth:`__getitem__` to slice out ``(x, y)`` pairs.
    """

    def __init__(self, token_ids: List[int] | torch.Tensor, block_size: int) -> None:
        # TODO: store token_ids as a torch.LongTensor and save block_size
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of non-overlapping windows in the dataset.

        TODO:
            - Return ``len(self.data) - self.block_size``.
        """
        # TODO: implement
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the (input, target) pair at index *idx*.

        Args:
            idx: Index of the starting position in the token sequence.

        Returns:
            A tuple ``(x, y)`` where both tensors have shape
            ``(block_size,)`` and ``y[i] == x[i + 1]`` for all valid *i*.

        TODO:
            - Slice ``self.data[idx : idx + block_size]`` as *x*.
            - Slice ``self.data[idx + 1 : idx + block_size + 1]`` as *y*.
        """
        # TODO: implement
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helper: build train/val/test splits from a raw text file
# ---------------------------------------------------------------------------


def prepare_splits(
    raw_text_path: str | Path,
    tokenizer,  # CharTokenizer instance
    processed_dir: str | Path,
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> None:
    """Tokenise a raw text file and save train/val/test splits to disk.

    Args:
        raw_text_path: Path to the raw ``.txt`` file.
        tokenizer: A fitted :class:`~src.tokenizer.CharTokenizer` instance.
        processed_dir: Directory where the split files will be saved.
        val_split: Fraction of tokens reserved for validation.
        test_split: Fraction of tokens reserved for testing.

    TODO:
        - Read the raw text file.
        - Build the tokeniser vocabulary from the full text.
        - Encode the full text into token IDs.
        - Split into train / val / test according to the given fractions.
        - Save each split as a ``.pt`` file using ``torch.save``.
    """
    # TODO: implement
    raise NotImplementedError
