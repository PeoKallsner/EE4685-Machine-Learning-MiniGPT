"""
tests/test_attention.py
-----------------------
Unit tests for :class:`src.attention.MultiHeadSelfAttention`.

These tests verify that:
- The output tensor has the correct shape.
- The module cannot attend to future tokens (causal masking).
- The module handles batched input correctly.

All tests are stubs until ``src/attention.py`` is implemented.
"""

import pytest

torch = pytest.importorskip("torch")

from src.attention import MultiHeadSelfAttention


@pytest.fixture
def attn() -> MultiHeadSelfAttention:
    """Return a small MultiHeadSelfAttention module for testing."""
    return MultiHeadSelfAttention(
        n_embd=64,
        n_head=4,
        block_size=32,
        dropout=0.0,  # deterministic for testing
        bias=False,
    )


class TestOutputShape:
    """Tests that the attention output has the expected shape."""

    def test_output_shape_matches_input(self, attn: MultiHeadSelfAttention) -> None:
        """Output shape should equal input shape (B, T, C).

        TODO: remove ``pytest.skip`` once MultiHeadSelfAttention.forward is
        implemented.
        """
        B, T, C = 2, 16, 64
        x = torch.randn(B, T, C)
        out = attn(x)
        assert out.shape == (B, T, C)


class TestCausalMask:
    """Tests that the causal mask prevents attending to future tokens."""

    def test_causal_property(self, attn: MultiHeadSelfAttention) -> None:
        """Changing future tokens should not affect earlier output positions.

        TODO: remove ``pytest.skip`` once the causal mask is implemented.
        """
        attn.eval()
        B, T, C = 1, 8, 64
        x = torch.randn(B, T, C)
        x_modified = x.clone()
        # Modify the last token — output at earlier positions should not change
        x_modified[0, -1, :] = torch.randn(C)
        with torch.no_grad():
            out1 = attn(x)
            out2 = attn(x_modified)
        assert torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5)
