"""
tests/test_model.py
-------------------
Unit tests for :class:`src.model.MiniGPT`.

These tests verify that:
- The model produces logits of the correct shape.
- The model computes a scalar loss when targets are provided.
- The parameter count is positive.
- The model can be moved to a device without errors.

All tests are stubs until ``src/model.py`` is implemented.
"""

import pytest

torch = pytest.importorskip("torch")

from src.config import ModelConfig
from src.model import MiniGPT


@pytest.fixture
def config() -> ModelConfig:
    """A tiny model config suitable for fast unit tests."""
    return ModelConfig(
        vocab_size=65,
        block_size=32,
        n_embd=64,
        n_layer=2,
        n_head=4,
        dropout=0.0,
        ffn_multiplier=4,
        bias=False,
    )


@pytest.fixture
def model(config: ModelConfig) -> MiniGPT:
    """Return a small MiniGPT model for testing."""
    return MiniGPT(config)


class TestForwardPass:
    """Tests for :meth:`MiniGPT.forward`."""

    def test_logit_shape(self, model: MiniGPT, config: ModelConfig) -> None:
        """Logits should have shape (B, T, vocab_size).

        TODO: remove ``pytest.skip`` once MiniGPT.forward is implemented.
        """
        pytest.skip("MiniGPT.forward not yet implemented")
        B, T = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        logits, loss = model(input_ids)
        assert logits.shape == (B, T, config.vocab_size)
        assert loss is None

    def test_loss_is_scalar_when_targets_provided(
        self, model: MiniGPT, config: ModelConfig
    ) -> None:
        """When targets are given, loss should be a scalar tensor.

        TODO: remove ``pytest.skip`` once MiniGPT.forward is implemented.
        """
        pytest.skip("MiniGPT.forward not yet implemented")
        B, T = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))
        logits, loss = model(input_ids, targets)
        assert loss is not None
        assert loss.shape == ()  # scalar

    def test_loss_is_positive(self, model: MiniGPT, config: ModelConfig) -> None:
        """Cross-entropy loss should always be positive.

        TODO: remove ``pytest.skip`` once MiniGPT.forward is implemented.
        """
        pytest.skip("MiniGPT.forward not yet implemented")
        B, T = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))
        _, loss = model(input_ids, targets)
        assert loss.item() > 0


class TestParameters:
    """Tests for :meth:`MiniGPT.count_parameters`."""

    def test_parameter_count_positive(self, model: MiniGPT) -> None:
        """A non-trivial model must have > 0 trainable parameters.

        TODO: remove ``pytest.skip`` once count_parameters is implemented.
        """
        pytest.skip("MiniGPT.count_parameters not yet implemented")
        assert model.count_parameters() > 0
