from __future__ import annotations

import torch

import pytest

from rsl_rl.modules import CrossAttention, SinusoidalPositionalEncoding


def test_positional_encoding_preserves_shape_and_is_deterministic() -> None:
    """Position features must be repeatable, batch-independent, and time-dependent."""
    x = torch.zeros(2, 7, 9)
    encoding = SinusoidalPositionalEncoding(9)

    first = encoding(x)
    second = encoding(x)

    assert first.shape == x.shape
    assert torch.equal(first, second)
    assert not torch.equal(first[:, 0], first[:, 1])
    assert torch.equal(first[0], first[1])


def test_cross_attention_preserves_batch_query_length_and_width() -> None:
    """Attention output follows the query token dimensions."""
    attention = CrossAttention(embedding_dim=12, num_heads=3)
    query = torch.randn(2, 4, 12)
    key = torch.randn(2, 6, 12)
    value = torch.randn(2, 6, 12)

    result = attention(query, key, value)

    assert result.shape == query.shape


def test_cross_attention_validates_key_padding_mask_shape() -> None:
    """Reject padding masks that cannot cover all keys."""
    attention = CrossAttention(embedding_dim=8, num_heads=2)
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 5, 8)
    value = torch.randn(2, 5, 8)

    with pytest.raises(ValueError, match="key_padding_mask"):
        attention(query, key, value, key_padding_mask=torch.zeros(2, 4, dtype=torch.bool))


def test_cross_attention_validates_batch_dimensions() -> None:
    """Reject queries and memory tensors with different batch sizes."""
    attention = CrossAttention(embedding_dim=8, num_heads=2)
    query = torch.randn(2, 3, 8)
    key = torch.randn(1, 5, 8)
    value = torch.randn(1, 5, 8)

    with pytest.raises(ValueError, match="batch dimension"):
        attention(query, key, value)


def test_cross_attention_can_be_scripted_for_policy_export() -> None:
    """The reusable attention block remains usable by TorchScript model exports."""
    attention = torch.jit.script(CrossAttention(embedding_dim=8, num_heads=2))
    query = torch.randn(2, 1, 8)
    key = torch.randn(2, 5, 8)

    assert attention(query, key, key).shape == query.shape
