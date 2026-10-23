# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reusable Transformer building blocks."""

from __future__ import annotations

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal position features to batch-first sequence embeddings."""

    def __init__(self, embedding_dim: int, max_len: int = 10000) -> None:
        """Create position encodings for a fixed embedding width and maximum length."""
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        encoding = torch.zeros(max_len, embedding_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``[batch, time, embedding_dim]`` input."""
        if x.ndim != 3:
            raise ValueError("expected a rank-3 [batch, time, width] tensor")
        if x.shape[-1] != self.embedding_dim:
            raise ValueError(f"input width must be {self.embedding_dim}, got {x.shape[-1]}")
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(f"sequence length {x.shape[1]} exceeds configured max_len {self.encoding.shape[1]}")
        position_features = self.encoding[:, : x.shape[1]].to(dtype=x.dtype)
        return x + position_features


class CrossAttention(nn.Module):
    """Batch-first multi-head attention from query tokens to key/value tokens."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        """Create attention with a shared query/key/value embedding width."""
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by positive num_heads ({num_heads})")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        self.embedding_dim = embedding_dim
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend from query tokens to key/value tokens and return query-shaped features."""
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query, key, and value must have rank 3 [batch, sequence, width]")
        if query.shape[-1] != self.embedding_dim:
            raise ValueError(f"query width must be {self.embedding_dim}, got {query.shape[-1]}")
        if key.shape[-1] != self.embedding_dim:
            raise ValueError(f"key width must be {self.embedding_dim}, got {key.shape[-1]}")
        if value.shape[-1] != self.embedding_dim:
            raise ValueError(f"value width must be {self.embedding_dim}, got {value.shape[-1]}")
        if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
            raise ValueError("query, key, and value must have the same batch dimension")
        if key.shape[1] != value.shape[1]:
            raise ValueError("key and value must have the same sequence length")

        if key_padding_mask is not None and (
            key_padding_mask.shape[0] != query.shape[0]
            or key_padding_mask.shape[1] != key.shape[1]
            or key_padding_mask.dtype != torch.bool
        ):
            raise ValueError("key_padding_mask must be a bool tensor with shape [batch, key_len]")
        if attn_mask is not None:
            valid_2d_shape = (
                attn_mask.dim() == 2
                and attn_mask.shape[0] == query.shape[1]
                and attn_mask.shape[1] == key.shape[1]
            )
            valid_3d_shape = (
                attn_mask.dim() == 3
                and attn_mask.shape[0] == query.shape[0] * self.attention.num_heads
                and attn_mask.shape[1] == query.shape[1]
                and attn_mask.shape[2] == key.shape[1]
            )
            if not valid_2d_shape and not valid_3d_shape:
                raise ValueError(
                    "attn_mask must have shape [query_len, key_len] or [batch*num_heads, query_len, key_len]"
                )
            if attn_mask.dtype != torch.bool and not torch.is_floating_point(attn_mask):
                raise ValueError("attn_mask must have bool or floating-point dtype")

        output, _ = self.attention(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return output
