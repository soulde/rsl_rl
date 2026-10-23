# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural building blocks for the TD-MPC2 state-based world model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    """Normalize latent coordinates over fixed-size simplices."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("SimNorm dimension must be positive.")
        self.dim = dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] % self.dim:
            raise ValueError(f"Latent dimension {values.shape[-1]} must be divisible by SimNorm dim {self.dim}.")
        return values.reshape(*values.shape[:-1], -1, self.dim).softmax(dim=-1).reshape_as(values)


class NormedLinear(nn.Module):
    """Linear layer followed by layer normalization, activation, and optional dropout."""

    def __init__(self, input_dim: int, output_dim: int, activation: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim) if activation else nn.Identity()
        self.activation = nn.Mish() if activation else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(self.linear(values))))


def mlp(
    input_dim: int,
    hidden_dims: list[int] | tuple[int, ...],
    output_dim: int,
    activation: nn.Module | None = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build a normalized MLP with a configurable final activation."""
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for index, hidden_dim in enumerate(hidden_dims):
        layers.append(NormedLinear(previous_dim, hidden_dim, dropout=dropout if index == 0 else 0.0))
        previous_dim = hidden_dim
    output = nn.Linear(previous_dim, output_dim)
    layers.append(output if activation is None else nn.Sequential(output, activation))
    return nn.Sequential(*layers)


def encoder(obs_dim: int, enc_dim: int, latent_dim: int, num_layers: int, simnorm_dim: int) -> nn.Sequential:
    """Build the state observation encoder."""
    hidden_dims = [enc_dim] * max(num_layers - 1, 1)
    return mlp(obs_dim, hidden_dims, latent_dim, activation=SimNorm(simnorm_dim))
