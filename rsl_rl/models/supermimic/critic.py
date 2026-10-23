"""Value critic for imagined SuperMimic trajectories."""

from __future__ import annotations

import torch
from torch import nn

from .world_model import _mlp


class SuperMimicCritic(nn.Module):
    """Estimate discounted value from policy observation and optional goal features."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("Critic input_dim must be positive")
        self.network = _mlp(input_dim, hidden_dims, 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)
