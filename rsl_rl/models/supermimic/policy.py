"""Deterministic goal-conditioned policy for SuperMimic."""

from __future__ import annotations

import torch
from torch import nn

from .world_model import _mlp


class SuperMimicActor(nn.Module):
    """Map clean policy observations and optional goal features to bounded actions."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        goal_dim: int = 0,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim) <= 0 or goal_dim < 0:
            raise ValueError("Actor observation/action dimensions must be positive and goal dimension nonnegative.")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.network = _mlp(observation_dim + goal_dim, hidden_dims, action_dim)

    def forward(self, observation: torch.Tensor, goal: torch.Tensor | None = None) -> torch.Tensor:
        if goal is None:
            if self.goal_dim:
                raise ValueError("goal input is required when goal_dim is nonzero")
            goal = observation.new_zeros((*observation.shape[:-1], 0))
        if observation.shape[-1] != self.observation_dim or goal.shape != (*observation.shape[:-1], self.goal_dim):
            raise ValueError("Actor observation or goal dimensions do not match configuration")
        return torch.tanh(self.network(torch.cat((observation, goal), dim=-1)))
