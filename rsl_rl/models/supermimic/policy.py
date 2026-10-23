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
        history_length: int = 10,
        hidden_dims: tuple[int, ...] = (256, 256),
        goal_latent_dim: int = 32,
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim, history_length) <= 0 or goal_dim < 0:
            raise ValueError("Actor observation/action dimensions must be positive and goal dimension nonnegative.")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.history_length = history_length
        self.goal_encoder = _mlp(goal_dim, (goal_latent_dim,), goal_latent_dim) if goal_dim else None
        self.network = _mlp(
            observation_dim * history_length + (goal_latent_dim if goal_dim else 0), hidden_dims, action_dim
        )

    def forward(self, observation: torch.Tensor, goal: torch.Tensor | None = None) -> torch.Tensor:
        if goal is None:
            if self.goal_dim:
                raise ValueError("goal input is required when goal_dim is nonzero")
            goal = observation.new_zeros((observation.shape[0], 0))
        if observation.shape[-2:] != (self.history_length, self.observation_dim):
            raise ValueError("Actor observation must contain configured fixed-length history")
        if goal.shape != (observation.shape[0], self.goal_dim):
            raise ValueError("Actor observation or goal dimensions do not match configuration")
        features = observation.flatten(start_dim=1)
        if self.goal_encoder is not None:
            goal = torch.tanh(self.goal_encoder(goal))
            features = torch.cat((features, goal), dim=-1)
        return torch.tanh(self.network(features))
