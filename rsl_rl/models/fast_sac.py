# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural networks used by the FastSAC algorithm."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict


class FastSACActor(nn.Module):
    """Tanh-squashed Gaussian actor for continuous actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("FastSAC observation and action dimensions must be positive.")
        if log_std_min >= log_std_max:
            raise ValueError("log_std_min must be less than log_std_max.")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        layers: list[nn.Module] = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.SiLU()))
            input_dim = hidden_dim
        if not hidden_dims:
            raise ValueError("FastSAC hidden_dims must contain at least one layer.")
        self.mlp = nn.Sequential(*layers)
        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Linear(input_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Gaussian mean and bounded log standard deviation."""
        features = self.mlp(observations)
        log_std = self.log_std(features).tanh()
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return self.mean(features), log_std

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a reparameterized action and its tanh-corrected log probability."""
        mean, log_std = self(observations)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        actions = torch.tanh(pre_tanh)
        log_probability = distribution.log_prob(pre_tanh) - torch.log(1.0 - actions.square() + 1e-6)
        return actions, log_probability.sum(dim=-1, keepdim=True)

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        """Return the squashed Gaussian mean for inference."""
        mean, _ = self(observations)
        return torch.tanh(mean)


class DistributionalCritic(nn.Module):
    """Ensemble of categorical Q-functions with C51 target projection."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        num_atoms: int = 101,
        v_min: float = -20.0,
        v_max: float = 20.0,
        num_q_networks: int = 2,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("Critic observation and action dimensions must be positive.")
        if num_atoms < 2 or v_min >= v_max:
            raise ValueError("Distributional critic requires num_atoms >= 2 and v_min < v_max.")
        if num_q_networks <= 0:
            raise ValueError("num_q_networks must be positive.")
        if not hidden_dims or min(hidden_dims) <= 0:
            raise ValueError("Critic hidden_dims must contain positive dimensions.")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.num_q_networks = num_q_networks
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        networks: list[nn.Module] = []
        for _ in range(num_q_networks):
            layers: list[nn.Module] = []
            input_dim = obs_dim + action_dim
            for hidden_dim in hidden_dims:
                layers.extend((nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()))
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, num_atoms))
            networks.append(nn.Sequential(*layers))
        self.q_networks = nn.ModuleList(networks)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``(num_q_networks, batch, num_atoms)``."""
        inputs = torch.cat((observations, actions), dim=-1)
        return torch.stack([network(inputs) for network in self.q_networks])

    def values(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute expected Q values from categorical logits."""
        return (logits.softmax(dim=-1) * self.support).sum(dim=-1)

    def project(
        self,
        probabilities: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float | torch.Tensor,
    ) -> torch.Tensor:
        """Project Bellman-shifted categorical probabilities onto the fixed support."""
        batch_size = rewards.shape[0]
        rewards = rewards.reshape(batch_size, 1)
        bootstrap = bootstrap.reshape(batch_size, 1).to(dtype=probabilities.dtype)
        discount = torch.as_tensor(discount, device=probabilities.device, dtype=probabilities.dtype)
        target_support = rewards + bootstrap * discount * self.support
        target_support = target_support.clamp(self.v_min, self.v_max)
        spacing = (self.v_max - self.v_min) / (self.num_atoms - 1)
        positions = (target_support - self.v_min) / spacing
        lower = positions.floor().long()
        upper = positions.ceil().long()
        equal = lower == upper
        lower_weight = torch.where(equal, torch.ones_like(positions), upper.to(positions.dtype) - positions)
        upper_weight = torch.where(equal, torch.zeros_like(positions), positions - lower.to(positions.dtype))
        projected = torch.zeros_like(probabilities)
        offsets = torch.arange(batch_size, device=probabilities.device).unsqueeze(1) * self.num_atoms
        projected.view(-1).index_add_(0, (lower + offsets).reshape(-1), (probabilities * lower_weight).reshape(-1))
        projected.view(-1).index_add_(0, (upper + offsets).reshape(-1), (probabilities * upper_weight).reshape(-1))
        return projected


class FastSACPolicy(nn.Module):
    """Deterministic inference wrapper accepting `rsl_rl` TensorDict observations."""

    def __init__(self, actor: FastSACActor, obs_groups: list[str]) -> None:
        super().__init__()
        self.actor = actor
        self.obs_groups = obs_groups

    def forward(self, observations: TensorDict) -> torch.Tensor:
        """Select configured observation groups and return deterministic actions."""
        actor_observations = torch.cat([observations[key] for key in self.obs_groups], dim=-1)
        return self.actor.deterministic(actor_observations)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Provide the policy reset hook used by recurrent policy consumers."""
        del dones
