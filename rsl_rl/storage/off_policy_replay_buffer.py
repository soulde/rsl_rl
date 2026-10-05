# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay storage for off-policy transition-based algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OffPolicyBatch:
    """A batch of transitions sampled from an off-policy replay buffer."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class OffPolicyReplayBuffer:
    """Fixed-capacity tensor ring buffer for vectorized transition batches."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str | torch.device = "cpu") -> None:
        if capacity <= 0:
            raise ValueError(f"Replay buffer capacity must be positive, got {capacity}.")
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("Observation and action dimensions must be positive.")

        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.observations = torch.empty(self.capacity, self.obs_dim, dtype=torch.float32, device=self.device)
        self.actions = torch.empty(self.capacity, self.action_dim, dtype=torch.float32, device=self.device)
        self.rewards = torch.empty(self.capacity, 1, dtype=torch.float32, device=self.device)
        self.next_observations = torch.empty_like(self.observations)
        self.dones = torch.empty(self.capacity, 1, dtype=torch.float32, device=self.device)
        self.size = 0
        self.position = 0

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Append a batch of transitions, overwriting the oldest entries when full."""
        tensors = (observations, actions, rewards, next_observations, dones)
        if any(tensor.ndim == 0 for tensor in tensors):
            raise ValueError("Replay inputs must have a batch dimension.")
        batch_size = observations.shape[0]
        if any(tensor.shape[0] != batch_size for tensor in tensors):
            raise ValueError("Replay inputs must have the same leading batch dimension.")
        if batch_size == 0:
            return
        expected_shapes = (
            (self.obs_dim,),
            (self.action_dim,),
            (1,),
            (self.obs_dim,),
            (1,),
        )
        for name, tensor, expected_shape in zip(
            ("observations", "actions", "rewards", "next_observations", "dones"), tensors, expected_shapes
        ):
            if tensor.shape[1:] != expected_shape:
                raise ValueError(f"Expected {name} trailing shape {expected_shape}, got {tuple(tensor.shape[1:])}.")

        values = [tensor.detach().to(device=self.device, dtype=torch.float32) for tensor in tensors]
        if batch_size >= self.capacity:
            for target, value in zip(self._fields(), values):
                target.copy_(value[-self.capacity :])
            self.size = self.capacity
            self.position = 0
            return

        indices = (torch.arange(batch_size, device=self.device) + self.position) % self.capacity
        for target, value in zip(self._fields(), values):
            target[indices] = value
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(self, batch_size: int) -> OffPolicyBatch:
        """Sample transitions uniformly with replacement."""
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if batch_size <= 0:
            raise ValueError(f"Sample batch size must be positive, got {batch_size}.")
        indices = torch.randint(self.size, (batch_size,), device=self.device)
        return OffPolicyBatch(*(field[indices] for field in self._fields()))

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        """Return the initialized data and ring cursor for checkpointing."""
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "size": self.size,
            "position": self.position,
            "observations": self.observations[: self.size].clone(),
            "actions": self.actions[: self.size].clone(),
            "rewards": self.rewards[: self.size].clone(),
            "next_observations": self.next_observations[: self.size].clone(),
            "dones": self.dones[: self.size].clone(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore initialized data and cursor from a checkpoint."""
        for name, expected in (("obs_dim", self.obs_dim), ("action_dim", self.action_dim)):
            if int(state_dict[name]) != expected:
                raise ValueError(f"Replay checkpoint {name}={state_dict[name]} does not match configured {expected}.")
        size = int(state_dict["size"])
        if size < 0 or size > self.capacity:
            raise ValueError(f"Replay checkpoint size {size} exceeds configured capacity {self.capacity}.")
        for name, target in zip(
            ("observations", "actions", "rewards", "next_observations", "dones"), self._fields()
        ):
            saved = state_dict[name].to(device=self.device, dtype=target.dtype)
            if saved.shape[0] != size:
                raise ValueError(f"Replay checkpoint field {name} has {saved.shape[0]} entries, expected {size}.")
            target[:size].copy_(saved)
        self.size = size
        self.position = int(state_dict["position"]) % self.capacity
        if size < self.capacity:
            self.position = min(self.position, size)

    def _fields(self) -> tuple[torch.Tensor, ...]:
        return self.observations, self.actions, self.rewards, self.next_observations, self.dones
