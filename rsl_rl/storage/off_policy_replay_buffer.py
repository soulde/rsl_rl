# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat transition replay backed by shared tensor-ring storage."""

from __future__ import annotations

import torch

from .replay import ReplayBatch, ReplaySampler, TensorRingStorage
from .replay_samplers import UniformTransitionSampler


class OffPolicyBatch(ReplayBatch):
    """Named transition batch; also implements the shared mapping contract."""

    def __init__(self, observations, actions, rewards, next_observations, dones):
        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
        )


class OffPolicyReplayBuffer:
    """Fixed-capacity tensor ring for vectorized transition batches."""

    _FIELD_NAMES = ("observations", "actions", "rewards", "next_observations", "dones")

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: str | torch.device = "cpu",
        sampler: ReplaySampler | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"Replay buffer capacity must be positive, got {capacity}.")
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("Observation and action dimensions must be positive.")
        self.capacity = int(capacity)
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.device = torch.device(device)
        self.storage = TensorRingStorage(
            self.capacity,
            {
                "observations": ((self.obs_dim,), torch.float32),
                "actions": ((self.action_dim,), torch.float32),
                "rewards": ((1,), torch.float32),
                "next_observations": ((self.obs_dim,), torch.float32),
                "dones": ((1,), torch.float32),
            },
            device=self.device,
        )
        self.sampler = sampler or UniformTransitionSampler()
        for name in self._FIELD_NAMES:
            setattr(self, name, self.storage.data[name])

    @property
    def size(self) -> int:
        return self.storage.size

    @property
    def position(self) -> int:
        return self.storage.position

    def gather(self, ages: torch.Tensor) -> ReplayBatch:
        return self.storage.gather(ages)

    def add(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        tensors = (observations, actions, rewards, next_observations, dones)
        if any(tensor.ndim == 0 for tensor in tensors):
            raise ValueError("Replay inputs must have a batch dimension.")
        batch_size = observations.shape[0]
        if any(tensor.shape[0] != batch_size for tensor in tensors):
            raise ValueError("Replay inputs must have the same leading batch dimension.")
        expected = {
            "observations": (self.obs_dim,),
            "actions": (self.action_dim,),
            "rewards": (1,),
            "next_observations": (self.obs_dim,),
            "dones": (1,),
        }
        for name, tensor in zip(self._FIELD_NAMES, tensors):
            if tensor.shape[1:] != expected[name]:
                raise ValueError(f"Expected {name} trailing shape {expected[name]}, got {tuple(tensor.shape[1:])}.")
        self.storage.add(dict(zip(self._FIELD_NAMES, tensors)))

    def sample(self, batch_size: int) -> OffPolicyBatch:
        sampled = self.sampler.sample(self, batch_size)
        return OffPolicyBatch(**sampled)

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        """Return initialized data and ring cursor for checkpointing."""
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "size": self.size,
            "position": self.position,
            **{name: self.storage.data[name][: self.size].clone() for name in self._FIELD_NAMES},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore initialized data and ring cursor from a checkpoint."""
        for name, expected in (("obs_dim", self.obs_dim), ("action_dim", self.action_dim)):
            if int(state_dict[name]) != expected:
                raise ValueError(f"Replay checkpoint {name}={state_dict[name]} does not match configured {expected}.")
        self.storage.load_state_dict(
            {
                "capacity": self.capacity,
                "size": state_dict["size"],
                "position": state_dict["position"],
                "data": {name: state_dict[name] for name in self._FIELD_NAMES},
            }
        )
