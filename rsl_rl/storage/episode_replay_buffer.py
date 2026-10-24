# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode replay storage for sequence-based model learning."""

from __future__ import annotations

from collections import deque
import torch

from .replay import ReplayBatch, ReplaySampler
from .replay_samplers import EpisodeSequenceSampler


class EpisodeSequenceBatch(ReplayBatch):
    """Contiguous state/action/reward sequences sampled from completed episodes."""

    def __init__(self, observations, actions, rewards, terminated):
        super().__init__(observations=observations, actions=actions, rewards=rewards, terminated=terminated)


class EpisodeReplayBuffer:
    """Fixed-capacity replay that stores complete per-environment episodes."""

    def __init__(
        self,
        capacity: int,
        sequence_length: int,
        batch_size: int,
        device: str | torch.device = "cpu",
        sampler: ReplaySampler | None = None,
    ) -> None:
        if capacity <= 0 or sequence_length <= 0 or batch_size <= 0:
            raise ValueError("Episode replay capacity, sequence length, and batch size must be positive.")
        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.sampler = sampler or EpisodeSequenceSampler()
        self._episodes: deque[list[dict[str, torch.Tensor]]] = deque()
        self._current: dict[int, list[dict[str, torch.Tensor]]] = {}
        self._order_queue: deque[tuple[int, int, list[dict[str, torch.Tensor]]]] = deque()
        self.size = 0
        self._obs_dim: int | None = None
        self._action_dim: int | None = None
        self._next_order = 0

    @property
    def num_episodes(self) -> int:
        """Number of completed episodes retained in replay."""
        return len(self._episodes)

    def add_step(
        self,
        env_id: int,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        terminated: bool | torch.Tensor,
    ) -> None:
        """Append one environment transition and close the episode on termination."""
        observation = observation.detach().to(device=self.device, dtype=torch.float32).reshape(-1).clone()
        action = action.detach().to(device=self.device, dtype=torch.float32).reshape(-1).clone()
        reward = reward.detach().to(device=self.device, dtype=torch.float32).reshape(1).clone()
        next_observation = next_observation.detach().to(device=self.device, dtype=torch.float32).reshape(-1).clone()
        if observation.numel() == 0 or action.numel() == 0:
            raise ValueError("Episode replay observations and actions must be non-empty.")
        if observation.shape != next_observation.shape:
            raise ValueError("Current and next observation dimensions must match.")
        if self._obs_dim is None:
            self._obs_dim, self._action_dim = observation.numel(), action.numel()
        if observation.numel() != self._obs_dim or action.numel() != self._action_dim:
            raise ValueError("Episode replay observation/action dimensions changed after the first transition.")
        transition = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
            "terminated": torch.as_tensor(terminated, device=self.device, dtype=torch.float32).reshape(1).clone(),
            "_order": self._next_order,
        }
        self._next_order += 1
        env_id = int(env_id)
        episode = self._current.setdefault(env_id, [])
        episode.append(transition)
        self._order_queue.append((self._next_order - 1, env_id, episode))
        self.size += 1
        self._enforce_capacity()
        if bool(transition["terminated"].item()):
            self.end_episode(env_id)

    def end_episode(self, env_id: int) -> None:
        """Close a trajectory without changing its per-transition terminal flags."""
        transitions = self._current.pop(int(env_id), [])
        if transitions:
            self._episodes.append(transitions)
            self._enforce_capacity()

    def sample(self) -> EpisodeSequenceBatch:
        """Sample fixed-length sequences without crossing episode boundaries."""
        batch = self.sampler.sample(self, self.batch_size)
        return EpisodeSequenceBatch(**batch)

    def state_dict(self) -> dict:
        """Serialize completed and in-progress episode data."""
        return {
            "capacity": self.capacity,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "size": self.size,
            "obs_dim": self._obs_dim,
            "action_dim": self._action_dim,
            "next_order": self._next_order,
            "episodes": [[{key: value.clone() if torch.is_tensor(value) else value for key, value in transition.items()} for transition in episode]
                         for episode in self._episodes],
            "current": {
                env_id: [{key: value.clone() if torch.is_tensor(value) else value for key, value in transition.items()} for transition in episode]
                for env_id, episode in self._current.items()
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore completed and in-progress episodes into the configured buffer."""
        if int(state_dict["sequence_length"]) != self.sequence_length:
            raise ValueError("Episode replay checkpoint sequence_length does not match the configured value.")
        self._episodes = deque(
            [
                [{key: value.to(self.device) if torch.is_tensor(value) else value for key, value in transition.items()} for transition in episode]
                for episode in state_dict["episodes"]
            ]
        )
        self._current = {
            int(env_id): [{key: value.to(self.device) if torch.is_tensor(value) else value for key, value in transition.items()} for transition in episode]
            for env_id, episode in state_dict["current"].items()
        }
        self._obs_dim = state_dict["obs_dim"]
        self._action_dim = state_dict["action_dim"]
        self._next_order = int(state_dict.get("next_order", self.size))
        self.size = sum(len(episode) for episode in self._episodes) + sum(
            len(episode) for episode in self._current.values()
        )
        ordered = []
        for episode in self._episodes:
            ordered.extend((transition["_order"], -1, episode) for transition in episode)
        for env_id, episode in self._current.items():
            ordered.extend((transition["_order"], env_id, episode) for transition in episode)
        self._order_queue = deque(sorted(ordered, key=lambda entry: entry[0]))
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        while self.size > self.capacity:
            if not self._order_queue:
                break
            _order, env_id, episode = self._order_queue.popleft()
            if not episode or episode[0]["_order"] != _order:
                raise RuntimeError("Episode replay FIFO order is inconsistent with stored transitions.")
            episode.pop(0)
            if not episode:
                if self._current.get(env_id) is episode:
                    del self._current[env_id]
                else:
                    for episode_index, completed in enumerate(self._episodes):
                        if completed is episode:
                            del self._episodes[episode_index]
                            break
            self.size -= 1
