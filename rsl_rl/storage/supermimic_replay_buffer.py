"""Sequence replay for SuperMimic world-model training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SuperMimicSequenceBatch:
    """Episode-local sequences of transitions and their aligned state targets."""

    states: torch.Tensor
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    clean_observations: torch.Tensor
    terminated: torch.Tensor
    contexts: torch.Tensor
    goals: torch.Tensor
    reference_bin_ids: torch.Tensor


class SuperMimicReplayBuffer:
    """Fixed-capacity replay that samples contiguous windows from closed episodes."""

    _FIELDS = (
        "state", "next_state", "observation", "next_observation", "action", "reward",
        "clean_observation", "next_clean_observation", "terminated", "context", "goal", "reference_bin_id",
    )

    def __init__(
        self,
        capacity: int,
        sequence_length: int,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> None:
        if capacity <= 0 or sequence_length <= 0 or batch_size <= 0:
            raise ValueError("Replay capacity, sequence length, and batch size must be positive.")
        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self._episodes: deque[list[dict[str, torch.Tensor]]] = deque()
        self._current: dict[int, list[dict[str, torch.Tensor]]] = {}
        self._field_shapes: dict[str, torch.Size] | None = None

    @property
    def size(self) -> int:
        return sum(map(len, self._episodes)) + sum(map(len, self._current.values()))

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def add_step(self, env_id: int, transition: dict[str, torch.Tensor]) -> None:
        """Append one transition; `terminated=True` closes its episode."""
        missing = set(self._FIELDS) - transition.keys()
        if missing:
            raise ValueError(f"SuperMimic replay transition is missing fields: {sorted(missing)}")
        values = {
            key: torch.as_tensor(transition[key], device=self.device).detach().clone()
            for key in self._FIELDS
        }
        values["reward"] = values["reward"].to(dtype=torch.float32).reshape(1)
        values["terminated"] = values["terminated"].to(dtype=torch.bool).reshape(1)
        values["reference_bin_id"] = values["reference_bin_id"].to(dtype=torch.long).reshape(())
        for key in self._FIELDS:
            if key not in {"reward", "terminated", "reference_bin_id"} and not values[key].is_floating_point():
                values[key] = values[key].to(dtype=torch.float32)
        if self._field_shapes is None:
            self._field_shapes = {key: value.shape for key, value in values.items()}
        elif any(values[key].shape != self._field_shapes[key] for key in self._FIELDS):
            raise ValueError("SuperMimic replay transition field shapes changed after the first transition.")
        episode = self._current.setdefault(int(env_id), [])
        episode.append(values)
        if bool(values["terminated"].item()):
            self.end_episode(int(env_id))

    def end_episode(self, env_id: int) -> None:
        """Close an episode, including timeouts whose terminal flag is false."""
        episode = self._current.pop(int(env_id), [])
        if episode:
            self._episodes.append(episode)
            self._enforce_capacity()

    def sample(self) -> SuperMimicSequenceBatch:
        eligible = [episode for episode in self._episodes if len(episode) >= self.sequence_length]
        if not eligible:
            raise RuntimeError(f"No completed episode can supply {self.sequence_length} transitions.")
        sequences = []
        for _ in range(self.batch_size):
            episode = eligible[int(torch.randint(len(eligible), ()).item())]
            start = int(torch.randint(len(episode) - self.sequence_length + 1, ()).item())
            sequences.append(episode[start : start + self.sequence_length])

        stack = lambda field: torch.stack([torch.stack([step[field] for step in seq]) for seq in sequences])
        states = torch.stack(
            [torch.stack([seq[0]["state"]] + [step["next_state"] for step in seq]) for seq in sequences]
        )
        observations = torch.stack(
            [torch.stack([seq[0]["observation"]] + [step["next_observation"] for step in seq]) for seq in sequences]
        )
        clean_observations = torch.stack(
            [torch.stack([seq[0]["clean_observation"]] + [step["next_clean_observation"] for step in seq]) for seq in sequences]
        )
        return SuperMimicSequenceBatch(
            states=states,
            observations=observations,
            actions=stack("action"),
            rewards=stack("reward"),
            clean_observations=clean_observations,
            terminated=stack("terminated"),
            contexts=stack("context"),
            goals=stack("goal"),
            reference_bin_ids=stack("reference_bin_id").squeeze(-1) if stack("reference_bin_id").ndim == 3 else stack("reference_bin_id"),
        )

    def state_dict(self) -> dict:
        clone = lambda episodes: [[{key: value.clone() for key, value in step.items()} for step in ep] for ep in episodes]
        return {
            "capacity": self.capacity,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "episodes": clone(self._episodes),
            "current": {env_id: clone([episode])[0] for env_id, episode in self._current.items()},
            "field_shapes": self._field_shapes,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if int(state_dict["sequence_length"]) != self.sequence_length:
            raise ValueError("Replay checkpoint sequence_length does not match configuration.")
        move = lambda episodes: [[{key: value.to(self.device) for key, value in step.items()} for step in ep] for ep in episodes]
        self._episodes = deque(move(state_dict["episodes"]))
        self._current = {int(env_id): move([episode])[0] for env_id, episode in state_dict["current"].items()}
        self._field_shapes = state_dict.get("field_shapes")
        self._enforce_capacity()

    def _enforce_capacity(self) -> None:
        while self.size > self.capacity and len(self._episodes) > 1:
            self._episodes.popleft()
        if self.size > self.capacity and self._episodes:
            episode = self._episodes[0]
            del episode[: len(episode) - self.capacity]
