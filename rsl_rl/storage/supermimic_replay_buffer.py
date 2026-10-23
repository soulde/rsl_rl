"""Fixed-size time-major ring replay for SuperMimic."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SuperMimicSequenceBatch:
    """Transition windows with a mask that excludes samples after episode ends."""

    states: torch.Tensor
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    clean_observations: torch.Tensor
    terminated: torch.Tensor
    contexts: torch.Tensor
    goals: torch.Tensor
    reference_bin_ids: torch.Tensor
    done: torch.Tensor
    mask: torch.Tensor


class SuperMimicReplayBuffer:
    """Time-major ring whose transitions are stored together across vector envs."""

    _FIELDS = (
        "state",
        "next_state",
        "observation",
        "next_observation",
        "action",
        "a_prev",
        "reward",
        "clean_observation",
        "next_clean_observation",
        "terminated",
        "done",
        "context",
        "goal",
        "reference_bin_id",
        "reference_motion_id",
        "reference_frame_id",
        "ep_step",
    )
    _OPTIONAL_FIELDS = ("reference_motion_id", "reference_frame_id")

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        sequence_length: int,
        batch_size: int,
        history_length: int = 10,
        device: str | torch.device = "cpu",
    ) -> None:
        if min(capacity, num_envs, sequence_length, batch_size, history_length) <= 0:
            raise ValueError(
                "Replay capacity, environment count, sequence length, batch size, and history must be positive"
            )
        self.capacity, self.num_envs = int(capacity), int(num_envs)
        self.sequence_length, self.batch_size = int(sequence_length), int(batch_size)
        self.history_length = int(history_length)
        self.device = torch.device(device)
        self.data: dict[str, torch.Tensor] = {}
        self.ptr = 0
        self.size = 0
        self.num_episodes = 0
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def add(self, transition: dict[str, torch.Tensor]) -> None:
        """Write one vectorized simulator step, with leading dimension `num_envs`."""
        # Episode step is derived locally so callers cannot accidentally carry
        # a stale counter across vectorized resets.
        missing = (set(self._FIELDS) - set(self._OPTIONAL_FIELDS) - {"ep_step"}) - transition.keys()
        if missing:
            raise ValueError(f"SuperMimic replay transition is missing fields: {sorted(missing)}")
        values = {}
        for key in self._FIELDS:
            if key == "ep_step":
                continue
            if key in transition:
                values[key] = torch.as_tensor(transition[key], device=self.device).detach()
            elif key in self._OPTIONAL_FIELDS:
                values[key] = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        values["ep_step"] = self.episode_steps.clone().to(torch.int32)
        for key, value in values.items():
            if value.ndim == 0 or value.shape[0] != self.num_envs:
                raise ValueError(f"Replay field {key!r} must have leading dimension {self.num_envs}")
        for key in ("reward",):
            values[key] = values[key].to(torch.float32).reshape(self.num_envs)
        for key in ("terminated", "done"):
            values[key] = values[key].to(torch.bool).reshape(self.num_envs)
        for key in ("reference_bin_id", *self._OPTIONAL_FIELDS):
            values[key] = values[key].to(torch.long).reshape(self.num_envs)
        for key in self._FIELDS:
            if key not in {"reward", "terminated", "done", "reference_bin_id", *self._OPTIONAL_FIELDS, "ep_step"}:
                values[key] = values[key].to(torch.float32)
        if not self.data:
            self.data = {
                key: torch.empty((self.capacity, *value.shape), dtype=value.dtype, device=self.device)
                for key, value in values.items()
            }
        for key, value in values.items():
            if value.shape != self.data[key].shape[1:]:
                raise ValueError(f"Replay field {key!r} changed shape after first insertion")
            self.data[key][self.ptr].copy_(value)
        dones = values["done"]
        self.episode_steps.add_(1)
        self.episode_steps[dones] = 0
        self.num_episodes += int(dones.sum().item())
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_windows(self, batch_size: int | None = None) -> SuperMimicSequenceBatch:
        """Sample fixed-length windows and mask transition suffixes after done flags."""
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        if self.size < self.sequence_length:
            raise RuntimeError(f"Replay has {self.size} steps; needs {self.sequence_length} for a window")
        start = torch.randint(self.size - self.sequence_length + 1, (batch_size,), device=self.device)
        env_ids = torch.randint(self.num_envs, (batch_size,), device=self.device)
        ages = start[:, None] + torch.arange(self.sequence_length, device=self.device)
        rows = self._row(ages)
        envs = env_ids[:, None].expand_as(rows)
        out = {key: self.data[key][rows, envs] for key in self._FIELDS}
        mask = torch.cumprod(
            torch.cat((torch.ones(batch_size, 1, device=self.device), 1.0 - out["done"][:, :-1].float()), dim=1), dim=1
        )
        states = torch.cat((out["state"][:, :1], out["next_state"]), dim=1)
        observations = torch.cat((out["observation"][:, :1], out["next_observation"]), dim=1)
        clean_observations = torch.cat((out["clean_observation"][:, :1], out["next_clean_observation"]), dim=1)
        return SuperMimicSequenceBatch(
            states=states,
            observations=observations,
            actions=out["action"],
            rewards=out["reward"].unsqueeze(-1),
            clean_observations=clean_observations,
            terminated=out["terminated"].unsqueeze(-1),
            contexts=out["context"],
            goals=out["goal"],
            reference_bin_ids=out["reference_bin_id"],
            done=out["done"].unsqueeze(-1),
            mask=mask,
        )

    def sample_starts(
        self, batch_size: int | None = None, history_length: int | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Sample arbitrary starts and histories that do not cross episode starts."""
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        history_length = self.history_length if history_length is None else int(history_length)
        if self.size <= 0:
            raise RuntimeError("Cannot sample starts from an empty replay")
        ages = torch.randint(self.size, (batch_size,), device=self.device)
        env_ids = torch.randint(self.num_envs, (batch_size,), device=self.device)
        rows = self._row(ages)
        starts = {
            key: self.data[key][rows, env_ids]
            for key in (
                "state",
                "observation",
                "clean_observation",
                "context",
                "goal",
                "a_prev",
                "reference_bin_id",
                "reference_motion_id",
                "reference_frame_id",
                "ep_step",
            )
        }
        step_in_episode = starts["ep_step"].long()
        offsets = torch.arange(history_length - 1, -1, -1, device=self.device)
        back = torch.minimum(offsets[None, :], step_in_episode[:, None])
        back = torch.minimum(back, ages[:, None])
        history_rows = self._row(ages[:, None] - back)
        history_envs = env_ids[:, None].expand_as(history_rows)
        history = {
            key: self.data[key][history_rows, history_envs]
            for key in (
                "observation",
                "clean_observation",
                "a_prev",
                "context",
                "goal",
                "reference_bin_id",
                "reference_motion_id",
                "reference_frame_id",
                "ep_step",
            )
        }
        return starts, history

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "num_envs": self.num_envs,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "history_length": self.history_length,
            "ptr": self.ptr,
            "size": self.size,
            "num_episodes": self.num_episodes,
            "episode_steps": self.episode_steps.clone(),
            "data": {key: value.clone() for key, value in self.data.items()},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        for key in ("capacity", "num_envs", "sequence_length", "history_length"):
            if int(state_dict[key]) != getattr(self, key):
                raise ValueError(f"Replay checkpoint {key} does not match configuration")
        self.ptr, self.size = int(state_dict["ptr"]), int(state_dict["size"])
        self.num_episodes = int(state_dict["num_episodes"])
        self.episode_steps = state_dict["episode_steps"].to(self.device)
        self.data = {key: value.to(self.device) for key, value in state_dict["data"].items()}
        for key in self._OPTIONAL_FIELDS:
            if key not in self.data:
                self.data[key] = torch.full((self.capacity, self.num_envs), -1, dtype=torch.long, device=self.device)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        return (self.ptr - self.size + age) % self.capacity
