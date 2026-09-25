"""Simulator-independent DAgger aggregation storage and schedules."""

from __future__ import annotations

import torch


class BetaSchedule:
    """Teacher-action mixing probability for a DAgger round."""

    def __init__(self, start: float = 1.0, end: float = 0.0, decay: float = 0.95) -> None:
        if not 0.0 <= end <= start <= 1.0 or not 0.0 < decay <= 1.0:
            raise ValueError("require 0 <= end <= start <= 1 and 0 < decay <= 1")
        self.start, self.end, self.decay = start, end, decay

    def __call__(self, round_index: int) -> float:
        if round_index < 0:
            raise ValueError("round_index must be non-negative")
        return max(self.end, self.start * (self.decay**round_index))


class DaggerStorage:
    """FIFO storage for student-visited states and teacher labels."""

    def __init__(self, capacity: int, device: str | torch.device = "cpu") -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.device = torch.device(device)
        self._fields: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return next(iter(self._fields.values())).shape[0] if self._fields else 0

    def add(
        self,
        observations: torch.Tensor,
        teacher_actions: torch.Tensor,
        student_actions: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
        round_index: int = 0,
    ) -> None:
        if (
            observations.ndim == 0
            or teacher_actions.ndim == 0
            or observations.shape[0] != teacher_actions.shape[0]
        ):
            raise ValueError("observations and teacher_actions must have the same non-empty batch dimension")
        batch = observations.shape[0]
        if student_actions is not None and student_actions.shape[0] != batch:
            raise ValueError("student_actions batch dimension does not match observations")
        if dones is not None and dones.shape[0] != batch:
            raise ValueError("dones batch dimension does not match observations")
        incoming = {
            "observations": observations.detach().to(self.device),
            "teacher_actions": teacher_actions.detach().to(self.device),
            "student_actions": (
                student_actions.detach().to(self.device)
                if student_actions is not None
                else torch.empty(batch, 0, device=self.device)
            ),
            "dones": (
                dones.detach().to(self.device, dtype=torch.bool)
                if dones is not None
                else torch.zeros(batch, dtype=torch.bool, device=self.device)
            ),
            "round": torch.full((batch,), round_index, dtype=torch.long, device=self.device),
        }
        if self._fields and set(incoming) != set(self._fields):
            raise ValueError("all additions must provide the same optional fields")
        self._fields = {
            key: torch.cat((self._fields.get(key, value[:0]), value), dim=0)
            for key, value in incoming.items()
        }
        if len(self) > self.capacity:
            self._fields = {key: value[-self.capacity :] for key, value in self._fields.items()}

    def dataset(self) -> dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self._fields.items()}

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        if not 0 < batch_size <= len(self):
            raise ValueError("batch_size must be in [1, len(storage)]")
        indices = torch.randperm(len(self), generator=generator, device=self.device)[:batch_size]
        return {key: value[indices] for key, value in self._fields.items()}

    def clear(self) -> None:
        self._fields.clear()
