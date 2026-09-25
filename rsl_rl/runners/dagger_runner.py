"""Simulator-independent DAgger collection and optimization orchestration."""

from __future__ import annotations

from collections.abc import Callable

import torch

from rsl_rl.storage import BetaSchedule, DaggerStorage


class DaggerRunner:
    """Collect student-visited states, query a teacher, and train a student."""

    def __init__(
        self,
        student_policy: Callable[[torch.Tensor], torch.Tensor],
        teacher_policy: Callable[[torch.Tensor], torch.Tensor],
        storage: DaggerStorage,
        beta_schedule: BetaSchedule,
    ) -> None:
        self.student_policy = student_policy
        self.teacher_policy = teacher_policy
        self.storage = storage
        self.beta_schedule = beta_schedule
        self.round_index = 0

    def collect(
        self,
        observations: torch.Tensor,
        env_step: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        num_steps: int,
        round_index: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        round_index = self.round_index if round_index is None else round_index
        beta = self.beta_schedule(round_index)
        current = observations
        for _ in range(num_steps):
            with torch.no_grad():
                student_actions = self.student_policy(current)
                teacher_actions = self.teacher_policy(current)
            if student_actions.shape != teacher_actions.shape:
                raise ValueError("student and teacher actions must have matching shapes")
            use_teacher = torch.rand(current.shape[0], generator=generator, device=current.device) < beta
            broadcast_shape = (current.shape[0],) + (1,) * (student_actions.ndim - 1)
            actions = torch.where(use_teacher.reshape(broadcast_shape), teacher_actions, student_actions)
            next_observations, dones = env_step(actions)
            self.storage.add(current, teacher_actions, student_actions, dones, round_index=round_index)
            current = next_observations
        self.round_index = max(self.round_index, round_index + 1)
        return current

    def update(
        self, optimizer: torch.optim.Optimizer, batch_size: int, epochs: int = 1
    ) -> dict[str, float]:
        if len(self.storage) == 0:
            raise ValueError("cannot update an empty DAgger storage")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        losses: list[float] = []
        for _ in range(epochs):
            batch = self.storage.sample(batch_size)
            prediction = self.student_policy(batch["observations"])
            loss = torch.nn.functional.mse_loss(prediction, batch["teacher_actions"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return {"loss": sum(losses) / len(losses)}

    def state_dict(self) -> dict[str, int]:
        return {"round_index": self.round_index}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.round_index = int(state_dict["round_index"])
