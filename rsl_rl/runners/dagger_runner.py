"""Simulator-independent DAgger collection and optimization orchestration."""

from __future__ import annotations

from collections.abc import Callable

import torch
from tensordict.base import TensorDictBase

from rsl_rl.storage import BetaSchedule, DaggerStorage

ObservationBatch = torch.Tensor | TensorDictBase


class DaggerRunner:
    """Collect student-visited states, query a teacher, and train a student."""

    def __init__(
        self,
        student_policy: Callable[[ObservationBatch], torch.Tensor],
        teacher_policy: Callable[[ObservationBatch], torch.Tensor],
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
        observations: ObservationBatch,
        teacher_observations: ObservationBatch | Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        env_step: Callable[[torch.Tensor], tuple[ObservationBatch, ObservationBatch, torch.Tensor]] | int | None = None,
        num_steps: int | None = None,
        round_index: int | None = None,
        generator: torch.Generator | None = None,
    ) -> ObservationBatch | tuple[ObservationBatch, ObservationBatch]:
        # Preserve the original collect(observations, env_step, num_steps) API;
        # grouped DAgger callers pass separate student and teacher observations.
        legacy_call = callable(teacher_observations)
        if legacy_call:
            step_fn = teacher_observations
            teacher_current = observations
            if num_steps is None and isinstance(env_step, int):
                num_steps = env_step
            env_step_fn = step_fn
        else:
            if not callable(env_step):
                raise TypeError("env_step must be callable when student and teacher observations are separate")
            teacher_current = teacher_observations
            env_step_fn = env_step
        if num_steps is None:
            raise TypeError("num_steps is required")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        round_index = self.round_index if round_index is None else round_index
        beta = self.beta_schedule(round_index)
        current = observations
        if current.shape[0] != teacher_current.shape[0]:
            raise ValueError("student and teacher observations must have matching batch dimensions")
        for _ in range(num_steps):
            with torch.no_grad():
                student_actions = self.student_policy(current)
                teacher_actions = self.teacher_policy(teacher_current)
            if student_actions.shape != teacher_actions.shape:
                raise ValueError("student and teacher actions must have matching shapes")
            action_device = student_actions.device
            use_teacher = torch.rand(current.shape[0], generator=generator, device=action_device) < beta
            broadcast_shape = (current.shape[0],) + (1,) * (student_actions.ndim - 1)
            actions = torch.where(use_teacher.reshape(broadcast_shape), teacher_actions, student_actions)
            step_result = env_step_fn(actions)
            if legacy_call:
                if len(step_result) != 2:
                    raise ValueError("legacy env_step must return (next_observations, dones)")
                next_observations, dones = step_result
                next_teacher_observations = next_observations
            else:
                if len(step_result) != 3:
                    raise ValueError("grouped env_step must return (next_student_obs, next_teacher_obs, dones)")
                next_observations, next_teacher_observations, dones = step_result
            self.storage.add(current, teacher_actions, student_actions, dones, round_index=round_index)
            current = next_observations
            teacher_current = next_teacher_observations
        self.round_index = max(self.round_index, round_index + 1)
        if legacy_call:
            return current
        return current, teacher_current

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
            teacher_actions = batch["teacher_actions"].to(prediction.device)
            loss = torch.nn.functional.mse_loss(prediction, teacher_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return {"loss": sum(losses) / len(losses)}

    def state_dict(self) -> dict[str, int]:
        return {"round_index": self.round_index}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.round_index = int(state_dict["round_index"])
