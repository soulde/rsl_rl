"""Reference-state sampler interfaces owned by motion datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class ReferenceStateBatch:
    states: dict[str, torch.Tensor]
    bin_ids: torch.Tensor | None = None


class ReferenceStateSampler(ABC):
    """Strategy interface for drawing reset states from a motion dataset."""

    @abstractmethod
    def sample(self, dataset, batch_size: int) -> ReferenceStateBatch:
        """Sample a batch of dataset states and optional sampler metadata."""

    def update_returns(self, bin_ids: torch.Tensor, returns: torch.Tensor) -> None:
        """Update sampling statistics; uniform samplers intentionally ignore returns."""

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        if state:
            raise ValueError("Uniform reference sampler does not have checkpoint state")


class UniformReferenceStateSampler(ReferenceStateSampler):
    """Use the dataset's ordinary uniform reference-frame sampling behavior."""

    def sample(self, dataset, batch_size: int) -> ReferenceStateBatch:
        return ReferenceStateBatch(dataset.sample_reference_states(batch_size))
