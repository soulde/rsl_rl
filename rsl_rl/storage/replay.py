"""Shared batch, sampler, and tensor-ring primitives for replay storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Any

import torch


class ReplayBatch(Mapping[str, Any]):
    """Mapping contract shared by transition and sequence replay batches."""

    def __init__(self, **fields: Any):
        self._fields = fields

    def __getitem__(self, key: str) -> Any:
        return self._fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._fields[name]
        except KeyError as error:
            raise AttributeError(name) from error


class ReplaySampler(ABC):
    """Sampling strategy owned by a replay buffer."""

    @abstractmethod
    def sample(self, buffer, batch_size: int, **kwargs) -> ReplayBatch:
        """Select and assemble a batch from the supplied buffer."""


class TensorRingStorage:
    """Fixed-capacity ring of named tensors with a vectorized leading item axis."""

    def __init__(
        self,
        capacity: int,
        field_specs: Mapping[str, tuple[tuple[int, ...], torch.dtype]] | None = None,
        device="cpu",
    ):
        if capacity <= 0:
            raise ValueError("Ring capacity must be positive")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.field_specs = None if field_specs is None else dict(field_specs)
        self.data = {
            name: torch.empty(self.capacity, *shape, dtype=dtype, device=self.device)
            for name, (shape, dtype) in self.field_specs.items()
        } if self.field_specs is not None else {}
        self.size = 0
        self.position = 0

    def add(self, batch: Mapping[str, torch.Tensor]) -> None:
        if self.field_specs is None:
            self.field_specs = {}
            for name, raw in batch.items():
                value = torch.as_tensor(raw, device=self.device).detach()
                if value.ndim == 0:
                    raise ValueError(f"Ring field {name} must have a batch dimension")
                self.field_specs[name] = (tuple(value.shape[1:]), value.dtype)
                self.data[name] = torch.empty(self.capacity, *value.shape[1:], dtype=value.dtype, device=self.device)
        if batch.keys() != self.field_specs.keys():
            missing = self.field_specs.keys() - batch.keys()
            extra = batch.keys() - self.field_specs.keys()
            raise ValueError(f"Ring fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        values = {}
        batch_size = None
        for name, (shape, dtype) in self.field_specs.items():
            value = torch.as_tensor(batch[name], device=self.device, dtype=dtype).detach()
            if value.ndim == 0 or value.shape[1:] != shape:
                raise ValueError(f"Ring field {name} must have trailing shape {shape}, got {tuple(value.shape)}")
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError("Ring fields must have matching leading batch sizes")
            values[name] = value
        if not batch_size:
            return
        if batch_size >= self.capacity:
            for name, value in values.items():
                self.data[name].copy_(value[-self.capacity :])
            self.size = self.capacity
            self.position = 0
            return
        indexes = (torch.arange(batch_size, device=self.device) + self.position) % self.capacity
        for name, value in values.items():
            self.data[name][indexes] = value
        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def row_indices(self, ages: torch.Tensor) -> torch.Tensor:
        """Convert ages from oldest retained row to physical ring indices."""
        ages = torch.as_tensor(ages, device=self.device, dtype=torch.long)
        return (self.position - self.size + ages) % self.capacity

    def get(self, field: str, ages: torch.Tensor, item_indices: torch.Tensor | None = None) -> torch.Tensor:
        rows = self.row_indices(ages)
        if item_indices is None:
            return self.data[field][rows]
        item_indices = torch.as_tensor(item_indices, device=self.device, dtype=torch.long)
        return self.data[field][rows, item_indices]

    def gather(self, ages: torch.Tensor) -> ReplayBatch:
        return ReplayBatch(**{name: self.get(name, ages) for name in self.field_specs})

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "data": {name: value[: self.size].clone() for name, value in self.data.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("Ring checkpoint capacity does not match configuration")
        size = int(state["size"])
        if not 0 <= size <= self.capacity:
            raise ValueError("Ring checkpoint size is outside the configured capacity")
        if self.field_specs is None:
            self.field_specs = {
                name: (tuple(value.shape[1:]), value.dtype)
                for name, value in state["data"].items()
            }
            self.data = {
                name: torch.empty(self.capacity, *shape, dtype=dtype, device=self.device)
                for name, (shape, dtype) in self.field_specs.items()
            }
        if state["data"].keys() != self.field_specs.keys():
            raise ValueError("Ring checkpoint fields do not match configured fields")
        for name, (shape, dtype) in self.field_specs.items():
            saved = torch.as_tensor(state["data"][name], device=self.device, dtype=dtype)
            if saved.shape == (self.capacity, *shape):
                self.data[name].copy_(saved)
            elif saved.shape == (size, *shape):
                self.data[name][:size].copy_(saved)
            else:
                raise ValueError(f"Ring checkpoint field {name} has the wrong shape")
        self.size = size
        self.position = int(state["position"]) % self.capacity
        if size < self.capacity:
            self.position = min(self.position, size)
