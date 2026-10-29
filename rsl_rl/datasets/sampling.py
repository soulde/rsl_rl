"""Distribution-only categorical sampling and adaptive tracker contract."""
from abc import ABC, abstractmethod
import torch


class SamplingTracker(ABC):
    @property
    @abstractmethod
    def probabilities(self) -> torch.Tensor:
        """Return the current categorical distribution without updating statistics."""

    @abstractmethod
    def state_dict(self) -> dict:
        pass

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        pass


class CategoricalSampler:
    """Sample bins, uniformly unless an explicit distribution is supplied."""
    def __init__(self, num_bins: int, device="cpu"):
        if num_bins <= 0:
            raise ValueError("num_bins must be positive")
        self.num_bins = num_bins
        self.device = torch.device(device)

    def sample(self, count: int, probabilities=None, generator=None):
        if count < 0:
            raise ValueError("count must be nonnegative")
        if probabilities is None:
            return torch.randint(self.num_bins, (count,), device=self.device, generator=generator)
        p = torch.as_tensor(probabilities, device=self.device)
        if p.shape != (self.num_bins,) or not torch.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
            raise ValueError("probabilities must be a finite nonnegative vector with positive mass")
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        # multinomial accepts unnormalized weights; avoid changing existing numerical paths.
        return torch.multinomial(p, count, replacement=True, generator=generator)
