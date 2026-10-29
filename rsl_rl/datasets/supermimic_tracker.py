"""Return-aware reference-state sampler owned by a motion dataset."""

from __future__ import annotations

import torch

from .sampling import SamplingTracker


class SuperMimicReturnTracker(SamplingTracker):
    """Mix uniform-bin, low-return, and historical sampling for SuperMimic."""

    def __init__(
        self,
        num_bins: int,
        uniform_fraction: float = 0.5,
        low_return_fraction: float = 0.4,
        historical_fraction: float = 0.1,
        temperature: float = 25.0,
        max_return: float = 1.0,
        device="cpu",
    ) -> None:
        fractions = torch.tensor([uniform_fraction, low_return_fraction, historical_fraction], dtype=torch.float32)
        if num_bins <= 0 or (fractions < 0).any() or not torch.isclose(fractions.sum(), torch.tensor(1.0)):
            raise ValueError("Sampler requires positive num_bins and mixture fractions summing to one.")
        if temperature < 0 or max_return <= 0:
            raise ValueError("temperature must be nonnegative and max_return positive")
        self.num_bins = int(num_bins)
        self.uniform_fraction = float(uniform_fraction)
        self.low_return_fraction = float(low_return_fraction)
        self.historical_fraction = float(historical_fraction)
        self.temperature = float(temperature)
        self.max_return = float(max_return)
        self.device = torch.device(device)
        self.return_estimates = torch.zeros(self.num_bins, device=self.device)
        self.historical_weights = torch.full((self.num_bins,), 1.0 / self.num_bins, device=self.device)
        self._probabilities = self._build_probabilities()

    @property
    def probabilities(self) -> torch.Tensor:
        return self._probabilities.clone()

    def prepare_sample(self) -> None:
        soft = self._soft_weights()
        self.historical_weights.mul_(0.999).add_(soft, alpha=0.001)
        self._probabilities = self._build_probabilities(soft)

    def update_returns(self, bin_ids: torch.Tensor, episode_returns: torch.Tensor) -> None:
        bin_ids = torch.as_tensor(bin_ids, device=self.device, dtype=torch.long).reshape(-1)
        episode_returns = torch.as_tensor(episode_returns, device=self.device, dtype=torch.float32).reshape(-1)
        if bin_ids.shape != episode_returns.shape:
            raise ValueError("bin_ids and episode_returns must have the same number of entries")
        if not torch.isfinite(episode_returns).all():
            raise ValueError("episode_returns must be finite")
        if (bin_ids < 0).any() or (bin_ids >= self.num_bins).any():
            raise ValueError(f"bin_ids must be in [0, {self.num_bins})")
        # Match the reference env's batched index_put_ update, including its
        # duplicate-bin behavior when several environments finish together.
        self.return_estimates.index_put_(
            (bin_ids,), 0.8 * self.return_estimates[bin_ids] + 0.2 * episode_returns
        )
        self._probabilities = self._build_probabilities()

    def state_dict(self) -> dict[str, torch.Tensor | float | int]:
        return {
            "num_bins": self.num_bins,
            "return_estimates": self.return_estimates.clone(),
            "historical_weights": self.historical_weights.clone(),
            "probabilities": self._probabilities.clone(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if int(state_dict["num_bins"]) != self.num_bins:
            raise ValueError("Sampler checkpoint num_bins does not match configuration")
        for name in ("return_estimates", "historical_weights", "probabilities"):
            value = torch.as_tensor(state_dict[name], device=self.device)
            if value.shape != (self.num_bins,):
                raise ValueError(f"Sampler checkpoint {name} has the wrong shape")
            setattr(self, "_probabilities" if name == "probabilities" else name, value.clone())
        if not torch.isfinite(self._probabilities).all() or not torch.isclose(
            self._probabilities.sum(), torch.tensor(1.0, device=self.device)
        ):
            raise ValueError("Sampler checkpoint probabilities must be finite and normalized")

    def _soft_weights(self) -> torch.Tensor:
        return torch.softmax(-self.temperature * self.return_estimates / self.max_return, dim=0)

    def _build_probabilities(self, soft: torch.Tensor | None = None) -> torch.Tensor:
        uniform = torch.full((self.num_bins,), 1.0 / self.num_bins, device=self.device)
        soft = self._soft_weights() if soft is None else soft
        historical_sum = self.historical_weights.sum()
        historical = (
            self.historical_weights / historical_sum if historical_sum > torch.finfo(torch.float32).eps else uniform
        )
        return self.uniform_fraction * uniform + self.low_return_fraction * soft + self.historical_fraction * historical
