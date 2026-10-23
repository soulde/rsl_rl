"""Return-aware reference-state initialization for SuperMimic tasks."""

from __future__ import annotations

import torch


class ReturnAwareReferenceSampler:
    """Sample one-second reference bins using uniform and return-aware weights."""

    def __init__(
        self,
        dataset,
        num_bins: int,
        uniform_fraction: float = 0.5,
        low_return_fraction: float = 0.4,
        historical_fraction: float = 0.1,
        temperature: float = 25.0,
    ) -> None:
        fractions = torch.tensor([uniform_fraction, low_return_fraction, historical_fraction], dtype=torch.float32)
        if num_bins <= 0 or (fractions < 0).any() or not torch.isclose(fractions.sum(), torch.tensor(1.0)):
            raise ValueError("Sampler requires positive num_bins and mixture fractions summing to one.")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        self.dataset = dataset
        self.num_bins = int(num_bins)
        self.uniform_fraction = float(uniform_fraction)
        self.low_return_fraction = float(low_return_fraction)
        self.historical_fraction = float(historical_fraction)
        self.temperature = float(temperature)
        self.device = torch.device(getattr(dataset, "device", "cpu"))
        if hasattr(dataset, "reference_bins") and len(dataset.reference_bins) != self.num_bins:
            raise ValueError("num_bins must match the dataset's reference_bins count")
        self.return_sums = torch.zeros(self.num_bins, device=self.device)
        self.return_counts = torch.zeros(self.num_bins, dtype=torch.long, device=self.device)
        self.historical_weights = torch.full((self.num_bins,), 1.0 / self.num_bins, device=self.device)
        self._probabilities = self._build_probabilities()

    @property
    def probabilities(self) -> torch.Tensor:
        return self._probabilities.clone()

    def sample_reference_states(self, env_ids: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        env_ids = torch.as_tensor(env_ids, device=self.device)
        if env_ids.ndim != 1:
            raise ValueError("RSI env_ids must be one-dimensional")
        bin_ids = torch.multinomial(self._probabilities, int(env_ids.numel()), replacement=True)
        states = self.dataset.sample_reference_states_by_bin(bin_ids)
        return states, bin_ids

    def update_returns(self, bin_ids: torch.Tensor, episode_returns: torch.Tensor) -> None:
        bin_ids = torch.as_tensor(bin_ids, device=self.device, dtype=torch.long).reshape(-1)
        episode_returns = torch.as_tensor(episode_returns, device=self.device, dtype=torch.float32).reshape(-1)
        if bin_ids.shape != episode_returns.shape:
            raise ValueError("bin_ids and episode_returns must have the same number of entries")
        if not torch.isfinite(episode_returns).all():
            raise ValueError("episode_returns must be finite")
        if (bin_ids < 0).any() or (bin_ids >= self.num_bins).any():
            raise ValueError(f"bin_ids must be in [0, {self.num_bins})")
        self.return_sums.scatter_add_(0, bin_ids, episode_returns)
        self.return_counts.scatter_add_(0, bin_ids, torch.ones_like(bin_ids))
        low_return = self._low_return_weights()
        self.historical_weights.mul_(0.9).add_(low_return, alpha=0.1)
        self._probabilities = self._build_probabilities()

    def state_dict(self) -> dict[str, torch.Tensor | float | int]:
        return {
            "num_bins": self.num_bins,
            "return_sums": self.return_sums.clone(),
            "return_counts": self.return_counts.clone(),
            "historical_weights": self.historical_weights.clone(),
            "probabilities": self._probabilities.clone(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if int(state_dict["num_bins"]) != self.num_bins:
            raise ValueError("Sampler checkpoint num_bins does not match configuration")
        for name in ("return_sums", "return_counts", "historical_weights", "probabilities"):
            value = torch.as_tensor(state_dict[name], device=self.device)
            if value.shape != (self.num_bins,):
                raise ValueError(f"Sampler checkpoint {name} has the wrong shape")
            setattr(self, "_probabilities" if name == "probabilities" else name, value.clone())
        if not torch.isfinite(self._probabilities).all() or not torch.isclose(
            self._probabilities.sum(), torch.tensor(1.0, device=self.device)
        ):
            raise ValueError("Sampler checkpoint probabilities must be finite and normalized")

    def _low_return_weights(self) -> torch.Tensor:
        observed = self.return_counts > 0
        result = torch.zeros(self.num_bins, device=self.device)
        if not observed.any():
            return torch.full_like(result, 1.0 / self.num_bins)
        means = self.return_sums[observed] / self.return_counts[observed]
        scale = means.abs().max()
        if scale <= torch.finfo(means.dtype).eps or torch.allclose(means, means[:1].expand_as(means)):
            result[observed] = 1.0 / int(observed.sum())
        else:
            result[observed] = torch.softmax(-self.temperature * means / scale, dim=0)
        return result

    def _build_probabilities(self) -> torch.Tensor:
        uniform = torch.full((self.num_bins,), 1.0 / self.num_bins, device=self.device)
        return (
            self.uniform_fraction * uniform
            + self.low_return_fraction * self._low_return_weights()
            + self.historical_fraction * self.historical_weights
        )
