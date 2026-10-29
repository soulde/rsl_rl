"""PHP failure statistics, preserving its control-step update schedule."""
import torch
from .sampling import SamplingTracker


class PHPFailureTracker(SamplingTracker):
    def __init__(self, num_bins, device="cpu", alpha=0.001, kernel_size=3, decay=0.8):
        self.num_bins = num_bins
        self.alpha = alpha
        self.current_bin_failed_count = torch.zeros(num_bins, device=device)
        self.bin_failed_count = torch.zeros_like(self.current_bin_failed_count)
        kernel = torch.tensor([decay**i for i in range(kernel_size)], device=device)
        self.kernel = kernel / kernel.sum()

    def observe_failures(self, bins):
        self.current_bin_failed_count += torch.bincount(bins, minlength=self.num_bins)

    def advance(self):
        self.bin_failed_count.mul_(1 - self.alpha).add_(self.current_bin_failed_count, alpha=self.alpha)
        self.current_bin_failed_count.zero_()

    @property
    def probabilities(self):
        p = self.bin_failed_count + 1e-6
        p = torch.nn.functional.pad(p[None, None, :], (0, self.kernel.numel() - 1), mode="replicate")
        p = torch.nn.functional.conv1d(p, self.kernel.view(1, 1, -1)).view(-1)
        p = p + 0.01
        return p / p.sum()

    def state_dict(self):
        return {"num_bins": self.num_bins, "current_bin_failed_count": self.current_bin_failed_count.clone(),
                "bin_failed_count": self.bin_failed_count.clone()}

    def load_state_dict(self, state):
        if state["num_bins"] != self.num_bins:
            raise ValueError("Tracker bin count mismatch")
        for name in ("current_bin_failed_count", "bin_failed_count"):
            value = torch.as_tensor(state[name], device=self.bin_failed_count.device)
            if value.shape != (self.num_bins,) or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError("Invalid failure statistics")
            getattr(self, name).copy_(value)
