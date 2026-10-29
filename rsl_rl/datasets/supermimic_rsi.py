"""Reference-data adapter for SuperMimic's return tracker."""
from __future__ import annotations
import torch
from .reference_sampler import ReferenceStateBatch, ReferenceStateSampler
from .sampling import CategoricalSampler
from .supermimic_tracker import SuperMimicReturnTracker


class ReturnAwareReferenceSampler(ReferenceStateSampler):
    def __init__(self, *args, **kwargs):
        self.tracker = SuperMimicReturnTracker(*args, **kwargs)
        self.sampler = CategoricalSampler(self.tracker.num_bins, self.tracker.device)

    @property
    def num_bins(self):
        return self.tracker.num_bins

    @property
    def device(self):
        return self.tracker.device

    @property
    def probabilities(self):
        return self.tracker.probabilities

    @property
    def return_estimates(self):
        return self.tracker.return_estimates

    @property
    def historical_weights(self):
        return self.tracker.historical_weights

    def sample(self, dataset, batch_size):
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        if len(dataset.reference_bins) != self.num_bins:
            raise ValueError("Reference bin count mismatch")
        self.tracker.prepare_sample()
        bins = self.sampler.sample(batch_size, self.tracker.probabilities)
        return ReferenceStateBatch(self._sample_states(dataset, bins), bins)

    def _sample_states(self, dataset, bin_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample a frame within each selected bin, retaining clip/frame IDs."""
        reference_bins = dataset.reference_bins
        decode = getattr(dataset, "_decode_reference_states", None)
        if decode is None:
            states = dataset.sample_reference_states_by_bin(bin_ids)
            return states

        # Match the reference GPU draw, including singleton terminal bins.
        starts = torch.tensor([frames[0][1] for frames in reference_bins], device=self.device)
        motion_ids = torch.tensor([frames[0][0] for frames in reference_bins], device=self.device)
        ends = torch.tensor([frames[-1][1] + 1 for frames in reference_bins], device=self.device)
        last = torch.tensor([len(m["joint_pos"]) - 1 for m in dataset.motions], device=self.device)
        selected_motions = motion_ids[bin_ids]
        start = starts[bin_ids]
        end = torch.minimum(ends[bin_ids], last[selected_motions])
        frames = start + (torch.rand(len(bin_ids), device=self.device) * (end - start).float()).long()
        selected_frames = list(zip(selected_motions.cpu().tolist(), frames.cpu().tolist(), strict=True))
        states = decode(selected_frames)
        with_joint_contract = getattr(dataset, "_with_joint_contract", None)
        if with_joint_contract is not None:
            states = with_joint_contract(states)
        states["reference_motion_ids"] = selected_motions
        states["reference_frame_ids"] = frames
        return states

    def update_returns(self, bin_ids, episode_returns):
        self.tracker.update_returns(bin_ids, episode_returns)

    def state_dict(self):
        return self.tracker.state_dict()

    def load_state_dict(self, state_dict):
        self.tracker.load_state_dict(state_dict)
