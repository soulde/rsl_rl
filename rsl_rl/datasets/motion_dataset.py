"""BeyondMimic-compatible AMP motion dataset."""

from __future__ import annotations

import numpy as np

from .base_motion_dataset import BaseMotionDataset


class BeyondMimicMotionDataset(BaseMotionDataset):
    """Load the BeyondMimic NPZ motion format and convert it to the shared contract."""

    input_quaternion_format = "wxyz"

    def _load_motion_file(self, motion_file, *, body_names, joint_names):
        data = np.load(motion_file, allow_pickle=True)
        if "joint_pos" not in data:
            print(f"[AMP] Skipping {motion_file} (not BeyondMimic format)")
            return None
        if self.key_body_names is not None and body_names is None and "body_names" not in data:
            print(f"[AMP] Skipping {motion_file} (no body_names, required for key_body_names)")
            return None
        return self._tensor_motion(data, self.device, self.input_quaternion_format)


# Backward-compatible name used by existing AMP configuration and project code.
MotionDataset = BeyondMimicMotionDataset
