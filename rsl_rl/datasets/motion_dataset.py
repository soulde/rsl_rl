# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion dataset for AMP training."""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch


class MotionDataset:
    """Dataset for loading motion capture data for AMP training.

    This dataset loads .npz files containing motion capture data and provides
    methods to sample AMP observations for discriminator training.
    """

    def __init__(
        self,
        motion_dir: str,
        device: str = "cpu",
        amp_observation_dim: int = 190,
        time_between_frames: float = 0.02,
        key_body_names: list[str] | None = None,
        body_names: list[str] | None = None,
        joint_names: list[str] | None = None,
        motion_file_pattern: str | None = None,
        motion_files: list[str] | None = None,
    ):
        """Initialize the motion dataset.

        Args:
            motion_dir: Directory containing .npz motion files.
            device: Device to load tensors on.
            amp_observation_dim: Dimension of AMP observations.
            time_between_frames: Time between frames (dt).
            key_body_names: Names of key bodies for relative position computation.
                           If None, uses all bodies.
            body_names: List of body names in the NPZ file order.
                       Required if key_body_names is specified.
            motion_file_pattern: Regex matched (fullmatch) against NPZ basenames to
                                select a subset of files.
            motion_files: Explicit list of NPZ basenames to load. Mutually
                          exclusive with motion_file_pattern.
        """
        self.device = device
        self.amp_observation_dim = amp_observation_dim
        self.time_between_frames = time_between_frames
        self.key_body_names = key_body_names
        self.joint_names = joint_names

        # Load all motion files (BeyondMimic format only)
        self.motions = []
        discovered_files = sorted(glob.glob(os.path.join(motion_dir, "*.npz")))
        if motion_file_pattern is not None and motion_files is not None:
            raise ValueError("AMP motion_file_pattern and motion_files are mutually exclusive")
        if motion_files is not None:
            available = {os.path.basename(path) for path in discovered_files}
            missing = sorted(set(motion_files) - available)
            if missing:
                raise ValueError(
                    f"AMP motion_files not found in {motion_dir}: {missing}"
                )
            selected = {os.path.basename(path) for path in motion_files}
            motion_files = [path for path in discovered_files if os.path.basename(path) in selected]
            if len(motion_files) != len(selected):
                raise ValueError("AMP motion_files contains duplicate entries")
        elif motion_file_pattern is None:
            motion_files = discovered_files
        else:
            pattern = re.compile(motion_file_pattern)
            motion_files = [path for path in discovered_files if pattern.fullmatch(os.path.basename(path))]
            if not motion_files:
                raise ValueError(
                    f"AMP motion_file_pattern {motion_file_pattern!r} matched no NPZ files in {motion_dir}"
                )
        print(f"[AMP] Selected {len(motion_files)} of {len(discovered_files)} NPZ files in {motion_dir}")

        for motion_file in motion_files:
            data = np.load(motion_file, allow_pickle=True)
            # Skip files that are not in BeyondMimic format
            if "joint_pos" not in data:
                print(f"[AMP] Skipping {motion_file} (not BeyondMimic format)")
                continue
            # Skip files without body_names when key_body_names is specified
            if key_body_names is not None and body_names is None and "body_names" not in data:
                print(f"[AMP] Skipping {motion_file} (no body_names, required for key_body_names)")
                continue
            motion = {
                "fps": int(np.asarray(data["fps"]).reshape(-1)[0]),
                "joint_pos": torch.tensor(data["joint_pos"], dtype=torch.float32, device=device),
                "joint_vel": torch.tensor(data["joint_vel"], dtype=torch.float32, device=device),
                "body_pos_w": torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device),
                "body_quat_w": torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device),
                "body_lin_vel_w": torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device),
                "body_ang_vel_w": torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device),
            }
            if joint_names is not None:
                if motion["joint_pos"].shape[-1] != len(joint_names):
                    raise ValueError(
                        f"AMP joint contract has {len(joint_names)} names but {motion['joint_pos'].shape[-1]} columns"
                    )
                if "joint_names" in data:
                    source_joint_names = [str(name) for name in data["joint_names"].tolist()]
                    if set(source_joint_names) != set(joint_names):
                        raise ValueError("AMP joint contract names do not match motion joint names")
                    reorder = [source_joint_names.index(name) for name in joint_names]
                    motion["joint_pos"] = motion["joint_pos"][:, reorder]
                    motion["joint_vel"] = motion["joint_vel"][:, reorder]
            # Read body_names if available in NPZ
            if "body_names" in data:
                motion["body_names"] = list(data["body_names"])
            self.motions.append(motion)
            print(f"[AMP] Loaded: {motion_file}")

        if len(self.motions) == 0:
            raise RuntimeError(f"No valid BeyondMimic format NPZ files found in {motion_dir}")

        # Get joint dimension from first motion file
        self.num_joints = self.motions[0]["joint_pos"].shape[-1]
        print(f"[AMP] Number of joints: {self.num_joints}")

        # Get body names from config or motion file. An NPZ carrying its own
        # body_names (e.g. soma-retargeter output with extra fixed links)
        # describes its body axis order, so the file wins over the config:
        # resolving key bodies against a mismatched name list would silently
        # pick wrong indices.
        if "body_names" in self.motions[0]:
            self.body_names = self.motions[0]["body_names"]
            print(f"[AMP] Body names loaded from motion file: {len(self.body_names)} bodies")
            if body_names is not None and list(body_names) != list(self.body_names):
                print(
                    f"[AMP WARNING] config body_names ({len(body_names)}) differ from "
                    f"motion file body_names ({len(self.body_names)}); using the file order"
                )
        elif body_names is not None:
            self.body_names = body_names
        else:
            self.body_names = None
            if self.key_body_names is not None:
                print("[AMP WARNING] key_body_names specified but body_names not available")

        # Compute key body indices from names
        self.key_body_indices = None
        if self.key_body_names is not None and self.body_names is not None:
            self.key_body_indices = [self.body_names.index(name) for name in self.key_body_names]
            print(f"[AMP] Key body indices: {self.key_body_indices}")

        # Compute total frames
        total_frames = sum(m["joint_pos"].shape[0] for m in self.motions)
        print(f"[AMP] Total frames: {total_frames}")

        # Pre-compute frame indices for sampling
        self.frame_indices = []
        for motion_idx, motion in enumerate(self.motions):
            num_frames = motion["joint_pos"].shape[0]
            if num_frames < 2:
                raise ValueError(f"AMP motion {motion_idx} must contain at least two frames")
            for frame_idx in range(num_frames - 1):
                self.frame_indices.append((motion_idx, frame_idx))

        # Validate observation dimension
        num_bodies = self.motions[0]["body_pos_w"].shape[1]
        num_key_bodies = len(self.key_body_indices) if self.key_body_indices is not None else num_bodies
        expected_dim = self.num_joints * 2 + num_key_bodies * 3 + 13
        if self.amp_observation_dim != expected_dim:
            print(f"[AMP WARNING] amp_observation_dim={self.amp_observation_dim} != expected {expected_dim}")
            print(f"  joints={self.num_joints}, key_bodies={num_key_bodies}")

        # Preload all expert transitions on the simulation device so that
        # discriminator sampling is a single gather instead of a per-sample
        # Python loop (which left the GPU idle for seconds per iteration).
        observations = [self._all_observations(motion).to(device) for motion in self.motions]
        self.transitions = torch.cat(
            [torch.cat((obs[:-1], obs[1:]), dim=-1) for obs in observations]
        )
        print(f"[AMP] Preloaded {len(self.transitions)} expert transitions on {device}")

    def sample_amp_observations(self, batch_size: int) -> torch.Tensor:
        """Sample random AMP transitions from the motion dataset.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tensor of shape (batch_size, 2 * amp_observation_dim) containing
            concatenated (state, next_state) transitions.
        """
        if batch_size == 0:
            return self.transitions[:0]
        indices = torch.randint(0, len(self.transitions), (batch_size,), device=self.transitions.device)
        return self.transitions[indices]

    def _get_observation(self, motion: dict, frame_idx: int) -> torch.Tensor:
        """AMP observation of a single frame (compatibility wrapper)."""
        return self._all_observations(motion)[frame_idx]

    def _all_observations(self, motion: dict) -> torch.Tensor:
        """Vectorized AMP observations for every frame of one motion."""
        joint_pos = motion["joint_pos"]
        joint_vel = motion["joint_vel"]
        root_pos = motion["body_pos_w"][:, 0:1, :]
        if self.key_body_indices is not None:
            key_body_pos = motion["body_pos_w"][:, self.key_body_indices, :]
        else:
            key_body_pos = motion["body_pos_w"]
        root_rotation = _quat_wxyz_to_matrix_batch(motion["body_quat_w"][:, 0])
        body_offsets = key_body_pos - root_pos
        body_pos_relative = torch.matmul(
            root_rotation.transpose(-1, -2), body_offsets.transpose(-1, -2)
        ).transpose(-1, -2).flatten(start_dim=1)
        root_orientation = root_rotation[..., :2, :].reshape(len(joint_pos), -1)
        return torch.cat(
            [
                root_pos[:, 0, 2:3],
                root_orientation,
                motion["body_lin_vel_w"][:, 0],
                motion["body_ang_vel_w"][:, 0],
                joint_pos,
                joint_vel,
                body_pos_relative,
            ],
            dim=-1,
        )

    def __len__(self) -> int:
        return len(self.frame_indices)


def _quat_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm().clamp_min(1.0e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        )
    ).reshape(3, 3)


def _quat_wxyz_to_matrix_batch(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert (N, 4) WXYZ quaternions to (N, 3, 3) rotation matrices."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
