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

        # Get body names from config or motion file
        if body_names is not None:
            self.body_names = body_names
        elif "body_names" in self.motions[0]:
            self.body_names = self.motions[0]["body_names"]
            print(f"[AMP] Body names loaded from motion file: {len(self.body_names)} bodies")
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

    def sample_amp_observations(self, batch_size: int) -> torch.Tensor:
        """Sample random AMP transitions from the motion dataset.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tensor of shape (batch_size, 2 * amp_observation_dim) containing
            concatenated (state, next_state) transitions.
        """
        # Sample random frame indices
        indices = np.random.randint(0, len(self.frame_indices), size=batch_size)

        transitions = []
        for idx in indices:
            motion_idx, frame_idx = self.frame_indices[idx]
            motion = self.motions[motion_idx]

            # Get current frame observation
            state = self._get_observation(motion, frame_idx)

            # Get next frame observation (with wrapping for episode boundaries)
            next_frame_idx = (frame_idx + 1) % motion["joint_pos"].shape[0]
            next_state = self._get_observation(motion, next_frame_idx)

            # Concatenate state and next_state
            transition = torch.cat([state, next_state])
            transitions.append(transition)

        return torch.stack(transitions)

    def _get_observation(self, motion: dict, frame_idx: int) -> torch.Tensor:
        """Extract AMP observation from a single frame.

        AMP observation contains:
        - joint_pos: Joint positions (num_joints,)
        - joint_vel: Joint velocities (num_joints,)
        - body_pos_relative: Key body positions relative to root (K*3,)
        - base_lin_vel: Base linear velocity (3,)
        - base_ang_vel: Base angular velocity (3,)
        - base_height: Base height in world frame (1,)

        Args:
            motion: Motion dictionary containing frame data.
            frame_idx: Index of the frame.

        Returns:
            Tensor of shape (amp_observation_dim,) containing AMP observation.
        """
        joint_pos = motion["joint_pos"][frame_idx]  # (num_joints,)
        joint_vel = motion["joint_vel"][frame_idx]  # (num_joints,)

        # Compute key body positions relative to root (body 0)
        root_pos = motion["body_pos_w"][frame_idx, 0:1, :]  # (1, 3)
        if self.key_body_indices is not None:
            key_body_pos = motion["body_pos_w"][frame_idx, self.key_body_indices, :]  # (K, 3)
        else:
            key_body_pos = motion["body_pos_w"][frame_idx]  # (B, 3) - all bodies
        root_rotation = _quat_wxyz_to_matrix(motion["body_quat_w"][frame_idx, 0])
        body_offsets = key_body_pos - root_pos
        body_pos_relative = torch.matmul(root_rotation.transpose(-1, -2), body_offsets.T).T.flatten()

        root_orientation = _quat_wxyz_to_matrix(motion["body_quat_w"][frame_idx, 0])[:, :2].reshape(-1)
        base_lin_vel = motion["body_lin_vel_w"][frame_idx, 0]  # (3,)
        base_ang_vel = motion["body_ang_vel_w"][frame_idx, 0]  # (3,)
        z_pos = motion["body_pos_w"][frame_idx, 0, 2:3]  # (1,) - z height of root

        return torch.cat(
            [z_pos, root_orientation, base_lin_vel, base_ang_vel, joint_pos, joint_vel, body_pos_relative]
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
