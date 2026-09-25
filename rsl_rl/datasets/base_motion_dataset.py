"""Shared AMP motion-dataset pipeline."""

from __future__ import annotations

import glob
import os
import re
from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseMotionDataset(ABC):
    """Format-neutral AMP dataset pipeline for NPZ motion records."""

    def __init__(
        self,
        motion_dir: str,
        device: str = "cpu",
        amp_observation_dim: int = 190,
        time_between_frames: float = 0.02,
        key_body_names: list[str] | None = None,
        body_names: list[str] | None = None,
        joint_names: list[str] | None = None,
        quaternion_format: str = "wxyz",
        motion_file_pattern: str | None = None,
        motion_files: list[str] | None = None,
    ):
        self.device = device
        self.amp_observation_dim = amp_observation_dim
        self.time_between_frames = time_between_frames
        self.key_body_names = key_body_names
        self.joint_names = joint_names
        self.quaternion_format = quaternion_format.lower()
        if self.quaternion_format not in {"wxyz", "xyzw"}:
            raise ValueError("quaternion_format must be 'wxyz' or 'xyzw'")

        self.motions = []
        discovered_count = len(glob.glob(os.path.join(motion_dir, "*.npz")))
        selected_files = self._select_motion_files(motion_dir, motion_file_pattern, motion_files)
        print(f"[AMP] Selected {len(selected_files)} of {discovered_count} NPZ files in {motion_dir}")
        for motion_file in selected_files:
            motion = self._load_motion_file(motion_file, body_names=body_names, joint_names=joint_names)
            if motion is None:
                continue
            self._normalize_joint_contract(motion, joint_names)
            self._normalize_body_contract(motion, body_names)
            self.motions.append(motion)
            print(f"[AMP] Loaded: {motion_file}")

        if len(self.motions) == 0:
            raise RuntimeError(f"No valid AMP motion files found in {motion_dir}")

        self.num_joints = self.motions[0]["joint_pos"].shape[-1]
        print(f"[AMP] Number of joints: {self.num_joints}")
        embedded_body_names = self.motions[0].get("body_names")
        if embedded_body_names is not None:
            self.body_names = embedded_body_names
            print(f"[AMP] Body names loaded from motion file: {len(self.body_names)} bodies")
            if body_names is not None and list(body_names) != list(self.body_names):
                print(
                    f"[AMP WARNING] config body_names ({len(body_names)}) differ from "
                    f"motion file body_names ({len(self.body_names)}); using the file order"
                )
        else:
            self.body_names = body_names
            if self.key_body_names is not None and self.body_names is None:
                print("[AMP WARNING] key_body_names specified but body_names not available")

        self.key_body_indices = None
        if self.key_body_names is not None and self.body_names is not None:
            self.key_body_indices = [self.body_names.index(name) for name in self.key_body_names]
            print(f"[AMP] Key body indices: {self.key_body_indices}")

        self.frame_indices = []
        for motion_idx, motion in enumerate(self.motions):
            num_frames = motion["joint_pos"].shape[0]
            if num_frames < 2:
                raise ValueError(f"AMP motion {motion_idx} must contain at least two frames")
            for frame_idx in range(num_frames - 1):
                self.frame_indices.append((motion_idx, frame_idx))

        num_bodies = self.motions[0]["body_pos_w"].shape[1]
        num_key_bodies = len(self.key_body_indices) if self.key_body_indices is not None else num_bodies
        expected_dim = self.num_joints * 2 + num_key_bodies * 3 + 13
        if self.amp_observation_dim != expected_dim:
            print(f"[AMP WARNING] amp_observation_dim={amp_observation_dim} != expected {expected_dim}")
            print(f"  joints={self.num_joints}, key_bodies={num_key_bodies}")

        observations = [self._all_observations(motion).to(device) for motion in self.motions]
        self.transitions = torch.cat([torch.cat((obs[:-1], obs[1:]), dim=-1) for obs in observations])
        print(f"[AMP] Preloaded {len(self.transitions)} expert transitions on {device}")

    @staticmethod
    def _select_motion_files(motion_dir, motion_file_pattern, motion_files):
        discovered_files = sorted(glob.glob(os.path.join(motion_dir, "*.npz")))
        if motion_file_pattern is not None and motion_files is not None:
            raise ValueError("AMP motion_file_pattern and motion_files are mutually exclusive")
        if motion_files is not None:
            available = {os.path.basename(path) for path in discovered_files}
            missing = sorted(set(motion_files) - available)
            if missing:
                raise ValueError(f"AMP motion_files not found in {motion_dir}: {missing}")
            selected = set(motion_files)
            motion_files = [path for path in discovered_files if os.path.basename(path) in selected]
            if len(motion_files) != len(selected):
                raise ValueError("AMP motion_files contains duplicate entries")
            return motion_files
        if motion_file_pattern is None:
            return discovered_files
        pattern = re.compile(motion_file_pattern)
        motion_files = [path for path in discovered_files if pattern.fullmatch(os.path.basename(path))]
        if not motion_files:
            raise ValueError(f"AMP motion_file_pattern {motion_file_pattern!r} matched no NPZ files in {motion_dir}")
        return motion_files

    @abstractmethod
    def _load_motion_file(self, motion_file, *, body_names, joint_names):
        """Load one format-specific motion record."""

    @staticmethod
    def _tensor_motion(data, device, quaternion_format):
        body_quat = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        body_quat_xyzw = body_quat if quaternion_format == "xyzw" else body_quat[..., [1, 2, 3, 0]]
        motion = {
            "fps": float(np.asarray(data["fps"]).reshape(-1)[0]),
            "joint_pos": torch.tensor(data["joint_pos"], dtype=torch.float32, device=device),
            "joint_vel": torch.tensor(data["joint_vel"], dtype=torch.float32, device=device),
            "body_pos_w": torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device),
            # Keep the source field for compatibility and expose the Isaac
            # Sim XYZW contract explicitly for consumers that need quaternions.
            "body_quat_w": body_quat,
            "body_quat_xyzw": body_quat_xyzw,
            "body_lin_vel_w": torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device),
            "body_ang_vel_w": torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device),
        }
        for key in ("joint_names", "body_names"):
            if key in data:
                motion[key] = [str(name) for name in np.asarray(data[key]).tolist()]
        return motion

    @staticmethod
    def _normalize_body_contract(motion, body_names):
        source_body_names = motion.get("body_names")
        if source_body_names is None:
            if body_names is not None:
                motion["body_names"] = list(body_names)
            return
        if body_names is None:
            return
        if len(source_body_names) != len(body_names) or set(source_body_names) != set(body_names):
            raise ValueError("AMP body contract names do not match motion body names")
        reorder = [source_body_names.index(name) for name in body_names]
        for key in ("body_pos_w", "body_quat_w", "body_quat_xyzw", "body_lin_vel_w", "body_ang_vel_w"):
            motion[key] = motion[key][:, reorder]
        motion["body_names"] = list(body_names)

    @staticmethod
    def _normalize_joint_contract(motion, joint_names):
        if joint_names is None:
            return
        if motion["joint_pos"].shape[-1] != len(joint_names):
            raise ValueError(
                f"AMP joint contract has {len(joint_names)} names but "
                f"{motion['joint_pos'].shape[-1]} columns"
            )
        source_joint_names = motion.get("joint_names")
        if source_joint_names is None:
            return
        if set(source_joint_names) != set(joint_names):
            raise ValueError("AMP joint contract names do not match motion joint names")
        reorder = [source_joint_names.index(name) for name in joint_names]
        motion["joint_pos"] = motion["joint_pos"][:, reorder]
        motion["joint_vel"] = motion["joint_vel"][:, reorder]

    def sample_amp_observations(self, batch_size: int) -> torch.Tensor:
        if batch_size == 0:
            return self.transitions[:0]
        indices = torch.randint(0, len(self.transitions), (batch_size,), device=self.transitions.device)
        return self.transitions[indices]

    def _get_observation(self, motion, frame_idx):
        return self._all_observations(motion)[frame_idx]

    def _all_observations(self, motion):
        joint_pos = motion["joint_pos"]
        joint_vel = motion["joint_vel"]
        root_pos = motion["body_pos_w"][:, 0:1, :]
        if self.key_body_indices is not None:
            key_body_pos = motion["body_pos_w"][:, self.key_body_indices, :]
        else:
            key_body_pos = motion["body_pos_w"]
        # Isaac Sim AMP observations use rotation matrices; the source
        # quaternions have already been normalized to the explicit XYZW field.
        root_rotation = _quat_xyzw_to_matrix_batch(motion["body_quat_xyzw"][:, 0])
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

    def __len__(self):
        return len(self.frame_indices)


def _quat_xyzw_to_matrix_batch(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    x, y, z, w = quaternion.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
