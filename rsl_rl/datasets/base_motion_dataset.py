"""Shared AMP motion-dataset pipeline."""

from __future__ import annotations

import glob
import numpy as np
import os
import re
import torch
from abc import ABC, abstractmethod


class BaseMotionDataset(ABC):
    """Format-neutral AMP dataset pipeline for NPZ motion records."""

    def __init__(
        self,
        motion_dir: str,
        device: str = "cpu",
        amp_observation_dim: int = 190,
        include_root_height: bool = True,
        observation_profile: str = "default",
        time_between_frames: float = 0.02,
        key_body_names: list[str] | None = None,
        body_names: list[str] | None = None,
        joint_names: list[str] | None = None,
        quaternion_format: str = "wxyz",
        motion_file_pattern: str | None = None,
        motion_files: list[str] | None = None,
        require_transitions: bool = True,
    ):
        self.device = device
        self.amp_observation_dim = amp_observation_dim
        self.include_root_height = include_root_height
        self.observation_profile = observation_profile.lower()
        if self.observation_profile not in {"default", "chocolate"}:
            raise ValueError(f"Unknown AMP observation profile: {observation_profile!r}")
        self.time_between_frames = time_between_frames
        self.key_body_names = key_body_names
        self.joint_names = joint_names
        self.require_transitions = require_transitions
        self.quaternion_format = quaternion_format.lower()
        if self.quaternion_format not in {"wxyz", "xyzw"}:
            raise ValueError("quaternion_format must be 'wxyz' or 'xyzw'")

        self.motions = []
        canonical_body_names = list(body_names) if body_names is not None else None
        discovered_count = len(glob.glob(os.path.join(motion_dir, "*.npz")))
        selected_files = self._select_motion_files(motion_dir, motion_file_pattern, motion_files)
        print(f"[AMP] Selected {len(selected_files)} of {discovered_count} NPZ files in {motion_dir}")
        for motion_file in selected_files:
            motion = self._load_motion_file(motion_file, body_names=body_names, joint_names=joint_names)
            if motion is None:
                continue
            self._normalize_joint_contract(motion, joint_names)
            if canonical_body_names is None:
                embedded_names = motion.get("body_names")
                if embedded_names is not None:
                    canonical_body_names = list(embedded_names)
            self._normalize_body_contract(motion, canonical_body_names)
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
            missing_key_bodies = [name for name in self.key_body_names if name not in self.body_names]
            if missing_key_bodies:
                raise ValueError(f"AMP key bodies are missing from motion body_names: {missing_key_bodies}")
            self.key_body_indices = [self.body_names.index(name) for name in self.key_body_names]
            print(f"[AMP] Key body indices: {self.key_body_indices}")
        elif self.observation_profile == "chocolate":
            raise ValueError("The chocolate AMP observation profile requires key_body_names and motion body_names")

        self.frame_indices = []
        self.reference_frame_indices = []
        self._reference_state_field_names: tuple[str, ...] = ()
        self._validation_applied = False
        for motion_idx, motion in enumerate(self.motions):
            num_frames = motion["joint_pos"].shape[0]
            if num_frames < 1:
                raise ValueError(f"AMP motion {motion_idx} must contain at least one frame")
            if self.require_transitions and num_frames < 2:
                raise ValueError(f"AMP motion {motion_idx} must contain at least two frames")
            self.reference_frame_indices.extend((motion_idx, frame_idx) for frame_idx in range(num_frames))
            for frame_idx in range(num_frames - 1):
                self.frame_indices.append((motion_idx, frame_idx))

        num_bodies = self.motions[0]["body_pos_w"].shape[1]
        num_key_bodies = len(self.key_body_indices) if self.key_body_indices is not None else num_bodies
        if self.observation_profile == "chocolate":
            expected_dim = self.num_joints * 2 + num_key_bodies * 3 + 6
        else:
            expected_dim = self.num_joints * 2 + num_key_bodies * 3 + 12 + int(self.include_root_height)
        if self.amp_observation_dim != expected_dim:
            if self.observation_profile == "chocolate":
                raise ValueError(
                    f"Chocolate AMP observation dimension is {self.amp_observation_dim}, "
                    f"but motion features require {expected_dim}"
                )
            print(f"[AMP WARNING] amp_observation_dim={amp_observation_dim} != expected {expected_dim}")
            print(f"  joints={self.num_joints}, key_bodies={num_key_bodies}")

        observations = [self._all_observations(motion).to(device) for motion in self.motions]
        self._motion_observation_dim = observations[0].shape[-1]
        if self.require_transitions:
            self.transitions = torch.cat([torch.cat((obs[:-1], obs[1:]), dim=-1) for obs in observations])
            if self.transitions.shape[-1] % 2:
                raise RuntimeError("AMP expert transitions must contain equally sized current and next observations")
            self._motion_observation_dim = self.transitions.shape[-1] // 2
        else:
            self.transitions = torch.empty((0, 2 * self._motion_observation_dim), device=device)
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
        if len(self.transitions) == 0:
            raise RuntimeError("AMP motion dataset has no expert transitions")
        reference_frames = self.sample_reference_frames(batch_size)
        return torch.cat((reference_frames["amp_obs"], reference_frames["amp_next_obs"]), dim=-1)

    def set_validate(
        self,
        transition_mask: torch.Tensor,
        reference_state_fields: dict[str, list[torch.Tensor]] | None = None,
    ) -> None:
        """Install a one-time transition mask and attach aligned per-frame reset fields."""
        if self._validation_applied:
            raise RuntimeError("AMP motion dataset validation has already been installed")
        if transition_mask.ndim != 1 or transition_mask.numel() != len(self.frame_indices):
            raise ValueError(
                "AMP transition validation mask must be one-dimensional with one value per reference transition"
            )

        prepared_fields: dict[str, list[torch.Tensor]] = {}
        for field_name, per_motion_values in (reference_state_fields or {}).items():
            if field_name in self.motions[0]:
                raise ValueError(f"AMP reference state field already exists: {field_name}")
            if len(per_motion_values) != len(self.motions):
                raise ValueError(f"AMP reference state field {field_name!r} must have one tensor per motion")
            prepared_values = []
            for motion_index, values in enumerate(per_motion_values):
                values = torch.as_tensor(values, device=self.motions[motion_index]["joint_pos"].device)
                expected_frames = self.motions[motion_index]["joint_pos"].shape[0]
                if values.ndim == 0 or values.shape[0] != expected_frames:
                    raise ValueError(
                        f"AMP reference state field {field_name!r} for motion {motion_index} must contain "
                        f"{expected_frames} frames"
                    )
                prepared_values.append(values)
            prepared_fields[field_name] = prepared_values

        keep = transition_mask.to(device=self.transitions.device, dtype=torch.bool)
        self.transitions = self.transitions[keep]
        keep_values = keep.detach().cpu().tolist()
        self.frame_indices = [frame for frame, valid in zip(self.frame_indices, keep_values, strict=True) if valid]
        for field_name, per_motion_values in prepared_fields.items():
            for motion, values in zip(self.motions, per_motion_values, strict=True):
                motion[field_name] = values
        self._reference_state_field_names = tuple(prepared_fields)
        self._validation_applied = True

    def sample_reference_frames(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample valid AMP transition frames and their aligned reset state.

        Both expert transition sampling and RSI use this path, so motion/frame
        selection, observation construction, and raw state decoding cannot drift.
        The returned row order is the sampled transition order.
        """
        if batch_size < 0:
            raise ValueError("AMP reference batch_size must be non-negative")
        if len(self.frame_indices) == 0:
            raise RuntimeError("AMP motion dataset has no expert transitions")
        if batch_size == 0:
            empty = self.transitions[:0]
            amp_obs, amp_next_obs = empty.split((self._motion_observation_dim,) * 2, dim=-1)
            sampled = {
                "amp_obs": amp_obs,
                "amp_next_obs": amp_next_obs,
                "joint_pos": self.motions[0]["joint_pos"][:0],
                "joint_vel": self.motions[0]["joint_vel"][:0],
                "root_pos": self.motions[0]["body_pos_w"][:0, 0],
                "root_quat_xyzw": self.motions[0]["body_quat_xyzw"][:0, 0],
                "root_lin_vel": self.motions[0]["body_lin_vel_w"][:0, 0],
                "root_ang_vel": self.motions[0]["body_ang_vel_w"][:0, 0],
            }
            for field_name in self._reference_state_field_names:
                sampled[field_name] = self.motions[0][field_name][:0]
            return sampled

        transition_ids = torch.randint(len(self.frame_indices), (batch_size,), device=self.transitions.device)
        transitions = self.transitions[transition_ids]
        amp_obs, amp_next_obs = transitions.split((self._motion_observation_dim,) * 2, dim=-1)
        selected_frames = [self.frame_indices[index] for index in transition_ids.cpu().tolist()]

        state_fields = {
            "joint_pos": ("joint_pos", None),
            "joint_vel": ("joint_vel", None),
            "root_pos": ("body_pos_w", 0),
            "root_quat_xyzw": ("body_quat_xyzw", 0),
            "root_lin_vel": ("body_lin_vel_w", 0),
            "root_ang_vel": ("body_ang_vel_w", 0),
        }
        state_fields.update({field_name: (field_name, None) for field_name in self._reference_state_field_names})
        sampled = {"amp_obs": amp_obs, "amp_next_obs": amp_next_obs}
        for output_name, (motion_field, body_index) in state_fields.items():
            shape = self.motions[0][motion_field].shape[1:]
            if body_index is not None:
                shape = shape[1:]
            sampled[output_name] = torch.empty(
                (batch_size, *shape), dtype=self.motions[0][motion_field].dtype, device=self.transitions.device
            )

        grouped_frames: dict[int, list[tuple[int, int]]] = {}
        for output_index, (motion_index, frame_index) in enumerate(selected_frames):
            grouped_frames.setdefault(motion_index, []).append((output_index, frame_index))
        for motion_index, frame_pairs in grouped_frames.items():
            motion = self.motions[motion_index]
            output_ids = torch.tensor([pair[0] for pair in frame_pairs], device=self.transitions.device)
            frame_ids = torch.tensor([pair[1] for pair in frame_pairs], device=self.transitions.device)
            for output_name, (motion_field, body_index) in state_fields.items():
                source = motion[motion_field][frame_ids]
                if body_index is not None:
                    source = source[:, body_index]
                if output_name == "root_pos":
                    # Place each clip's horizontal trajectory around the
                    # selected environment origin while retaining source height.
                    source = source.clone()
                    source[:, :2] -= motion[motion_field][0, body_index, :2]
                sampled[output_name][output_ids] = source
        return sampled

    def sample_reference_states(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample raw reset states from any frame, including single-frame RSI-only clips."""
        if batch_size < 0:
            raise ValueError("AMP reference batch_size must be non-negative")
        if batch_size == 0:
            motion = self.motions[0]
            return {
                "joint_pos": motion["joint_pos"][:0],
                "joint_vel": motion["joint_vel"][:0],
                "root_pos": motion["body_pos_w"][:0, 0],
                "root_quat_xyzw": motion["body_quat_xyzw"][:0, 0],
                "root_lin_vel": motion["body_lin_vel_w"][:0, 0],
                "root_ang_vel": motion["body_ang_vel_w"][:0, 0],
            }
        if len(self.reference_frame_indices) == 0:
            raise RuntimeError("AMP RSI dataset has no reference frames")

        selected_ids = torch.randint(
            len(self.reference_frame_indices), (batch_size,), device=self.motions[0]["joint_pos"].device
        )
        selected_frames = [self.reference_frame_indices[index] for index in selected_ids.cpu().tolist()]
        state_fields = {
            "joint_pos": ("joint_pos", None),
            "joint_vel": ("joint_vel", None),
            "root_pos": ("body_pos_w", 0),
            "root_quat_xyzw": ("body_quat_xyzw", 0),
            "root_lin_vel": ("body_lin_vel_w", 0),
            "root_ang_vel": ("body_ang_vel_w", 0),
        }
        sampled = {}
        for output_name, (motion_field, body_index) in state_fields.items():
            source = self.motions[0][motion_field]
            shape = source.shape[1:] if body_index is None else source.shape[2:]
            sampled[output_name] = torch.empty(
                (batch_size, *shape), dtype=source.dtype, device=source.device
            )

        grouped_frames: dict[int, list[tuple[int, int]]] = {}
        for output_index, (motion_index, frame_index) in enumerate(selected_frames):
            grouped_frames.setdefault(motion_index, []).append((output_index, frame_index))
        for motion_index, frame_pairs in grouped_frames.items():
            motion = self.motions[motion_index]
            output_ids = torch.tensor([pair[0] for pair in frame_pairs], device=selected_ids.device)
            frame_ids = torch.tensor([pair[1] for pair in frame_pairs], device=selected_ids.device)
            for output_name, (motion_field, body_index) in state_fields.items():
                source = motion[motion_field][frame_ids]
                if body_index is not None:
                    source = source[:, body_index]
                if output_name == "root_pos":
                    source = source.clone()
                    source[:, :2] -= motion[motion_field][0, body_index, :2]
                sampled[output_name][output_ids] = source
        return sampled

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
        if self.observation_profile == "chocolate":
            root_lin_vel_local = torch.matmul(root_rotation.transpose(-1, -2), motion["body_lin_vel_w"][:, 0, :, None])
            root_ang_vel_local = torch.matmul(root_rotation.transpose(-1, -2), motion["body_ang_vel_w"][:, 0, :, None])
            return torch.cat(
                [
                    joint_pos,
                    body_pos_relative,
                    root_lin_vel_local.squeeze(-1),
                    root_ang_vel_local.squeeze(-1),
                    joint_vel,
                ],
                dim=-1,
            )
        root_orientation = root_rotation[..., :2, :].reshape(len(joint_pos), -1)
        return torch.cat(
            [
                *([root_pos[:, 0, 2:3]] if self.include_root_height else []),
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
