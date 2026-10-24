"""Shared AMP motion-dataset pipeline."""

from __future__ import annotations

import glob
import itertools
import numpy as np
import os
import re
import torch
from abc import ABC, abstractmethod

from .amp_features import amp_features_to_heading_frame, amp_features_to_body_frame
from .reference_sampler import ReferenceStateSampler, UniformReferenceStateSampler


class BaseMotionDataset(ABC):
    """NPZ motion pipeline with a shared world- or heading-axis AMP feature layout.

    Features concatenate root linear/angular velocity, joint position/velocity,
    and root-relative key-point positions, in that order. Loaded quaternions
    use XYZW; subclasses define the input file order for their dataset format.
    """

    reference_joint_pos_field = "joint_pos"
    reference_joint_vel_field = "joint_vel"

    def __init__(
        self,
        motion_dir: str,
        device: str = "cpu",
        amp_observation_dim: int | None = None,
        time_between_frames: float = 0.02,
        key_body_names: list[str] | None = None,
        body_names: list[str] | None = None,
        joint_names: list[str] | None = None,
        motion_file_pattern: str | None = None,
        motion_files: list[str] | None = None,
        require_transitions: bool = True,
        amp_coordinate_frame: str = "world",
        marker_body_points: dict | None = None,
    ):
        if amp_coordinate_frame not in ("world", "heading", "body"):
            raise ValueError("amp_coordinate_frame must be 'world', 'heading', or 'body'")
        self.amp_coordinate_frame = amp_coordinate_frame
        self.device = device
        self.amp_observation_dim = amp_observation_dim
        self.time_between_frames = time_between_frames
        self.marker_body_points = marker_body_points or {}
        self.key_body_names = key_body_names
        self.joint_names = joint_names
        self.require_transitions = require_transitions
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

        self.marker_points = []
        for name, point in self.marker_body_points.items():
            body_name = point["body_name"]
            position = point["position"]
            if self.body_names is None or body_name not in self.body_names:
                raise ValueError(f"AMP marker {name} has unknown body: {body_name}")
            if len(position) != 3 or not np.isfinite(position).all():
                raise ValueError(f"AMP marker {name} requires a finite 3D local position")
            self.marker_points.append((self.body_names.index(body_name), position))

        self.frame_indices = []
        self.reference_frame_indices = []
        self.reference_bins: list[list[tuple[int, int]]] = []
        self._reference_state_field_names: tuple[str, ...] = ()
        self._validation_applied = False
        for motion_idx, motion in enumerate(self.motions):
            num_frames = motion["joint_pos"].shape[0]
            if num_frames < 1:
                raise ValueError(f"AMP motion {motion_idx} must contain at least one frame")
            if self.require_transitions and num_frames < 2:
                raise ValueError(f"AMP motion {motion_idx} must contain at least two frames")
            self.reference_frame_indices.extend((motion_idx, frame_idx) for frame_idx in range(num_frames))
            frames_per_second = float(motion["fps"])
            motion_bin_count = int(np.ceil(num_frames / frames_per_second))
            for bin_index in range(motion_bin_count):
                start_frame = int(np.floor(bin_index * frames_per_second))
                end_frame = min(int(np.floor((bin_index + 1) * frames_per_second)), num_frames)
                self.reference_bins.append([(motion_idx, frame_idx) for frame_idx in range(start_frame, end_frame)])
            for frame_idx in range(num_frames - 1):
                self.frame_indices.append((motion_idx, frame_idx))

        num_bodies = self.motions[0]["body_pos_w"].shape[1]
        num_key_bodies = len(self.key_body_indices) if self.key_body_indices is not None else num_bodies
        expected_dim = self.num_joints + self.motions[0]["joint_vel"].shape[-1] + (num_key_bodies + len(self.marker_points)) * 3 + 6
        if self.amp_observation_dim is None:
            self.amp_observation_dim = expected_dim
        elif self.amp_observation_dim != expected_dim:
            raise ValueError(
                f"AMP observation dimension is {self.amp_observation_dim}, but motion features require {expected_dim}"
            )

        observations = [self._all_observations(motion).to(device) for motion in self.motions]
        self._motion_observation_dim = observations[0].shape[-1]
        if self.require_transitions:
            self.transitions = torch.cat([torch.cat((obs[:-1], obs[1:]), dim=-1) for obs in observations])
            if self.transitions.shape[-1] % 2:
                raise RuntimeError("AMP expert transitions must contain equally sized current and next observations")
            self._motion_observation_dim = self.transitions.shape[-1] // 2
        else:
            self.transitions = torch.empty((0, 2 * self._motion_observation_dim), device=device)
        self.reference_state_sampler = UniformReferenceStateSampler()
        print(f"[AMP] Preloaded {len(self.transitions)} expert transitions on {device}")

    def set_reference_state_sampler(self, sampler: ReferenceStateSampler) -> None:
        if not isinstance(sampler, ReferenceStateSampler):
            raise TypeError("reference_state_sampler must implement ReferenceStateSampler")
        self.reference_state_sampler = sampler

    def sample_reference_state_batch(self, batch_size: int):
        """Sample reference states through the sampler owned by this dataset."""
        return self.reference_state_sampler.sample(self, batch_size)

    def update_reference_state_sampler(self, bin_ids: torch.Tensor, returns: torch.Tensor) -> None:
        """Pass completed-episode returns to the dataset's sampling strategy."""
        self.reference_state_sampler.update_returns(bin_ids, returns)

    def reference_state_sampler_state_dict(self) -> dict:
        """Return checkpoint state for the currently configured sampling strategy."""
        return self.reference_state_sampler.state_dict()

    def load_reference_state_sampler_state_dict(self, state: dict) -> None:
        """Restore checkpoint state into the currently configured sampling strategy."""
        self.reference_state_sampler.load_state_dict(state)

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
        if quaternion_format not in {"wxyz", "xyzw"}:
            raise ValueError("quaternion_format must be 'wxyz' or 'xyzw'")
        body_quat = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        body_quat_xyzw = body_quat if quaternion_format == "xyzw" else body_quat[..., [1, 2, 3, 0]]
        motion = {
            "fps": float(np.asarray(data["fps"]).reshape(-1)[0]),
            "joint_pos": torch.tensor(data["joint_pos"], dtype=torch.float32, device=device),
            "joint_vel": torch.tensor(data["joint_vel"], dtype=torch.float32, device=device),
            "body_pos_w": torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device),
            # All loaded body quaternions use XYZW, independently of file order.
            "body_quat_w": body_quat_xyzw,
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
                "joint_pos": self.motions[0][self.reference_joint_pos_field][:0],
                "joint_vel": self.motions[0][self.reference_joint_vel_field][:0],
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
            "joint_pos": (self.reference_joint_pos_field, None),
            "joint_vel": (self.reference_joint_vel_field, None),
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
                "joint_pos": motion[self.reference_joint_pos_field][:0],
                "joint_vel": motion[self.reference_joint_vel_field][:0],
                "root_pos": motion["body_pos_w"][:0, 0],
                "root_quat_xyzw": motion["body_quat_xyzw"][:0, 0],
                "root_lin_vel": motion["body_lin_vel_w"][:0, 0],
                "root_ang_vel": motion["body_ang_vel_w"][:0, 0],
                **{name: motion[name][:0] for name in self._reference_state_field_names},
            }
        if len(self.reference_frame_indices) == 0:
            raise RuntimeError("AMP RSI dataset has no reference frames")

        selected_ids = torch.randint(
            len(self.reference_frame_indices), (batch_size,), device=self.motions[0]["joint_pos"].device
        )
        selected_frames = [self.reference_frame_indices[index] for index in selected_ids.cpu().tolist()]
        return self._decode_reference_states(selected_frames)

    def sample_reference_states_by_bin(self, bin_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Sample an aligned reset state uniformly within each one-second motion bin."""
        bin_ids = torch.as_tensor(bin_ids, device=self.motions[0]["joint_pos"].device, dtype=torch.long)
        if bin_ids.ndim != 1:
            raise ValueError("AMP reference bin IDs must be one-dimensional")
        if bin_ids.numel() == 0:
            return self.sample_reference_states(0)
        if (bin_ids < 0).any() or (bin_ids >= len(self.reference_bins)).any():
            raise ValueError(f"AMP reference bin IDs must be in [0, {len(self.reference_bins)})")
        selected_frames = []
        for bin_id in bin_ids.cpu().tolist():
            frames = self.reference_bins[bin_id]
            if not frames:
                raise RuntimeError(f"AMP reference bin {bin_id} contains no frames")
            selected_frames.append(frames[int(torch.randint(len(frames), ()).item())])
        return self._decode_reference_states(selected_frames)

    def _decode_reference_states(self, selected_frames: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
        """Decode aligned reset fields for the selected motion/frame pairs."""
        batch_size = len(selected_frames)
        state_fields = {
            "joint_pos": (self.reference_joint_pos_field, None),
            "joint_vel": (self.reference_joint_vel_field, None),
            "root_pos": ("body_pos_w", 0),
            "root_quat_xyzw": ("body_quat_xyzw", 0),
            "root_lin_vel": ("body_lin_vel_w", 0),
            "root_ang_vel": ("body_ang_vel_w", 0),
        }
        state_fields.update({name: (name, None) for name in self._reference_state_field_names})
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
            output_ids = torch.tensor([pair[0] for pair in frame_pairs], device=self.motions[0]["joint_pos"].device)
            frame_ids = torch.tensor([pair[1] for pair in frame_pairs], device=self.motions[0]["joint_pos"].device)
            for output_name, (motion_field, body_index) in state_fields.items():
                source = motion[motion_field][frame_ids]
                if body_index is not None:
                    source = source[:, body_index]
                if output_name == "root_pos":
                    source = source.clone()
                    source[:, :2] -= motion[motion_field][0, body_index, :2]
                sampled[output_name][output_ids] = source
        return sampled

    def configure_rsi_support_height_adjustment(self, support_body_boxes, clearance: float = 0.005) -> None:
        """Precompute compact RSI-only support corrections from loaded body poses."""
        if not np.isfinite(clearance) or clearance < 0:
            raise ValueError("RSI support clearance must be finite and nonnegative")
        if not support_body_boxes:
            return
        if self._reference_state_field_names:
            raise RuntimeError("RSI support height adjustment is already configured")
        signs = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=3)), dtype=torch.float32, device=self.device)
        height_offsets, velocity_offsets = [], []
        for motion in self.motions:
            if not motion.get("body_names"):
                raise ValueError("RSI support correction requires body_names in the motion dataset")
            fps = float(motion["fps"])
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("RSI support correction requires a finite positive motion FPS")
            lowest_parts = []
            for name, (center, size) in support_body_boxes.items():
                if name not in motion["body_names"]:
                    raise ValueError(f"RSI support body {name!r} is missing from the motion dataset")
                center = torch.as_tensor(center, dtype=torch.float32, device=self.device)
                size = torch.as_tensor(size, dtype=torch.float32, device=self.device)
                if center.shape != (3,) or size.shape != (3,) or not bool((size > 0).all()):
                    raise ValueError(f"Invalid RSI support box for {name}")
                body_id = motion["body_names"].index(name)
                corners = center + signs * size / 2.0
                quat = motion["body_quat_xyzw"][:, body_id]
                norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
                if not bool(torch.isfinite(quat).all()) or bool((norm < 1e-8).any()):
                    raise ValueError(f"Invalid RSI support quaternion for {name}")
                quat = quat / norm
                xyz, w = quat[:, None, :3], quat[:, None, 3:4]
                local = corners[None].expand(len(quat), -1, -1)
                twice_cross = 2.0 * torch.cross(xyz.expand_as(local), local, dim=-1)
                rotated = local + w * twice_cross + torch.cross(xyz.expand_as(local), twice_cross, dim=-1)
                lowest_parts.append(motion["body_pos_w"][:, body_id, 2] + rotated[..., 2].min(dim=1).values)
            lowest = torch.stack(lowest_parts).min(dim=0).values
            if not bool(torch.isfinite(lowest).all()):
                raise ValueError("Non-finite RSI support height")
            offset = clearance - torch.quantile(lowest, 0.05)
            offset = offset + torch.clamp(clearance - (lowest + offset), min=0.0)
            if len(offset) > 1:
                rate = torch.as_tensor(
                    np.gradient(offset.detach().cpu().numpy(), 1.0 / fps),
                    dtype=offset.dtype,
                    device=offset.device,
                )
            else:
                rate = torch.zeros_like(offset)
            height_offsets.append(offset)
            velocity_offsets.append(rate)
        for motion, height, velocity in zip(self.motions, height_offsets, velocity_offsets, strict=True):
            motion["root_height_offset"] = height
            motion["root_height_velocity_offset"] = velocity
        self._reference_state_field_names = ("root_height_offset", "root_height_velocity_offset")

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
        # Features use world axes. Positions are relative to the root,
        # while velocities are absolute velocities expressed in world axes.
        marker_positions = []
        for body_id, local_position in self.marker_points:
            q = motion["body_quat_xyzw"][:, body_id]
            local = q.new_tensor(local_position).expand(q.shape[0], 3)
            cross = torch.cross(q[:, :3], local, dim=-1)
            rotated = local + 2.0 * (q[:, 3:] * cross + torch.cross(q[:, :3], cross, dim=-1))
            marker_positions.append((motion["body_pos_w"][:, body_id] + rotated).unsqueeze(1))
        # Marker points precede body origins, matching the environment contract.
        if marker_positions:
            key_body_pos = torch.cat([*marker_positions, key_body_pos], dim=1)
        body_pos_relative = (key_body_pos - root_pos).flatten(start_dim=1)
        features = torch.cat(
            [
                motion["body_lin_vel_w"][:, 0],
                motion["body_ang_vel_w"][:, 0],
                joint_pos,
                joint_vel,
                body_pos_relative,
            ],
            dim=-1,
        )
        if self.amp_coordinate_frame == "body":
            return amp_features_to_body_frame(
                features, motion["body_quat_xyzw"][:, 0], joint_pos.shape[-1] + joint_vel.shape[-1]
            )
        if self.amp_coordinate_frame == "heading":
            return amp_features_to_heading_frame(
                features, motion["body_quat_xyzw"][:, 0], joint_pos.shape[-1] + joint_vel.shape[-1]
            )
        return features

    def __len__(self):
        return len(self.frame_indices)
