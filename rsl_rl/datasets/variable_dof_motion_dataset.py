# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named-joint NPZ records with independent position and velocity dimensions."""

from __future__ import annotations

import numpy as np
import torch

from .base_motion_dataset import BaseMotionDataset


class VariableDofMotionDataset(BaseMotionDataset):
    """Select AMP and RSI coordinates independently by their joint names.

    Joint quaternions remain WXYZ. Conversion to a simulator's convention belongs
    at the state-write boundary. Key points must already exist in body arrays.
    """

    reference_joint_pos_field = "rsi_joint_pos"
    reference_joint_vel_field = "rsi_joint_vel"
    _WIDTHS = {"hinge": (1, 1), "slide": (1, 1), "ball": (4, 3), "free": (7, 6)}

    def __init__(self, *args, rsi_joint_names=None, **kwargs):
        if rsi_joint_names is not None and not rsi_joint_names:
            raise ValueError("rsi_joint_names must be nonempty or None")
        self.rsi_joint_names = list(rsi_joint_names) if rsi_joint_names is not None else None
        super().__init__(*args, **kwargs)

    def _load_motion_file(self, motion_file, *, body_names, joint_names):
        with np.load(motion_file, allow_pickle=False) as record:
            data = {key: record[key] for key in record.files}
        required = (
            "schema_version",
            "quaternion_order",
            "joint_names",
            "joint_types",
            "joint_pos_offsets",
            "joint_vel_offsets",
            "joint_pos",
            "joint_vel",
            "body_names",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "fps",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{motion_file}: missing fields {missing}")
        if str(data["schema_version"].item()) != "named-joints-v1":
            raise ValueError("Unsupported variable-DOF schema_version")
        if str(data["quaternion_order"].item()) != "wxyz" or self.quaternion_format != "wxyz":
            raise ValueError("named-joints-v1 quaternion order must be wxyz")
        names = np.asarray(data["joint_names"]).tolist()
        types = np.asarray(data["joint_types"]).tolist()
        if not names or len(set(names)) != len(names):
            raise ValueError("joint_names must be nonempty and contain no duplicates")
        if len(types) != len(names) or any(kind not in self._WIDTHS for kind in types):
            raise ValueError("Unsupported or inconsistent joint types")
        offsets = []
        for field, axis in (("joint_pos_offsets", 0), ("joint_vel_offsets", 1)):
            values = np.asarray(data[field])
            widths = [self._WIDTHS[kind][axis] for kind in types]
            if (
                values.ndim != 1
                or values.shape != (len(names) + 1,)
                or not np.issubdtype(values.dtype, np.integer)
                or values[0] != 0
                or not np.array_equal(np.diff(values), widths)
            ):
                raise ValueError(f"Invalid {field}: offsets must match joint types")
            offsets.append(values)
        frames = data["joint_pos"].shape[0]
        for field, widths in (("joint_pos", offsets[0]), ("joint_vel", offsets[1])):
            if data[field].shape != (frames, int(widths[-1])) or frames < 1:
                raise ValueError(f"Invalid {field} frame count or coordinate width")
        bodies = np.asarray(data["body_names"]).tolist()
        if not bodies or len(set(bodies)) != len(bodies):
            raise ValueError("body_names must be nonempty and contain no duplicates")
        for field, width in (("body_pos_w", 3), ("body_quat_w", 4), ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
            if data[field].shape != (frames, len(bodies), width):
                raise ValueError(f"Invalid {field} frame count or body dimensions")
        for field in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            if not np.isfinite(data[field]).all():
                raise ValueError(f"{field} must be finite")
        if not np.isfinite(data["fps"]).all() or float(data["fps"]) <= 0:
            raise ValueError("fps must be positive and finite")
        if np.any(np.linalg.norm(data["body_quat_w"], axis=-1) < 1e-8):
            raise ValueError("Body quaternion must be nonzero")
        for index, kind in enumerate(types):
            if kind in {"ball", "free"}:
                start = int(offsets[0][index]) + (3 if kind == "free" else 0)
                if np.any(np.linalg.norm(data["joint_pos"][:, start : start + 4], axis=-1) < 1e-8):
                    raise ValueError("Joint quaternion must be nonzero")

        if not hasattr(self, "full_joint_names"):
            self.full_joint_names = names
            self.full_joint_types = types
            self.joint_pos_offsets = offsets[0].tolist()
            self.joint_vel_offsets = offsets[1].tolist()
        if set(names) != set(self.full_joint_names):
            raise ValueError("Full joint names differ across motion files")
        if [types[names.index(name)] for name in self.full_joint_names] != self.full_joint_types:
            raise ValueError("Full joint types differ across motion files")

        def coordinates(selected, field, axis):
            missing = [name for name in selected if name not in names]
            if missing or len(set(selected)) != len(selected) or not selected:
                raise ValueError(f"Invalid configured joint_names; missing or duplicate joints: {missing}")
            indices = [names.index(name) for name in selected]
            return np.concatenate([data[field][:, offsets[axis][i] : offsets[axis][i + 1]] for i in indices], axis=-1)

        selected = list(joint_names) if joint_names is not None else self.full_joint_names
        reference_names = self.rsi_joint_names if self.rsi_joint_names is not None else selected
        reference_types = [types[names.index(name)] for name in reference_names if name in names]
        # Validate names through the same decoder used for AMP, before building metadata.
        reference_pos = coordinates(reference_names, "joint_pos", 0)
        reference_vel = coordinates(reference_names, "joint_vel", 1)
        self.reference_joint_names = list(reference_names)
        self.reference_joint_types = reference_types
        self.reference_pos_offsets = np.cumsum([0] + [self._WIDTHS[kind][0] for kind in reference_types]).tolist()
        self.reference_vel_offsets = np.cumsum([0] + [self._WIDTHS[kind][1] for kind in reference_types]).tolist()
        feature_data = dict(data)
        feature_data["joint_names"] = selected
        feature_data["joint_pos"] = coordinates(selected, "joint_pos", 0)
        feature_data["joint_vel"] = coordinates(selected, "joint_vel", 1)
        motion = self._tensor_motion(feature_data, self.device, "wxyz")
        motion["rsi_joint_pos"] = torch.tensor(reference_pos, device=self.device, dtype=torch.float32)
        motion["rsi_joint_vel"] = torch.tensor(reference_vel, device=self.device, dtype=torch.float32)
        return motion

    @staticmethod
    def _normalize_joint_contract(motion, joint_names):
        # Named slices were already gathered while their offsets were available.
        pass

    def _with_joint_contract(self, sampled):
        sampled.update(
            joint_names=list(self.reference_joint_names),
            joint_types=list(self.reference_joint_types),
            joint_pos_offsets=list(self.reference_pos_offsets),
            joint_vel_offsets=list(self.reference_vel_offsets),
            quaternion_order="wxyz",
        )
        return sampled

    def sample_reference_frames(self, batch_size):
        return self._with_joint_contract(super().sample_reference_frames(batch_size))

    def sample_reference_states(self, batch_size):
        return self._with_joint_contract(super().sample_reference_states(batch_size))
