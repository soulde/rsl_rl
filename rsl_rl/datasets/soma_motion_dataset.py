"""SOMA-format AMP motion dataset."""

from __future__ import annotations

import numpy as np

from .base_motion_dataset import BaseMotionDataset


class SomaMotionDataset(BaseMotionDataset):
    """Load SOMA-retargeter NPZ files with explicit embedded contracts."""

    input_quaternion_format = "xyzw"

    _REQUIRED_FIELDS = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "body_names",
    )

    def _load_motion_file(self, motion_file, *, body_names, joint_names):
        data = np.load(motion_file, allow_pickle=True)
        missing = [name for name in self._REQUIRED_FIELDS if name not in data]
        if missing:
            raise ValueError(
                f"SOMA motion {motion_file} is missing required fields: {', '.join(missing)}"
            )
        quaternion_key = "body_quat_xyzw" if "body_quat_xyzw" in data else "body_quat_w"
        if quaternion_key not in data:
            raise ValueError(
                f"SOMA motion {motion_file} is missing quaternion field 'body_quat_xyzw'"
            )

        source_joint_names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
        source_body_names = [str(name) for name in np.asarray(data["body_names"]).tolist()]
        if len(source_joint_names) != len(set(source_joint_names)):
            raise ValueError(f"SOMA motion {motion_file} has duplicate joint_names")
        if len(source_body_names) != len(set(source_body_names)):
            raise ValueError(f"SOMA motion {motion_file} has duplicate body_names")
        if body_names is not None and (
            set(body_names) != set(source_body_names) or len(body_names) != len(source_body_names)
        ):
            raise ValueError(f"SOMA motion {motion_file} body_names do not match configured body_names")

        # Dataset type selects XYZW for SOMA. The explicit SOMA field is
        # preferred; body_quat_w is accepted as a format-selected legacy alias.
        tensor_data = {key: data[key] for key in data.files}
        tensor_data["body_quat_w"] = tensor_data[quaternion_key]
        motion = self._tensor_motion(tensor_data, self.device, self.input_quaternion_format)
        return motion
