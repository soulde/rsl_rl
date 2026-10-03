"""Simulator-independent RSI contract: WXYZ quaternions and world-frame root velocities."""

from typing import TypedDict

import torch


class RsiState(TypedDict):
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


class RsiJointLayout(TypedDict):
    joint_names: tuple[str, ...]
    joint_types: tuple[str, ...]
    joint_pos_offsets: tuple[int, ...]
    joint_vel_offsets: tuple[int, ...]
    quaternion_order: str


def reference_joint_layout(dataset) -> RsiJointLayout:
    """Resolve named variable-DOF exports or the legacy scalar-joint contract once."""
    names = getattr(dataset, "reference_joint_names", None) or getattr(dataset, "joint_names", None)
    if names is None and getattr(dataset, "motions", None):
        names = dataset.motions[0].get("joint_names")
    if names is None or len(names) == 0 or len(set(names)) != len(names):
        raise ValueError("RSI requires unique joint names in the dataset or configuration")
    names = tuple(names)
    types = tuple(getattr(dataset, "reference_joint_types", ("hinge",) * len(names)))
    widths = {"hinge": (1, 1), "slide": (1, 1), "ball": (4, 3), "free": (7, 6)}
    if len(types) != len(names) or any(kind not in widths for kind in types):
        raise ValueError("Invalid RSI joint types")
    offsets = []
    for axis, field in enumerate(("reference_pos_offsets", "reference_vel_offsets")):
        expected = [0]
        for kind in types:
            expected.append(expected[-1] + widths[kind][axis])
        supplied = tuple(getattr(dataset, field, expected))
        if supplied != tuple(expected):
            raise ValueError("RSI joint offsets do not match joint types")
        offsets.append(supplied)
    return dict(joint_names=names, joint_types=types, joint_pos_offsets=offsets[0],
                joint_vel_offsets=offsets[1], quaternion_order="wxyz")


def reference_state(sample: dict, batch_size: int, layout: RsiJointLayout) -> RsiState:
    """Normalize loader output without filtering frames or solving physical constraints."""
    state = {name: sample[name] for name in ("root_pos", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel")}
    state["root_quat"] = sample["root_quat_xyzw"][:, [3, 0, 1, 2]]
    dimensions = {"root_pos": 3, "root_quat": 4, "root_lin_vel": 3, "root_ang_vel": 3,
                  "joint_pos": layout["joint_pos_offsets"][-1], "joint_vel": layout["joint_vel_offsets"][-1]}
    for name, width in dimensions.items():
        if state[name].shape != (batch_size, width):
            raise ValueError(f"RSI {name} must have shape ({batch_size}, {width})")
    return state
