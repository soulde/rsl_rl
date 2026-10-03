# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch

from rsl_rl.algorithms.amp import resolve_motion_dataset_class


def write_motion(path, **changes):
    data = dict(
        schema_version=np.array("named-joints-v1"),
        quaternion_order=np.array("wxyz"),
        fps=np.float32(50),
        joint_names=np.array(["hip", "rod", "ankle"]),
        joint_types=np.array(["hinge", "ball", "hinge"]),
        joint_pos_offsets=np.array([0, 1, 5, 6]),
        joint_vel_offsets=np.array([0, 1, 4, 5]),
        joint_pos=np.array([[1, 1, 0, 0, 0, 2], [3, 1, 0, 0, 0, 4], [5, 1, 0, 0, 0, 6]], dtype=np.float32),
        joint_vel=np.array([[11, 12, 13, 14, 15], [21, 22, 23, 24, 25], [31, 32, 33, 34, 35]], dtype=np.float32),
        body_names=np.array(["pelvis", "ankle_point"]),
        body_pos_w=np.tile(np.array([[10, 0, 1], [10, 0, 0]], dtype=np.float32), (3, 1, 1)),
        body_quat_w=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (3, 2, 1)),
        body_lin_vel_w=np.zeros((3, 2, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((3, 2, 3), dtype=np.float32),
    )
    data.update(changes)
    np.savez(path, **data)


def load(path, **kwargs):
    cls = resolve_motion_dataset_class({"motion_dataset_format": "variable_dof"})
    kwargs.setdefault("rsi_joint_names", ["hip", "rod", "ankle"])
    return cls(
        str(path),
        joint_names=["ankle", "hip"],
        key_body_names=["ankle_point"],
        observation_profile="chocolate",
        amp_observation_dim=13,
        **kwargs,
    )


def test_selects_feature_slices_and_samples_complete_joint_state(tmp_path):
    write_motion(tmp_path / "motion.npz")
    dataset = load(tmp_path)
    torch.testing.assert_close(dataset.transitions[0, :2], torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(dataset.transitions[0, 2:5], torch.tensor([0.0, 0.0, -1.0]))
    torch.testing.assert_close(dataset.transitions[0, 11:13], torch.tensor([15.0, 11.0]))
    dataset.set_validate(torch.tensor([True, False]), {"extra": [torch.arange(3.0)[:, None]]})
    sample = dataset.sample_reference_frames(4)
    assert sample["joint_names"] == ["hip", "rod", "ankle"]
    assert sample["joint_types"] == ["hinge", "ball", "hinge"]
    assert sample["joint_pos_offsets"] == [0, 1, 5, 6]
    assert sample["joint_vel_offsets"] == [0, 1, 4, 5]
    assert sample["quaternion_order"] == "wxyz"
    torch.testing.assert_close(sample["joint_pos"], torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 2.0]]).repeat(4, 1))
    assert sample["extra"].shape == (4, 1)
    assert dataset.sample_reference_frames(0)["joint_pos"].shape == (0, 6)
    assert dataset.sample_amp_observations(4).shape == (4, 26)


def test_single_frame_rsi_uses_same_full_state_decoder(tmp_path):
    write_motion(tmp_path / "motion.npz")
    with np.load(tmp_path / "motion.npz") as n:
        data = {k: n[k][:1] if n[k].ndim > 1 else n[k] for k in n.files}
    np.savez(tmp_path / "motion.npz", **data)
    dataset = load(tmp_path, require_transitions=False)
    assert dataset.sample_reference_states(5)["joint_vel"].shape == (5, 5)
    assert dataset.sample_reference_states(0)["joint_pos"].shape == (0, 6)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"joint_pos_offsets": np.array([0, 1, 4, 6])}, "offset"),
        ({"joint_names": np.array(["hip", "hip", "ankle"])}, "duplicate"),
        ({"joint_types": np.array(["hinge", "unknown", "hinge"])}, "type"),
        ({"joint_vel": np.zeros((2, 5))}, "frame"),
        ({"quaternion_order": np.array("xyzw")}, "quaternion"),
        ({"fps": np.float32(0)}, "fps"),
        ({"joint_pos": np.full((3, 6), np.nan)}, "finite"),
    ],
)
def test_rejects_malformed_contract(tmp_path, changes, match):
    write_motion(tmp_path / "motion.npz", **changes)
    with pytest.raises(ValueError, match=match):
        load(tmp_path)


def test_missing_configured_point_is_not_generated(tmp_path):
    write_motion(tmp_path / "motion.npz", body_names=np.array(["pelvis", "not_ankle"]))
    with pytest.raises(ValueError, match="key bodies"):
        load(tmp_path)


def test_ball_joint_features_have_distinct_position_velocity_widths(tmp_path):
    write_motion(tmp_path / "motion.npz")
    cls = resolve_motion_dataset_class({"motion_dataset_format": "variable_dof"})
    dataset = cls(
        str(tmp_path),
        joint_names=["rod"],
        key_body_names=["ankle_point"],
        observation_profile="chocolate",
        amp_observation_dim=16,
    )
    assert dataset.sample_amp_observations(2).shape == (2, 32)


def test_joint_blocks_reorder_consistently_between_files(tmp_path):
    write_motion(tmp_path / "a.npz")
    write_motion(
        tmp_path / "b.npz",
        joint_names=np.array(["ankle", "hip", "rod"]),
        joint_types=np.array(["hinge", "hinge", "ball"]),
        joint_pos_offsets=np.array([0, 1, 2, 6]),
        joint_vel_offsets=np.array([0, 1, 2, 5]),
        joint_pos=np.array([[2, 1, 1, 0, 0, 0], [4, 3, 1, 0, 0, 0], [6, 5, 1, 0, 0, 0]], dtype=np.float32),
        joint_vel=np.array([[15, 11, 12, 13, 14], [25, 21, 22, 23, 24], [35, 31, 32, 33, 34]], dtype=np.float32),
    )
    dataset = load(tmp_path)
    dataset.set_validate(torch.tensor([False, False, True, False]))
    sampled = dataset.sample_reference_frames(1)
    torch.testing.assert_close(sampled["joint_pos"], torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 2.0]]))
    torch.testing.assert_close(sampled["joint_vel"], torch.tensor([[11.0, 12.0, 13.0, 14.0, 15.0]]))
    torch.testing.assert_close(sampled["amp_obs"][:, :2], torch.tensor([[2.0, 1.0]]))


def test_missing_selected_joint_is_rejected(tmp_path):
    write_motion(tmp_path / "motion.npz", joint_names=np.array(["hip", "rod", "other"]))
    with pytest.raises(ValueError, match="missing"):
        load(tmp_path)


def test_zero_ball_quaternion_is_rejected(tmp_path):
    write_motion(tmp_path / "motion.npz", joint_pos=np.zeros((3, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="quaternion"):
        load(tmp_path)


@pytest.mark.parametrize("selection", [None, ["rod", "hip"]])
def test_rsi_selection_is_independent_and_none_defaults_to_amp(tmp_path, selection):
    write_motion(tmp_path / "motion.npz")
    dataset = load(tmp_path, rsi_joint_names=selection)
    dataset.set_validate(torch.tensor([True, False]))
    sample = dataset.sample_reference_frames(1)
    torch.testing.assert_close(sample["amp_obs"][:, :2], torch.tensor([[2.0, 1.0]]))
    if selection:
        assert sample["joint_names"] == ["rod", "hip"]
        assert sample["joint_pos_offsets"] == [0, 4, 5]
        assert sample["joint_vel_offsets"] == [0, 3, 4]
        torch.testing.assert_close(sample["joint_pos"], torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]]))
        torch.testing.assert_close(sample["joint_vel"], torch.tensor([[12.0, 13.0, 14.0, 11.0]]))
    else:
        assert sample["joint_names"] == ["ankle", "hip"]
        assert sample["joint_pos_offsets"] == [0, 1, 2]
        torch.testing.assert_close(sample["joint_pos"], torch.tensor([[2.0, 1.0]]))
    assert dataset.sample_reference_states(3)["joint_pos"].shape[-1] == sample["joint_pos"].shape[-1]


def test_empty_rsi_selection_is_rejected(tmp_path):
    write_motion(tmp_path / "motion.npz")
    with pytest.raises(ValueError, match="nonempty"):
        load(tmp_path, rsi_joint_names=[])
