import torch
import numpy as np
import pytest

from rsl_rl.algorithms.amp import AmpReplayBuffer, compute_amp_reward, valid_amp_transition_mask
from rsl_rl.datasets import MotionDataset


def test_beyondmimic_motion_dataset_uses_shared_base():
    from rsl_rl.datasets.base_motion_dataset import BaseMotionDataset

    assert issubclass(MotionDataset, BaseMotionDataset)


def test_amp_reward_matches_lsgan_quadratic():
    predictions = torch.tensor([[-1.0], [0.0], [1.0], [3.0], [4.0]])

    rewards = compute_amp_reward(predictions)

    assert torch.allclose(
        rewards.squeeze(-1),
        torch.tensor([0.0, 0.75, 1.0, 0.0, 0.0]),
    )


def test_amp_replay_buffer_stores_full_transitions():
    buffer = AmpReplayBuffer(capacity=4, observation_dim=6, device="cpu")
    transitions = torch.arange(18, dtype=torch.float32).reshape(3, 6)

    buffer.add(transitions)

    assert buffer.size == 3
    assert buffer.data[:3].shape == (3, 6)
    assert torch.equal(buffer.data[:3], transitions)


def test_motion_dataset_uses_external_body_names_and_root_orientation(tmp_path):
    motion_path = tmp_path / "walk.npz"
    np.savez(
        motion_path,
        fps=np.array([50]),
        joint_pos=np.zeros((2, 2), dtype=np.float32),
        joint_vel=np.zeros((2, 2), dtype=np.float32),
        body_pos_w=np.zeros((2, 2, 3), dtype=np.float32),
        body_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 2, 1)),
        body_lin_vel_w=np.zeros((2, 2, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 2, 3), dtype=np.float32),
    )

    dataset = MotionDataset(
        str(tmp_path),
        amp_observation_dim=20,
        key_body_names=["foot"],
        body_names=["base", "foot"],
    )

    assert len(dataset.motions) == 1
    assert dataset.sample_amp_observations(3).shape == (3, 40)


def test_motion_dataset_filters_npz_basenames_with_regex(tmp_path, capsys):
    payload = {
        "fps": np.array([50]),
        "joint_pos": np.zeros((2, 2), dtype=np.float32),
        "joint_vel": np.zeros((2, 2), dtype=np.float32),
        "body_pos_w": np.zeros((2, 1, 3), dtype=np.float32),
        "body_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1, 1)),
        "body_lin_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
    }
    np.savez(tmp_path / "walking_fast01.npz", **payload)
    np.savez(tmp_path / "debug_motion.npz", **payload)

    dataset = MotionDataset(
        str(tmp_path),
        amp_observation_dim=17,
        motion_file_pattern=r"walking_.*\.npz",
    )

    assert len(dataset.motions) == 1
    assert "Selected 1 of 2 NPZ files" in capsys.readouterr().out


def test_motion_dataset_selects_one_file_per_motion_kind_with_regex(tmp_path, capsys):
    payload = {
        "fps": np.array([50]),
        "joint_pos": np.zeros((2, 2), dtype=np.float32),
        "joint_vel": np.zeros((2, 2), dtype=np.float32),
        "body_pos_w": np.zeros((2, 1, 3), dtype=np.float32),
        "body_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1, 1)),
        "body_lin_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
    }
    np.savez(tmp_path / "walking_fast01_stageii.npz", **payload)
    np.savez(tmp_path / "walking_fast07_stageii.npz", **payload)
    np.savez(tmp_path / "4_WalkInClockwiseCircle03_stageii.npz", **payload)
    np.savez(tmp_path / "7_WalkInClockwiseCircle10_stageii.npz", **payload)

    dataset = MotionDataset(
        str(tmp_path),
        amp_observation_dim=17,
        motion_file_pattern=r".*01_stageii\.npz",
    )

    assert len(dataset.motions) == 1
    assert "Selected 1 of 4 NPZ files" in capsys.readouterr().out


def test_motion_dataset_selects_explicit_file_list(tmp_path):
    payload = {
        "fps": np.array([50]),
        "joint_pos": np.zeros((2, 2), dtype=np.float32),
        "joint_vel": np.zeros((2, 2), dtype=np.float32),
        "body_pos_w": np.zeros((2, 1, 3), dtype=np.float32),
        "body_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1, 1)),
        "body_lin_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
    }
    np.savez(tmp_path / "run01_stageii.npz", **payload)
    np.savez(tmp_path / "turn_left01_stageii.npz", **payload)
    np.savez(tmp_path / "debug.npz", **payload)

    dataset = MotionDataset(
        str(tmp_path),
        amp_observation_dim=17,
        motion_files=["turn_left01_stageii.npz", "run01_stageii.npz"],
    )

    assert len(dataset.motions) == 2


def test_motion_dataset_rejects_unknown_explicit_files(tmp_path):
    np.savez(tmp_path / "run01_stageii.npz", fps=np.array([50]))

    with pytest.raises(ValueError, match="motion_files not found"):
        MotionDataset(str(tmp_path), motion_files=["missing.npz"])


def test_motion_dataset_rejects_pattern_and_list_together(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        MotionDataset(
            str(tmp_path),
            motion_file_pattern=r".*\.npz",
            motion_files=["run01_stageii.npz"],
        )


def test_motion_dataset_rejects_regex_without_matches(tmp_path):
    np.savez(tmp_path / "walking.npz", joint_pos=np.zeros((1, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="matched no NPZ files"):
        MotionDataset(str(tmp_path), motion_file_pattern=r"run_.*\.npz")


def test_motion_dataset_converts_key_bodies_to_root_frame_and_excludes_last_frame(tmp_path):
    root_quat = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
    body_pos = np.array(
        [
            [[3.0, 4.0, 1.0], [4.0, 4.0, 1.0]],
            [[3.0, 4.0, 1.0], [4.0, 4.0, 1.0]],
            [[3.0, 4.0, 1.0], [4.0, 4.0, 1.0]],
        ],
        dtype=np.float32,
    )
    np.savez(
        tmp_path / "walk.npz",
        fps=np.array([50]),
        joint_pos=np.zeros((3, 1), dtype=np.float32),
        joint_vel=np.zeros((3, 1), dtype=np.float32),
        body_pos_w=body_pos,
        body_quat_w=np.tile(root_quat, (3, 2, 1)),
        body_lin_vel_w=np.zeros((3, 2, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((3, 2, 3), dtype=np.float32),
    )

    dataset = MotionDataset(
        str(tmp_path),
        amp_observation_dim=17,
        key_body_names=["foot"],
        body_names=["base", "foot"],
    )

    assert len(dataset) == 2
    observation = dataset._get_observation(dataset.motions[0], 0)
    assert torch.allclose(observation[-3:], torch.tensor([0.0, -1.0, 0.0]), atol=1e-5)


def test_motion_dataset_preloads_vectorized_transitions(tmp_path):
    root_quat = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32)
    body_pos = np.random.default_rng(0).normal(size=(5, 2, 3)).astype(np.float32) + np.array([0.0, 0.0, 1.0])
    body_quat = np.tile(root_quat, (5, 2, 1)).astype(np.float32)
    np.savez(
        tmp_path / "walk.npz",
        fps=np.array([50]),
        joint_pos=np.random.default_rng(1).normal(size=(5, 2)).astype(np.float32),
        joint_vel=np.random.default_rng(2).normal(size=(5, 2)).astype(np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.random.default_rng(3).normal(size=(5, 2, 3)).astype(np.float32),
        body_ang_vel_w=np.random.default_rng(4).normal(size=(5, 2, 3)).astype(np.float32),
    )

    dataset = MotionDataset(
        str(tmp_path),
        amp_observation_dim=17,
        key_body_names=["foot"],
        body_names=["base", "foot"],
    )

    assert len(dataset.transitions) == len(dataset)
    for frame_idx in range(len(dataset)):
        expected = torch.cat(
            [dataset._get_observation(dataset.motions[0], frame_idx),
             dataset._get_observation(dataset.motions[0], frame_idx + 1)]
        )
        assert torch.allclose(dataset.transitions[frame_idx], expected, atol=1e-5)
    sample = dataset.sample_amp_observations(64)
    assert sample.shape == (64, 40)
    assert sample.device == dataset.transitions.device


def test_motion_dataset_rejects_joint_contract_mismatch(tmp_path):
    payload = {
        "fps": np.array([50]),
        "joint_pos": np.zeros((2, 2), dtype=np.float32),
        "joint_vel": np.zeros((2, 2), dtype=np.float32),
        "body_pos_w": np.zeros((2, 1, 3), dtype=np.float32),
        "body_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1, 1)),
        "body_lin_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
    }
    np.savez(tmp_path / "walk.npz", **payload)

    with pytest.raises(ValueError, match="joint contract"):
        MotionDataset(str(tmp_path), joint_names=["joint_a"])


def test_amp_transition_mask_excludes_terminated_environments():
    dones = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    assert torch.equal(valid_amp_transition_mask(dones), torch.tensor([True, False, True, False]))
