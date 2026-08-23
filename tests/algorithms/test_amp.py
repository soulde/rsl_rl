import torch
import numpy as np
import pytest

from rsl_rl.algorithms.amp import AmpReplayBuffer, compute_amp_reward
from rsl_rl.datasets import MotionDataset


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


def test_motion_dataset_rejects_regex_without_matches(tmp_path):
    np.savez(tmp_path / "walking.npz", joint_pos=np.zeros((1, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="matched no NPZ files"):
        MotionDataset(str(tmp_path), motion_file_pattern=r"run_.*\.npz")
