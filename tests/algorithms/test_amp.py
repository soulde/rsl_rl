import torch
import numpy as np
import pytest

from rsl_rl.algorithms.amp import (
    AMP,
    AmpReplayBuffer,
    _resolve_rsi_frame_sampler,
    compute_amp_reward,
    valid_amp_transition_mask,
)
from rsl_rl.datasets import MotionDataset


def test_beyondmimic_motion_dataset_uses_shared_base():
    from rsl_rl.datasets.base_motion_dataset import BaseMotionDataset

    assert issubclass(MotionDataset, BaseMotionDataset)


SOMA_TEST_BODY_NAMES = [
    "base_link", "waist_z_link", "waist_x_link", "body",
    "left_shoulder_y_link", "left_shoulder_x_link", "left_shoulder_z_link", "left_elbow_link",
    "left_wrist_z_link", "left_wrist_y_link", "left_wrist_x_link",
    "right_shoulder_y_link", "right_shoulder_x_link", "right_shoulder_z_link", "right_elbow_link",
    "right_wrist_z_link", "right_wrist_y_link", "right_wrist_x_link", "neck_link", "head_link",
    "left_hip_y_link", "left_hip_x_link", "left_hip_z_link", "left_knee_link", "left_ankle_y_link",
    "left_ankle_x_link", "left_toe_link", "right_hip_y_link", "right_hip_x_link", "right_hip_z_link",
    "right_knee_link", "right_ankle_y_link", "right_ankle_x_link", "right_toe_link",
]


def _write_soma_npz(path, *, joint_names=None, body_names=None, frames=3, zero_pose=False):
    joint_names = joint_names or ["joint_a", "joint_b"]
    body_names = body_names or SOMA_TEST_BODY_NAMES
    joint_pos = (
        np.zeros((frames, len(joint_names)), dtype=np.float32)
        if zero_pose
        else np.arange(frames * len(joint_names), dtype=np.float32).reshape(frames, len(joint_names))
    )
    body_pos = np.zeros((frames, len(body_names), 3), dtype=np.float32)
    body_pos[:, body_names.index("left_ankle_x_link"), 0] = np.arange(frames, dtype=np.float32)
    np.savez(
        path,
        fps=np.float32(50.0),
        joint_pos=joint_pos,
        joint_vel=np.zeros_like(joint_pos) if zero_pose else joint_pos + 100.0,
        body_pos_w=body_pos,
        body_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, len(body_names), 1)),
        body_lin_vel_w=np.zeros_like(body_pos),
        body_ang_vel_w=np.zeros_like(body_pos),
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(body_names),
    )


def test_soma_motion_dataset_requires_embedded_name_contract(tmp_path):
    from rsl_rl.datasets.soma_motion_dataset import SomaMotionDataset

    _write_soma_npz(tmp_path / "walk.npz")
    with np.load(tmp_path / "walk.npz") as data:
        payload = {key: data[key] for key in data.files if key not in {"joint_names", "body_names"}}
    np.savez(tmp_path / "missing_names.npz", **payload)

    with pytest.raises(ValueError, match="SOMA.*joint_names.*body_names"):
        SomaMotionDataset(
            str(tmp_path), amp_observation_dim=20, body_names=SOMA_TEST_BODY_NAMES, joint_names=["joint_a", "joint_b"]
        )


def test_soma_motion_dataset_accepts_configured_body_order_and_keeps_adjacent_frames(tmp_path):
    from rsl_rl.datasets.soma_motion_dataset import SomaMotionDataset

    _write_soma_npz(tmp_path / "walk.npz")
    dataset = SomaMotionDataset(
        str(tmp_path),
        amp_observation_dim=20,
        key_body_names=["left_ankle_x_link"],
        body_names=SOMA_TEST_BODY_NAMES,
        joint_names=["joint_a", "joint_b"],
    )

    assert len(dataset) == 2
    assert dataset.motions[0]["fps"] == 50.0
    assert torch.equal(dataset.motions[0]["joint_pos"][:, 0], torch.tensor([0.0, 2.0, 4.0]))
    assert torch.equal(dataset.motions[0]["joint_vel"][:, 1], torch.tensor([101.0, 103.0, 105.0]))
    assert torch.equal(dataset.transitions[:, 13].sort().values, torch.tensor([0.0, 2.0]))


def test_soma_motion_dataset_supports_single_frame_rsi_only_source(tmp_path):
    from rsl_rl.datasets.soma_motion_dataset import SomaMotionDataset

    expert_dir = tmp_path / "expert"
    rsi_dir = tmp_path / "rsi"
    expert_dir.mkdir()
    rsi_dir.mkdir()
    _write_soma_npz(expert_dir / "walk.npz", frames=3)
    _write_soma_npz(rsi_dir / "zero_pose.npz", frames=1, zero_pose=True)
    expert_dataset = SomaMotionDataset(
        str(expert_dir),
        amp_observation_dim=20,
        key_body_names=["left_ankle_x_link"],
        body_names=SOMA_TEST_BODY_NAMES,
        joint_names=["joint_a", "joint_b"],
    )
    dataset = SomaMotionDataset(
        str(rsi_dir),
        amp_observation_dim=20,
        key_body_names=["left_ankle_x_link"],
        body_names=SOMA_TEST_BODY_NAMES,
        joint_names=["joint_a", "joint_b"],
        require_transitions=False,
    )

    sampled = dataset.sample_reference_states(5)

    assert dataset.transitions.shape == (0, 40)
    assert sampled["joint_pos"].shape == (5, 2)
    assert torch.equal(sampled["joint_pos"], torch.zeros(5, 2))
    assert torch.equal(sampled["joint_vel"], torch.zeros(5, 2))
    assert sampled["root_pos"].shape == (5, 3)
    assert sampled["root_quat_xyzw"].shape == (5, 4)
    assert torch.equal(sampled["root_quat_xyzw"], torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(5, 4))

    with pytest.raises(RuntimeError, match="no expert transitions"):
        dataset.sample_amp_observations(1)

    assert len(expert_dataset) == 2
    assert expert_dataset.sample_amp_observations(4).shape == (4, 40)


def test_motion_dataset_set_validate_filters_transitions_and_attaches_aligned_reset_fields(tmp_path):
    from rsl_rl.datasets.soma_motion_dataset import SomaMotionDataset

    _write_soma_npz(tmp_path / "walk.npz", frames=3)
    dataset = SomaMotionDataset(
        str(tmp_path),
        amp_observation_dim=20,
        key_body_names=["left_ankle_x_link"],
        body_names=SOMA_TEST_BODY_NAMES,
        joint_names=["joint_a", "joint_b"],
    )
    motor_positions = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    motor_velocities = motor_positions + 10.0

    dataset.set_validate(
        torch.tensor([False, True]),
        reference_state_fields={
            "motor_joint_pos": [motor_positions],
            "motor_joint_vel": [motor_velocities],
        },
    )

    assert len(dataset) == 1
    assert dataset.transitions.shape == (1, 40)
    sampled = dataset.sample_reference_frames(4)
    assert torch.equal(sampled["joint_pos"], dataset.motions[0]["joint_pos"][1:2].expand(4, -1))
    assert torch.equal(sampled["motor_joint_pos"], motor_positions[1:2].expand(4, -1))
    assert torch.equal(sampled["motor_joint_vel"], motor_velocities[1:2].expand(4, -1))


def test_amp_rsi_sampler_prefers_separate_dataset_and_keeps_legacy_fallback():
    class Dataset:
        def sample_reference_frames(self, batch_size):
            return {"source": "expert", "batch_size": batch_size}

        def sample_reference_states(self, batch_size):
            return {"source": "rsi", "batch_size": batch_size}

    expert_dataset = Dataset()
    rsi_dataset = Dataset()

    assert _resolve_rsi_frame_sampler(expert_dataset, rsi_dataset)(3) == {"source": "rsi", "batch_size": 3}
    assert _resolve_rsi_frame_sampler(expert_dataset)(2) == {"source": "expert", "batch_size": 2}


def test_amp_construction_keeps_expert_and_single_frame_rsi_sources_separate(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import rsl_rl.algorithms.amp as amp_module

    expert_dir = tmp_path / "expert"
    rsi_dir = tmp_path / "rsi"
    expert_dir.mkdir()
    rsi_dir.mkdir()
    _write_soma_npz(expert_dir / "walk.npz", frames=3)
    _write_soma_npz(rsi_dir / "zero_pose.npz", frames=1, zero_pose=True)

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, device):
            return self

    class Algorithm:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class Storage:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        amp_module,
        "resolve_callable",
        lambda name: {"fake_amp": Algorithm, "fake_model": Model}[name],
    )
    monkeypatch.setattr(amp_module, "resolve_obs_groups", lambda obs, groups, defaults: groups)
    monkeypatch.setattr(amp_module, "resolve_rnd_config", lambda cfg, obs, groups, env: cfg)
    monkeypatch.setattr(amp_module, "resolve_symmetry_config", lambda cfg, env: cfg)
    monkeypatch.setattr(amp_module, "RolloutStorage", Storage)

    obs = {"amp": torch.zeros(1, 20)}
    cfg = {
        "algorithm": {
            "class_name": "fake_amp",
            "motion_dir": str(expert_dir),
            "rsi_motion_dir": str(rsi_dir),
            "motion_dataset_format": "soma",
            "joint_names": ["joint_a", "joint_b"],
            "body_names": SOMA_TEST_BODY_NAMES,
            "key_body_names": ["left_ankle_x_link"],
            "observation_profile": "default",
            "reference_state_initialization": True,
        },
        "actor": {"class_name": "fake_model"},
        "critic": {"class_name": "fake_model"},
        "obs_groups": {"actor": [], "critic": [], "discriminator": ["amp"]},
        "num_steps_per_env": 1,
        "multi_gpu": {},
    }
    env = SimpleNamespace(
        num_actions=2,
        num_envs=1,
        unwrapped=SimpleNamespace(cfg=SimpleNamespace(sim=SimpleNamespace(dt=0.02), decimation=1)),
    )

    algorithm = AMP.construct_algorithm(obs, env, cfg, "cpu")

    assert algorithm.kwargs["collect_reference_motions"](4).shape == (4, 40)
    sampled_rsi = algorithm.kwargs["reference_frame_sampler"](5)
    assert torch.equal(sampled_rsi["joint_pos"], torch.zeros(5, 2))
    assert algorithm.kwargs["reference_state_initialization"] is True


def test_motion_dataset_normalizes_body_order_and_quaternion_to_isaacsim(tmp_path):
    from rsl_rl.datasets.soma_motion_dataset import SomaMotionDataset

    source_body_names = list(reversed(SOMA_TEST_BODY_NAMES))
    body_pos = np.zeros((3, len(source_body_names), 3), dtype=np.float32)
    body_pos[:, source_body_names.index("left_ankle_x_link"), 0] = 2.0
    _write_soma_npz(tmp_path / "walk.npz", body_names=source_body_names)
    with np.load(tmp_path / "walk.npz") as data:
        payload = {key: data[key] for key in data.files}
    payload["body_pos_w"] = body_pos
    # MuJoCo/GMR WXYZ (90 degrees around Z) must expose Isaac Sim XYZW.
    payload["body_quat_w"] = np.tile(
        np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], dtype=np.float32),
        (3, len(source_body_names), 1),
    )
    np.savez(tmp_path / "walk.npz", **payload)

    dataset = SomaMotionDataset(
        str(tmp_path),
        amp_observation_dim=20,
        key_body_names=["left_ankle_x_link"],
        body_names=SOMA_TEST_BODY_NAMES,
        joint_names=["joint_a", "joint_b"],
    )

    motion = dataset.motions[0]
    assert motion["body_names"] == SOMA_TEST_BODY_NAMES
    assert torch.all(motion["body_pos_w"][:, SOMA_TEST_BODY_NAMES.index("left_ankle_x_link"), 0] == 2.0)
    assert torch.allclose(
        motion["body_quat_xyzw"][0, 0],
        torch.tensor([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=torch.float32),
        atol=1e-6,
    )


def test_motion_dataset_accepts_xyzw_source_quaternions(tmp_path):
    motion_path = tmp_path / "xyzw.npz"
    np.savez(
        motion_path,
        fps=np.array([50]),
        joint_pos=np.zeros((2, 1), dtype=np.float32),
        joint_vel=np.zeros((2, 1), dtype=np.float32),
        body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_quat_w=np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1, 1)),
        body_lin_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((2, 1, 3), dtype=np.float32),
    )
    dataset = MotionDataset(str(tmp_path), amp_observation_dim=15, quaternion_format="xyzw")
    assert torch.allclose(dataset.motions[0]["body_quat_xyzw"][0, 0], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_soma_motion_dataset_rejects_body_name_mismatch(tmp_path):
    from rsl_rl.datasets.soma_motion_dataset import SomaMotionDataset

    mismatched = list(SOMA_TEST_BODY_NAMES)
    mismatched[0] = "not_a_robot_body"
    _write_soma_npz(tmp_path / "walk.npz", body_names=mismatched)

    with pytest.raises(ValueError, match="body_names"):
        SomaMotionDataset(
            str(tmp_path), amp_observation_dim=20, body_names=SOMA_TEST_BODY_NAMES, joint_names=["joint_a", "joint_b"]
        )


def test_amp_motion_dataset_format_selects_soma_and_defaults_to_beyondmimic():
    from rsl_rl.algorithms.amp import resolve_motion_dataset_class
    from rsl_rl.datasets import SomaMotionDataset

    assert resolve_motion_dataset_class({}) is MotionDataset
    assert resolve_motion_dataset_class({"motion_dataset_format": "soma"}) is SomaMotionDataset


def test_amp_motion_dataset_format_rejects_unknown_format():
    from rsl_rl.algorithms.amp import resolve_motion_dataset_class

    with pytest.raises(ValueError, match="Unknown AMP motion dataset format"):
        resolve_motion_dataset_class({"motion_dataset_format": "unknown"})


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

    reference = dataset.sample_reference_frames(64)
    assert reference["amp_obs"].shape == (64, 20)
    assert reference["amp_next_obs"].shape == (64, 20)
    assert reference["joint_pos"].shape == (64, 2)
    assert reference["root_pos"].shape == (64, 3)

    empty_reference = dataset.sample_reference_frames(0)
    assert empty_reference["amp_obs"].shape == (0, 20)
    assert empty_reference["amp_next_obs"].shape == (0, 20)
    assert empty_reference["joint_pos"].shape == (0, 2)


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
