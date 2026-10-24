import numpy as np
import torch

from rsl_rl.datasets import (
    BaseMotionDataset,
    BeyondMimicMotionDataset,
    MotionDataset,
    UniformReferenceStateSampler,
)


def test_beyondmimic_format_is_a_named_concrete_dataset(tmp_path):
    assert issubclass(BeyondMimicMotionDataset, BaseMotionDataset)
    assert MotionDataset is BeyondMimicMotionDataset

    motion = {
        "fps": np.array([50.0]),
        "joint_pos": np.zeros((2, 1), dtype=np.float32),
        "joint_vel": np.zeros((2, 1), dtype=np.float32),
        "body_pos_w": np.zeros((2, 1, 3), dtype=np.float32),
        "body_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (2, 1, 1)),
        "body_lin_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((2, 1, 3), dtype=np.float32),
    }
    np.savez(tmp_path / "motion.npz", **motion)

    dataset = BeyondMimicMotionDataset(str(tmp_path))

    assert isinstance(dataset.reference_state_sampler, UniformReferenceStateSampler)
    sampled = dataset.reference_state_sampler.sample(dataset, 4)
    assert sampled.bin_ids is None
    assert sampled.states["joint_pos"].shape == (4, 1)
