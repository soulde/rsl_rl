from __future__ import annotations

import torch
from pathlib import Path
from tensordict import TensorDict
from typing import ClassVar

from rsl_rl.runners import DaggerOnPolicyRunner


class DummyEnv:
    """Small CPU environment used to exercise the runner lifecycle."""

    num_envs = 3
    num_actions = 2
    device = "cpu"
    cfg: ClassVar[dict] = {}

    def get_observations(self) -> TensorDict:
        """Return student and teacher observation groups."""
        return TensorDict(
            {
                "policy": torch.randn(3, 4),
                "depth": torch.randn(3, 1, 8, 8),
                "privileged": torch.randn(3, 5),
            },
            batch_size=[3],
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Advance one simulated step."""
        return self.get_observations(), torch.zeros(3), torch.zeros(3), {}


def make_cfg(teacher_path: Path | None = None) -> dict:
    """Create a compact model and DAgger configuration for tests."""
    teacher = {"class_name": "MLPModel", "hidden_dims": [8], "activation": "elu"}
    if teacher_path is not None:
        teacher["checkpoint"] = str(teacher_path)
    return {
        "num_steps_per_env": 2,
        "save_interval": 100,
        "obs_groups": {"student": ["policy", "depth"], "teacher": ["privileged"]},
        "student": {
            "class_name": "CNNModel", "hidden_dims": [8], "activation": "elu",
            "stochastic": {}, "init_noise_std": {}, "noise_std_type": "scalar", "state_dependent_std": False,
            "cnn_cfg": {"output_channels": [4], "kernel_size": 3, "stride": 2},
        },
        "teacher": {
            **teacher,
            "stochastic": {}, "init_noise_std": {}, "noise_std_type": "scalar", "state_dependent_std": False,
        },
        "algorithm": {
            "collection_steps": 2, "updates_per_round": 1, "batch_size": 3,
            "storage_capacity": 32, "learning_rate": 1e-3,
            "beta": {"start": 1.0, "end": 0.0, "decay": 0.9},
        },
    }


def test_runner_constructs_cnn_student_and_frozen_deterministic_teacher() -> None:
    """Construct a CNN student and freeze the teacher model."""
    runner = DaggerOnPolicyRunner(DummyEnv(), make_cfg())
    assert runner.student.__class__.__name__ == "CNNModel"
    assert not runner.teacher.training
    assert all(not param.requires_grad for param in runner.teacher.parameters())
    assert runner.student.training


def test_runner_learn_updates_student_but_never_teacher() -> None:
    """Supervised learning changes student weights while teacher weights stay fixed."""
    runner = DaggerOnPolicyRunner(DummyEnv(), make_cfg())
    student_before = [p.detach().clone() for p in runner.student.parameters()]
    teacher_before = [p.detach().clone() for p in runner.teacher.parameters()]
    runner.learn(1)
    assert any(not torch.equal(a, b) for a, b in zip(student_before, runner.student.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(teacher_before, runner.teacher.parameters()))
    assert runner.current_learning_iteration == 1


def test_runner_can_clear_rollout_storage_after_each_round() -> None:
    """PHP pure DAgger consumes and clears the current rollout after its update."""
    cfg = make_cfg()
    cfg["algorithm"]["clear_storage_after_update"] = True
    runner = DaggerOnPolicyRunner(DummyEnv(), cfg)

    runner.learn(1)

    assert len(runner.storage) == 0


def test_runner_saves_and_loads_student_and_optimizer_state(tmp_path: Path) -> None:
    """Checkpoint round-trip restores parameters and iteration."""
    runner = DaggerOnPolicyRunner(DummyEnv(), make_cfg())
    runner.learn(1)
    path = tmp_path / "student.pt"
    runner.save(str(path))
    restored = DaggerOnPolicyRunner(DummyEnv(), make_cfg())
    restored.load(str(path))
    for expected, actual in zip(runner.student.parameters(), restored.student.parameters()):
        assert torch.equal(expected, actual)
    for expected, actual in zip(runner.teacher.parameters(), restored.teacher.parameters()):
        assert torch.equal(expected, actual)
    assert restored.current_learning_iteration == 1


def test_runner_loads_ppo_actor_checkpoint_as_teacher(tmp_path: Path) -> None:
    """Load the actor_state_dict convention used by PPO checkpoints."""
    source = DaggerOnPolicyRunner(DummyEnv(), make_cfg())
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"actor_state_dict": source.teacher.state_dict()}, checkpoint)
    runner = DaggerOnPolicyRunner(DummyEnv(), make_cfg(checkpoint))
    for expected, actual in zip(source.teacher.parameters(), runner.teacher.parameters()):
        assert torch.equal(expected, actual)


def test_runner_loads_model_state_dict_teacher_checkpoint(tmp_path: Path) -> None:
    """Load the alternate model_state_dict checkpoint convention."""
    source = DaggerOnPolicyRunner(DummyEnv(), make_cfg())
    checkpoint = tmp_path / "teacher_model_state.pt"
    torch.save({"model_state_dict": source.teacher.state_dict()}, checkpoint)
    runner = DaggerOnPolicyRunner(DummyEnv(), make_cfg(checkpoint))
    for expected, actual in zip(source.teacher.parameters(), runner.teacher.parameters()):
        assert torch.equal(expected, actual)


def test_runner_inference_policy_accepts_grouped_observations() -> None:
    """Inference consumes the selected TensorDict groups and returns actions."""
    runner = DaggerOnPolicyRunner(DummyEnv(), make_cfg())
    policy = runner.get_inference_policy()
    obs = runner.env.get_observations().select("policy", "depth")
    assert policy(obs).shape == (3, 2)


def test_runner_exports_student_to_torchscript(tmp_path: Path) -> None:
    """The standard play launcher can export a trained depth policy."""
    runner = DaggerOnPolicyRunner(DummyEnv(), make_cfg())
    runner.export_policy_to_jit(str(tmp_path))
    assert (tmp_path / "policy.pt").is_file()
