"""Tests for the native TD-MPC2 algorithm and planner."""

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import TDMPC2
from rsl_rl.env import VecEnv


class _Env(VecEnv):
    def __init__(self) -> None:
        self.num_envs = 2
        self.num_actions = 2
        self.device = "cpu"
        self.max_episode_length = 8
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.cfg = {}

    def get_observations(self):
        return _obs()

    def step(self, actions):
        return _obs(), torch.ones(2), torch.zeros(2, dtype=torch.bool), {}


def _obs() -> TensorDict:
    return TensorDict({"policy": torch.randn(2, 7), "privileged": torch.randn(2, 3)}, batch_size=[2])


def _cfg(learning_starts: int = 0) -> dict:
    return {
        "algorithm": {
            "class_name": "TDMPC2",
            "horizon": 2,
            "num_samples": 8,
            "num_elites": 2,
            "num_pi_trajs": 0,
            "iterations": 1,
            "min_std": 0.05,
            "max_std": 1.0,
            "temperature": 0.5,
            "mpc": True,
            "episodic": True,
            "tau": 0.1,
            "lr": 1e-3,
            "learning_starts": learning_starts,
        },
        "model": {
            "class_name": "TDMPC2WorldModel",
            "latent_dim": 16,
            "enc_dim": 16,
            "mlp_dim": 32,
            "num_enc_layers": 2,
            "num_q": 2,
            "num_bins": 11,
            "vmin": -5.0,
            "vmax": 5.0,
            "simnorm_dim": 4,
            "dropout": 0.0,
            "log_std_min": -5.0,
            "log_std_max": 1.0,
        },
        "memory": {"capacity": 64, "batch_size": 2},
        "obs_groups": {"actor": ["policy"]},
        "num_steps_per_env": 2,
    }


def test_construct_uses_configured_state_groups_and_planner_bounds() -> None:
    algorithm = TDMPC2.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    actions = algorithm.act(_obs())

    assert algorithm.model.obs_dim == 7
    assert actions.shape == (2, 2)
    assert torch.all(actions.abs() <= 1.0)
    assert algorithm.model.cfg["num_bins"] == 11
    assert algorithm.model.cfg["vmin"] == -5.0
    assert algorithm.model.cfg["vmax"] == 5.0


def test_random_action_warmup_uses_configured_learning_starts(monkeypatch) -> None:
    algorithm = TDMPC2.construct_algorithm(_obs(), _Env(), _cfg(learning_starts=4), "cpu")
    monkeypatch.setattr(algorithm, "_random_action", lambda count: torch.full((count, 2), 0.25))

    actions = algorithm.act(_obs())

    assert torch.equal(actions, torch.full((2, 2), 0.25))


def test_update_learns_from_completed_episode_sequences() -> None:
    algorithm = TDMPC2.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    for step in range(3):
        observation = torch.randn(7)
        next_observation = torch.randn(7)
        algorithm.replay_buffer.add_step(
            0,
            observation,
            torch.rand(2) * 2 - 1,
            torch.randn(()),
            next_observation,
            terminated=step == 2,
        )

    metrics = algorithm.update()

    assert set(metrics) >= {"total_loss", "consistency_loss", "reward_loss", "value_loss", "pi_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert algorithm.num_updates == 1


def test_timeout_closes_episode_without_setting_termination() -> None:
    algorithm = TDMPC2.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    algorithm.act(_obs())
    dones = torch.tensor([True, False])
    algorithm.process_env_step(
        _obs(), torch.ones(2), dones, {"time_outs": torch.tensor([True, False]), "terminated": torch.zeros(2)}
    )

    assert algorithm.replay_buffer.num_episodes == 1
    assert algorithm.replay_buffer._episodes[0][0]["terminated"].item() == 0.0


def test_final_observation_is_stored_for_autoreset_done() -> None:
    algorithm = TDMPC2.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    algorithm.act(_obs())
    final_observations = TensorDict({"policy": torch.full((2, 7), 99.0)}, batch_size=[2])
    algorithm.process_env_step(
        _obs(),
        torch.ones(2),
        torch.tensor([True, False]),
        {
            "final_observation": final_observations,
            "time_outs": torch.tensor([True, False]),
            "terminated": torch.zeros(2),
        },
    )

    assert torch.equal(algorithm.replay_buffer._episodes[0][0]["next_observation"], torch.full((7,), 99.0))


def test_policy_reset_and_checkpoint_round_trip(tmp_path) -> None:
    algorithm = TDMPC2.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    policy = algorithm.get_policy()
    assert policy(_obs()).shape == (2, 2)
    policy.reset(torch.tensor([True, False]))
    assert policy.algorithm._episode_start.tolist() == [True, False]
    policy(_obs())
    assert policy.algorithm._episode_start.tolist() == [False, False]
    algorithm.num_updates = 3
    state = algorithm.save()
    restored = TDMPC2.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    path = tmp_path / "tdmpc2.pt"
    torch.save(state, path)

    restored.load(torch.load(path, weights_only=False), load_cfg=None, strict=True)

    assert restored.num_updates == 3
    assert restored.replay_buffer.size == algorithm.replay_buffer.size
    assert torch.equal(
        next(restored.model.parameters()),
        next(algorithm.model.parameters()),
    )
    assert not restored.replay_buffer._current


def test_load_closes_partial_episode_at_fresh_environment_boundary() -> None:
    algorithm = TDMPC2.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    algorithm.replay_buffer.add_step(0, torch.zeros(7), torch.zeros(2), torch.ones(()), torch.ones(7), False)
    state = algorithm.save()
    restored = TDMPC2.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")

    restored.load(state)

    assert not restored.replay_buffer._current
    assert restored.replay_buffer.num_episodes == 1
