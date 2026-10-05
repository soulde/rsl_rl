"""Tests for the native FastSAC algorithm."""

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import FastSAC
from rsl_rl.env import VecEnv
from rsl_rl.models import DistributionalCritic, FastSACActor
from rsl_rl.storage import OffPolicyBatch


class _Env(VecEnv):
    num_envs = 3
    num_actions = 2
    device = "cpu"
    max_episode_length = 20
    episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    cfg = {}

    def get_observations(self):
        return _obs(self.num_envs)

    def step(self, actions):
        return _obs(self.num_envs), torch.ones(self.num_envs), torch.zeros(self.num_envs), {}


def _obs(num_envs: int = 3) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(num_envs, 3),
            "privileged": torch.randn(num_envs, 5),
        },
        batch_size=[num_envs],
    )


def _cfg() -> dict:
    return {
        "algorithm": {
            "class_name": "FastSAC",
            "gamma": 0.99,
            "tau": 0.01,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,
            "init_alpha": 0.2,
            "target_entropy_ratio": 1.0,
            "auto_entropy": True,
        },
        "actor": {"hidden_dims": [16, 16], "log_std_min": -5.0, "log_std_max": 1.0},
        "critic": {
            "hidden_dims": [16, 16],
            "num_atoms": 11,
            "v_min": -10.0,
            "v_max": 10.0,
            "num_q_networks": 2,
        },
        "memory": {"capacity": 64, "batch_size": 4},
        "obs_groups": {"actor": ["policy"], "critic": ["privileged"]},
        "num_steps_per_env": 2,
    }


def test_fast_sac_models_have_expected_shapes_and_project_probability_mass() -> None:
    actor = FastSACActor(obs_dim=3, action_dim=2, hidden_dims=(16, 16))
    critic = DistributionalCritic(obs_dim=5, action_dim=2, hidden_dims=(16, 16), num_atoms=11, v_min=-10, v_max=10)
    observations = torch.randn(4, 3)
    actions, log_prob = actor.sample(observations)
    logits = critic(torch.randn(4, 5), actions)
    projected = critic.project(torch.softmax(logits[0], dim=-1), torch.ones(4, 1), torch.ones(4, 1), 0.99)

    assert actions.shape == (4, 2)
    assert torch.all(actions.abs() <= 1.0)
    assert log_prob.shape == (4, 1)
    assert logits.shape == (2, 4, 11)
    assert torch.allclose(projected.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_construct_algorithm_uses_actor_and_privileged_critic_groups() -> None:
    algorithm = FastSAC.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")

    assert algorithm.actor.obs_dim == 3
    assert algorithm.critic.obs_dim == 5
    assert algorithm.act(_obs()).shape == (3, 2)


def test_fast_sac_warmup_uses_uniform_bounded_actions(monkeypatch) -> None:
    algorithm = FastSAC.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    monkeypatch.setattr(algorithm.actor, "sample", lambda *_args: (_ for _ in ()).throw(AssertionError("actor used")))

    actions = algorithm.act(_obs())

    assert actions.shape == (3, 2)
    assert torch.all((actions >= -1.0) & (actions <= 1.0))


def test_fast_sac_updates_from_replay_and_bootstraps_timeouts() -> None:
    algorithm = FastSAC.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    for step in range(2):
        algorithm.act(_obs())
        dones = torch.tensor([True, False, False]) if step == 0 else torch.zeros(3, dtype=torch.bool)
        timeouts = dones.clone()
        algorithm.process_env_step(_obs(), torch.ones(3), dones, {"time_outs": timeouts})

    assert algorithm.replay_buffer.size == 6
    assert torch.equal(algorithm.replay_buffer.dones[:3, 0], torch.zeros(3))
    losses = algorithm.update()
    assert set(losses) >= {"actor_loss", "critic_loss", "alpha"}
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_fast_sac_replay_uses_final_observation_for_autoreset_done() -> None:
    algorithm = FastSAC.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    algorithm.act(_obs())
    final_observations = TensorDict(
        {"policy": torch.full((3, 3), 20.0), "privileged": torch.full((3, 5), 30.0)}, batch_size=[3]
    )
    algorithm.process_env_step(
        _obs(), torch.ones(3), torch.tensor([True, False, False]), {"final_observation": final_observations}
    )

    assert torch.equal(algorithm.replay_buffer.next_observations[0], torch.tensor([20.0] * 3 + [30.0] * 5))


def test_entropy_shift_is_discounted_and_masked_at_true_terminals(monkeypatch) -> None:
    algorithm = FastSAC.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    batch = OffPolicyBatch(
        observations=torch.randn(4, 8),
        actions=torch.randn(4, 2),
        rewards=torch.ones(4, 1),
        next_observations=torch.randn(4, 8),
        dones=torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
    )
    monkeypatch.setattr(algorithm.replay_buffer, "sample", lambda _batch_size: batch)
    algorithm.replay_buffer.size = 4
    def fixed_sample(observations):
        return algorithm.actor.deterministic(observations), torch.full((observations.shape[0], 1), 2.0)

    monkeypatch.setattr(algorithm.actor, "sample", fixed_sample)
    captured = {}
    original_project = algorithm.critic.project

    def capture(probabilities, rewards, bootstrap, discount):
        captured["rewards"] = rewards.detach().clone()
        captured["bootstrap"] = bootstrap.detach().clone()
        return original_project(probabilities, rewards, bootstrap, discount)

    monkeypatch.setattr(algorithm.critic, "project", capture)
    algorithm.update()

    assert torch.equal(captured["rewards"][:2], torch.ones(2, 1))
    expected = 1.0 - algorithm.gamma * algorithm.alpha.detach() * 2.0
    assert torch.allclose(captured["rewards"][2:], expected.expand(2, 1))


def test_fast_sac_checkpoint_round_trip_restores_models_optimizer_and_replay() -> None:
    algorithm = FastSAC.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    algorithm.act(_obs())
    algorithm.process_env_step(_obs(), torch.ones(3), torch.zeros(3), {})
    state = algorithm.save()
    restored = FastSAC.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")

    restored.load(state, load_cfg=None, strict=True)

    assert restored.replay_buffer.size == algorithm.replay_buffer.size
    assert torch.equal(restored.actor.mean.weight, algorithm.actor.mean.weight)
    assert torch.equal(restored.log_alpha, algorithm.log_alpha)
