import copy

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import SuperMimic
from rsl_rl.env import VecEnv


class _Dataset:
    device = "cpu"
    reference_bins = [[(0, 0)], [(0, 1)]]

    def sample_reference_states_by_bin(self, bin_ids):
        return {"sampled_bin_ids": bin_ids.clone()}


class _Env(VecEnv):
    num_envs = 2
    num_actions = 2
    device = "cpu"
    max_episode_length = 8
    episode_length_buf = torch.zeros(2, dtype=torch.long)
    cfg = {}

    def __init__(self):
        self.unwrapped = self
        self._supermimic_reference_dataset = _Dataset()

    def get_observations(self):
        return _obs()

    def step(self, actions):
        return _obs(), torch.ones(2), torch.zeros(2, dtype=torch.bool), {}


def _obs():
    return TensorDict(
        {
            "policy": torch.randn(2, 4),
            "state": torch.randn(2, 5),
            "clean": torch.randn(2, 4),
            "reference_bin_id": torch.tensor([0, 1]),
        },
        batch_size=[2],
    )


def _cfg():
    return {
        "algorithm": {"class_name": "SuperMimic", "horizon": 3, "learning_starts": 0, "gradient_steps": 1},
        "model": {"hidden_dims": [16, 16]},
        "actor": {"hidden_dims": [16, 16]},
        "critic": {"hidden_dims": [16, 16]},
        "memory": {"capacity": 64, "sequence_length": 2, "batch_size": 4},
        "obs_groups": {
            "actor": ["policy"],
            "state": ["state"],
            "clean_observation": ["clean"],
            "critic": ["policy"],
        },
    }


def test_construct_collects_transitions_and_requires_reference_bin_ids() -> None:
    algorithm = SuperMimic.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    actions = algorithm.act(_obs())
    algorithm.process_env_step(
        _obs(),
        torch.ones(2),
        torch.tensor([False, True]),
        {"time_outs": torch.tensor([False, True]), "reference_bin_id": torch.tensor([0, 1])},
    )

    assert actions.shape == (2, 2)
    assert algorithm.replay_buffer.size == 1
    assert algorithm.replay_buffer.num_episodes == 1
    with pytest.raises(ValueError, match="reference_bin_id"):
        algorithm.act(
            TensorDict(
                {"policy": torch.zeros(2, 4), "state": torch.zeros(2, 5), "clean": torch.zeros(2, 4)}, batch_size=[2]
            )
        )


def test_construct_rejects_policy_and_clean_observation_dimension_mismatch() -> None:
    cfg = _cfg()
    cfg["obs_groups"]["clean_observation"] = ["state"]

    with pytest.raises(ValueError, match="clean_observation"):
        SuperMimic.construct_algorithm(_obs(), _Env(), cfg, "cpu")


def test_autoreset_keeps_terminal_transition_and_reset_history_separate() -> None:
    algorithm = SuperMimic.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    current = _obs()
    algorithm.act(current)
    reset_obs = _obs()
    final_obs = _obs()
    final_obs["policy"] = torch.full_like(final_obs["policy"], 99.0)
    final_obs["clean"] = torch.full_like(final_obs["clean"], 99.0)

    algorithm.process_env_step(
        reset_obs,
        torch.ones(2),
        torch.tensor([True, False]),
        {"final_observation": final_obs, "time_outs": torch.tensor([False, False])},
    )

    torch.testing.assert_close(algorithm.replay_buffer.data["next_observation"][0, 0], final_obs["policy"][0])
    torch.testing.assert_close(algorithm._obs_history[0], reset_obs["policy"][0].expand(10, -1))


def test_construct_allows_pre_reset_observation_but_act_requires_reset_bin_id() -> None:
    observation = TensorDict(
        {"policy": torch.zeros(2, 4), "state": torch.zeros(2, 5), "clean": torch.zeros(2, 4)},
        batch_size=[2],
    )
    algorithm = SuperMimic.construct_algorithm(observation, _Env(), _cfg(), "cpu")

    with pytest.raises(ValueError, match="reference_bin_id"):
        algorithm.act(observation)


def test_act_rejects_out_of_range_reference_bin_id() -> None:
    algorithm = SuperMimic.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    observation = _obs()
    observation["reference_bin_id"][0] = 2

    with pytest.raises(ValueError, match="reference_bin_id.*range"):
        algorithm.act(observation)


def test_update_trains_model_actor_and_critic_with_finite_losses() -> None:
    algorithm = SuperMimic.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    for _ in range(3):
        obs = _obs()
        algorithm.act(obs)
        algorithm.process_env_step(
            obs, torch.ones(2), torch.zeros(2, dtype=torch.bool), {"reference_bin_id": obs["reference_bin_id"]}
        )
    metrics = algorithm.update()

    assert set(metrics) >= {"model_loss", "actor_loss", "critic_loss"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert algorithm.num_updates == 1


def test_imagined_actor_update_backpropagates_through_frozen_world_model() -> None:
    algorithm = SuperMimic.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    for step in range(3):
        obs = _obs()
        algorithm.act(obs)
        algorithm.process_env_step(
            obs,
            torch.ones(2),
            torch.full((2,), step == 2),
            {"reference_bin_id": obs["reference_bin_id"]},
        )
    batch = algorithm.replay_buffer.sample_windows()
    model_before = [parameter.detach().clone() for parameter in algorithm.model.parameters()]

    algorithm._update_imagined_policy(batch)

    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in algorithm.actor.parameters()
    )
    assert all(torch.equal(before, after) for before, after in zip(model_before, algorithm.model.parameters()))
    assert all(parameter.requires_grad for parameter in algorithm.model.parameters())


def test_gae_uses_terminal_masks_and_bootstrap_value() -> None:
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.25]])
    next_value = torch.tensor([0.75])
    terminated = torch.tensor([[False], [True]])

    advantages, returns = SuperMimic.compute_gae(rewards, values, next_value, terminated, gamma=0.9, gae_lambda=0.95)

    assert advantages.shape == returns.shape == (2, 1)
    torch.testing.assert_close(advantages[-1], torch.tensor([1.75]))
    torch.testing.assert_close(returns, advantages + values)


def test_save_load_restores_replay_sampler_and_optimizers() -> None:
    algorithm = SuperMimic.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")
    state = algorithm.save()
    restored = SuperMimic.construct_algorithm(_obs(), _Env(), _cfg(), "cpu")

    restored.load(copy.deepcopy(state))

    assert restored.num_updates == algorithm.num_updates
    assert torch.equal(restored.sampler.probabilities, algorithm.sampler.probabilities)
