"""Numerical AMP-Go2 contracts expressed through native RSL-RL interfaces."""

import copy

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.amp import AMP, AmpNormalizer
from rsl_rl.storage import RolloutStorage
from tests.algorithms.test_ppo import _make_actor, _make_critic


def _build_amp(**kwargs):
    obs = TensorDict({"policy": torch.randn(4, 3), "amp": torch.randn(4, 2)}, batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = _make_actor(obs, groups, num_actions=2)
    critic = _make_critic(obs, groups)
    storage = RolloutStorage("rl", 4, 2, obs, [2])
    options = dict(
        discriminator_hidden_dims=[8, 4],
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=5e-4,
        schedule="fixed",
    )
    options.update(kwargs)
    algorithm = AMP(
        actor, critic, storage,
        amp_observation_groups=["amp"],
        amp_observation_dim=2,
        collect_reference_motions=lambda count: torch.ones(count, 4),
        **options,
    )
    return algorithm, obs


def _reference_moments(batches):
    # AMP-Go2 utils.RunningMeanStd.update_from_moments uses a 1e-4 prior.
    mean, variance, count = np.zeros(2), np.ones(2), 1e-4
    for batch in batches:
        batch = batch.numpy().astype(np.float64)
        delta = batch.mean(0) - mean
        total = count + len(batch)
        moment = variance * count + batch.var(0) * len(batch) + delta**2 * count * len(batch) / total
        mean = mean + delta * len(batch) / total
        variance, count = moment / total, total
    return mean, variance, count


def test_normalizer_matches_reference_moments_epsilon_and_clipping():
    normalizer = AmpNormalizer(2)
    batches = [torch.tensor([[2.0, 4.0], [6.0, 8.0]]), torch.tensor([[10.0, -2.0]])]
    for batch in batches:
        normalizer.update(batch)
    mean, variance, count = _reference_moments(batches)
    inputs = torch.tensor([[1e4, -1e4], [3.0, 5.0]])
    reference = ((inputs - torch.tensor(mean, dtype=torch.float32)) /
                 torch.tensor(variance + 1e-4, dtype=torch.float32).sqrt()).clamp(-10, 10)

    assert torch.allclose(normalizer._mean[0], torch.tensor(mean), atol=1e-12)
    assert torch.allclose(normalizer._var[0], torch.tensor(variance), atol=1e-12)
    assert normalizer.count.item() == pytest.approx(count)
    assert torch.equal(normalizer(inputs), reference)


def test_loss_gradients_adam_and_post_step_normalizer_match_reference(monkeypatch):
    algorithm, _obs = _build_amp()
    policy = torch.tensor([[1., 2., 20., 40.], [3., 4., 60., 80.]])
    expert = torch.tensor([[5., 6., 100., 120.], [7., 8., 140., 160.]])
    algorithm.amp_replay_buffer.add(policy)
    monkeypatch.setattr(algorithm.amp_replay_buffer, "sample", lambda count: policy)
    algorithm.collect_reference_motions = lambda count: expert
    reference_model = copy.deepcopy(algorithm.discriminator)
    reference_optimizer = torch.optim.Adam([
        {"params": list(reference_model.parameters())[:-2], "weight_decay": 1e-3},
        {"params": list(reference_model.parameters())[-2:], "weight_decay": 1e-1},
    ], lr=algorithm.learning_rate)

    loss, metrics = algorithm._compute_auxiliary_loss(None, 2)
    policy_normalized = (policy / torch.tensor(1.0001).sqrt()).clamp(-10, 10)
    expert_normalized = (expert / torch.tensor(1.0001).sqrt()).clamp(-10, 10).requires_grad_(True)
    policy_prediction = reference_model(policy_normalized)
    expert_prediction = reference_model(expert_normalized)
    prediction_loss = 0.5 * ((policy_prediction + 1).square().mean() + (expert_prediction - 1).square().mean())
    gradient = torch.autograd.grad(expert_prediction.sum(), expert_normalized, create_graph=True)[0]
    reference_loss = prediction_loss + 10 * gradient.square().sum(-1).mean()

    assert torch.allclose(loss, reference_loss)
    assert algorithm.amp_normalizer.count.item() == pytest.approx(1e-4)
    assert torch.allclose(metrics["amp_policy_prediction"], policy_prediction.mean())
    algorithm.optimizer.zero_grad()
    loss.backward()
    reference_loss.backward()
    for actual, expected in zip(algorithm.discriminator.parameters(), reference_model.parameters()):
        assert torch.allclose(actual.grad, expected.grad, atol=1e-7)
    algorithm.optimizer.step()
    reference_optimizer.step()
    algorithm._after_optimizer_step()
    for actual, expected in zip(algorithm.discriminator.parameters(), reference_model.parameters()):
        assert torch.allclose(actual, expected, atol=1e-7)
    mean, variance, count = _reference_moments([policy[:, :2], expert[:, :2]])
    assert torch.allclose(algorithm.amp_normalizer._mean[0], torch.tensor(mean), atol=1e-12)
    assert torch.allclose(algorithm.amp_normalizer._var[0], torch.tensor(variance), atol=1e-12)
    assert algorithm.amp_normalizer.count.item() == pytest.approx(count)


@pytest.mark.parametrize("task_fraction", [0.0, 0.8, 1.0])
def test_collection_mixes_rewards_and_inserts_terminal_states_before_update(task_fraction):
    algorithm, obs = _build_amp(task_reward_lerp=task_fraction)
    for parameter in algorithm.discriminator.parameters():
        parameter.data.zero_()
    with torch.no_grad():
        algorithm.act(obs)
        next_obs = obs.clone()
        next_obs["amp"] = torch.zeros(4, 2)
        algorithm.process_env_step(
            next_obs, torch.full((4,), 2.0), torch.tensor([True, False, False, False]),
            {"terminal_amp_observations": torch.full((4, 2), 9.0)},
        )
    assert algorithm.amp_replay_buffer.size == 4
    assert torch.equal(algorithm.amp_replay_buffer.data[0, 2:], torch.full((2,), 9.0))
    # d=0 -> bounded reward 0.75; AMP-Go2 scales it by 0.2 before lerp.
    expected = task_fraction * 2.0 + (1 - task_fraction) * 0.2 * 0.75
    assert torch.allclose(algorithm.storage.rewards[0], torch.full((4, 1), expected))


def test_full_update_runs_one_discriminator_step_per_ppo_minibatch():
    algorithm, obs = _build_amp(min_std=[0.1, 0.2])
    for _ in range(2):
        with torch.no_grad():
            algorithm.act(obs)
            algorithm.process_env_step(obs, torch.ones(4), torch.zeros(4), {})
    with torch.no_grad():
        algorithm.compute_returns(obs)
    losses = algorithm.update()

    assert algorithm.discriminator_updates == 4
    assert algorithm.discriminator_batch_size == 4
    assert algorithm.amp_replay_buffer.size == 8
    assert algorithm.amp_normalizer.count.item() == pytest.approx(32.0001)
    assert all(np.isfinite(value) for value in losses.values())
    assert [group["weight_decay"] for group in algorithm.optimizer.param_groups[-2:]] == [1e-3, 1e-1]
    assert isinstance(algorithm.discriminator[1], torch.nn.LeakyReLU)
    assert algorithm._amp_normalizer_samples is None


def test_minimum_std_is_applied_to_parameters_after_optimizer_step():
    algorithm, _obs = _build_amp(min_std=[0.1, 0.2])
    algorithm.actor.distribution.std_param.data.fill_(0.01)
    algorithm._after_optimizer_step()
    assert torch.equal(algorithm.actor.distribution.std_param, torch.tensor([0.1, 0.2]))


def test_terminal_transition_requires_pre_reset_states():
    algorithm, obs = _build_amp()
    with torch.no_grad():
        algorithm.act(obs)
        with pytest.raises(ValueError, match="pre-reset AMP states"):
            algorithm.process_env_step(obs, torch.ones(4), torch.tensor([True, False, False, False]), {})


def test_compact_terminal_state_batch_is_inserted_in_done_mask_order():
    algorithm, obs = _build_amp()
    with torch.no_grad():
        algorithm.act(obs)
        algorithm.process_env_step(
            obs, torch.ones(4), torch.tensor([False, True, False, True]),
            {"terminal_amp_observations": torch.tensor([[11., 12.], [13., 14.]])},
        )
    assert torch.equal(algorithm.amp_replay_buffer.data[[1, 3], 2:], torch.tensor([[11., 12.], [13., 14.]]))


def test_legacy_reward_scales_preserve_their_numeric_weights():
    algorithm, _obs = _build_amp(task_reward_scale=0.6, style_reward_scale=0.4)
    assert algorithm.task_reward_lerp == 0.6
    assert (1 - algorithm.task_reward_lerp) * algorithm.amp_reward_coef == pytest.approx(0.4)


def test_discriminator_batch_size_uses_transition_count_for_recurrent_batches(monkeypatch):
    algorithm, _obs = _build_amp()
    algorithm.amp_replay_buffer.add(torch.ones(8, 4))
    requested = []
    sample = algorithm.amp_replay_buffer.sample

    def capture(count):
        requested.append(count)
        return sample(count)

    monkeypatch.setattr(algorithm.amp_replay_buffer, "sample", capture)
    algorithm._compute_auxiliary_loss(None, original_batch_size=2)

    assert requested == [4]


def test_amp_clips_mean_network_and_critic_without_clipping_std():
    algorithm, _obs = _build_amp(max_grad_norm=1.0)
    for parameter in algorithm.actor.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)
    for parameter in algorithm.critic.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)

    algorithm._clip_gradients()

    assert torch.equal(algorithm.actor.distribution.std_param.grad, torch.full((2,), 100.0))
    network_norm = torch.stack([p.grad.norm() for p in algorithm.actor.mlp.parameters()]).norm()
    critic_norm = torch.stack([p.grad.norm() for p in algorithm.critic.parameters()]).norm()
    assert network_norm.item() == pytest.approx(1.0, abs=1e-6)
    assert critic_norm.item() == pytest.approx(1.0, abs=1e-6)
