import copy
import pytest
import torch
from rsl_rl.algorithms import SuperMimic
from tests.supermimic_helpers import make_mdp, make_env, observation, config


def algorithm(tmp_path):
    mdp = make_mdp(tmp_path)
    return SuperMimic.construct_algorithm(observation(mdp), make_env(mdp), config(), "cpu")


def collect(alg, count=4):
    for step in range(count):
        obs = observation(alg.mdp, step)
        alg.act(obs)
        alg.process_env_step(observation(alg.mdp, step + 1), torch.ones(2) * 0.09, torch.zeros(2, dtype=torch.bool), {})


def test_collect_update_analytic_actor_gradients_and_normalizers(tmp_path):
    alg = algorithm(tmp_path)
    collect(alg)
    before = [p.clone() for p in alg.actor.parameters()]
    metrics = alg.update(0)
    assert {"model_loss", "actor_loss", "critic_loss"} <= metrics.keys()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, alg.actor.parameters()))
    assert alg.model.in_norm.count == 8 and alg.actor.obs_norm.count == 8
    assert all(p.requires_grad for p in alg.model.parameters())
    assert set(alg.replay_buffer.data) == set(alg.replay_buffer._FIELDS)


def test_terminal_mapping_preserves_final_state_and_timeout_bootstrap(tmp_path):
    alg = algorithm(tmp_path)
    alg.act(observation(alg.mdp))
    reset = observation(alg.mdp, 0)
    final = observation(alg.mdp, 1)
    final["state"][0, 0] = 3.0
    alg.process_env_step(
        reset,
        torch.tensor([0.1, 0.1]),
        torch.tensor([True, False]),
        {"final_obs": dict(final.items()), "time_outs": torch.tensor([True, False])},
    )
    assert alg.replay_buffer.data["next_state"][0, 0, 0] == 3
    assert alg.dataset.reference_state_sampler.max_return == pytest.approx(10.0)
    assert alg.dataset.reference_state_sampler.return_estimates[0] == pytest.approx(2.0)
    assert not hasattr(alg, "sampler")
    alg.act(reset)
    with pytest.raises(RuntimeError, match="pre-reset"):
        alg.process_env_step(reset, torch.zeros(2), torch.ones(2, dtype=torch.bool), {})


def test_reference_schedule_model_only_stage_and_lr_decay(tmp_path):
    alg = algorithm(tmp_path)
    assert [alg.collection_steps(i) for i in range(4)] == [2, 3, 2, 3]
    collect(alg)
    alg.wm_only_iters = 500
    before = [p.clone() for p in alg.actor.parameters()]
    assert "actor_loss" not in alg.update(1)
    assert all(torch.equal(a, b) for a, b in zip(before, alg.actor.parameters()))
    alg.update(alg.lr_decay_iters)
    assert alg.learning_rate == pytest.approx(alg.initial_lr * 0.03)


def test_algorithm_binds_rsi_through_public_task_api_before_reset(tmp_path):
    alg = algorithm(tmp_path)

    class Task:
        def set_reference_state_sampler(self, sampler, layout):
            self.sampler, self.layout = sampler, layout

    class Env:
        def __init__(self):
            self.unwrapped = Task()
            self.reset_called = False

        def reset(self):
            assert callable(self.unwrapped.sampler)
            assert self.unwrapped.layout["joint_names"] == tuple(alg.mdp.dataset.joint_names)
            self.reset_called = True

    env = Env()
    alg.initialize_env(env)

    assert env.unwrapped.sampler.__self__ is alg
    assert env.reset_called


def test_checkpoint_version_and_restore(tmp_path):
    alg = algorithm(tmp_path)
    collect(alg)
    alg.update()
    state = copy.deepcopy(alg.save())
    new = algorithm(tmp_path)
    new.load(state)
    assert new.num_updates == alg.num_updates and new.num_steps == alg.num_steps
    assert torch.equal(new.actor.obs_norm.mean, alg.actor.obs_norm.mean)
    with pytest.raises(ValueError, match="Incompatible"):
        new.load({"actor_state_dict": {}})


def test_imagination_rejects_clip_tail_starts(tmp_path):
    alg = algorithm(tmp_path)
    collect(alg)
    alg.replay_buffer.data["reference_frame_id"].fill_(79)
    with pytest.raises(RuntimeError, match="valid reference"):
        alg._sample_valid_starts()


def test_lambda_returns_and_soft_termination():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.25]])
    advantages, returns = SuperMimic.compute_gae(
        rewards, values, torch.tensor([0.75]), torch.tensor([[0.0], [1.0]]), 0.9, 0.95
    )
    torch.testing.assert_close(returns[-1], torch.tensor([2.0]))
    torch.testing.assert_close(returns, advantages + values)


def test_collection_policy_has_reference_action_clipping(tmp_path):
    alg = algorithm(tmp_path)
    alg.action_noise = 0
    with torch.no_grad():
        alg.actor.network[-1].bias.fill_(30)
    action = alg.act(observation(alg.mdp))
    torch.testing.assert_close(action, torch.full_like(action, 20.0))


def test_replay_resume_closes_old_session_boundary(tmp_path):
    alg = algorithm(tmp_path)
    alg.save_replay = True
    collect(alg)
    state = copy.deepcopy(alg.save())
    new = algorithm(tmp_path)
    new.load(state)
    row = (new.replay_buffer.ptr - 1) % new.replay_buffer.capacity
    assert new.replay_buffer.data["done"][row].all()
    assert not new.replay_buffer.episode_steps.any()
    collect(new, 1)
    batch = new.replay_buffer.sample_windows(128)
    # Any sampled old-session suffix after the marked row must be masked.
    assert (batch.mask[:, 1:] <= (1 - batch.done[:, :-1].float())).all()


def test_inference_history_reset_uses_next_reset_observation(tmp_path):
    alg = algorithm(tmp_path)
    policy = alg.get_policy()
    first = observation(alg.mdp)
    first["policy"].fill_(1.0)
    policy(first)
    policy.reset(torch.tensor([True, False]))
    fresh = observation(alg.mdp)
    fresh["policy"].fill_(7.0)
    policy(fresh)
    torch.testing.assert_close(policy._history[0], torch.full_like(policy._history[0], 7.0))
    torch.testing.assert_close(policy._history[1, :-1], torch.ones_like(policy._history[1, :-1]))
