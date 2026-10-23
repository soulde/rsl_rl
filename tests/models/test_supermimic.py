import torch

from rsl_rl.models.supermimic import SuperMimicActor, SuperMimicCritic, SuperMimicWorldModel
from rsl_rl.storage.supermimic_replay_buffer import SuperMimicSequenceBatch


def _batch() -> SuperMimicSequenceBatch:
    return SuperMimicSequenceBatch(
        states=torch.randn(3, 4, 5),
        observations=torch.randn(3, 4, 6),
        actions=torch.randn(3, 3, 2).tanh(),
        rewards=torch.randn(3, 3, 1),
        clean_observations=torch.randn(3, 4, 6),
        terminated=torch.zeros(3, 3, 1, dtype=torch.bool),
        contexts=torch.randn(3, 3, 2),
        goals=torch.randn(3, 3, 3),
        reference_bin_ids=torch.zeros(3, 3, dtype=torch.long),
    )


def test_world_model_predicts_each_head_and_has_finite_supervised_gradients() -> None:
    model = SuperMimicWorldModel(5, 2, 6, context_dim=2, goal_dim=3, hidden_dims=(16, 16))
    batch = _batch()
    predictions = model.predict(batch.states[:, 0], batch.actions[:, 0], batch.contexts[:, 0], batch.goals[:, 0])
    losses = model.loss(batch)

    assert predictions["next_state"].shape == (3, 5)
    assert predictions["reward"].shape == (3, 1)
    assert predictions["clean_observation"].shape == (3, 6)
    assert predictions["termination_logit"].shape == (3, 1)
    assert set(losses) == {"total_loss", "state_loss", "reward_loss", "clean_observation_loss", "termination_loss"}
    assert all(torch.isfinite(loss) for loss in losses.values())
    losses["total_loss"].backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_actor_is_deterministic_and_bounded_and_critic_returns_values() -> None:
    actor = SuperMimicActor(6, 2, goal_dim=3, hidden_dims=(16, 16))
    critic = SuperMimicCritic(9, hidden_dims=(16, 16))
    observations, goals = torch.randn(5, 6), torch.randn(5, 3)

    actions = actor(observations, goals)

    assert actions.shape == (5, 2)
    assert torch.equal(actions, actor(observations, goals))
    assert torch.all(actions.abs() <= 1.0)
    assert critic(torch.cat((observations, goals), dim=-1)).shape == (5, 1)


def test_world_model_state_dict_round_trip_preserves_predictions() -> None:
    model = SuperMimicWorldModel(5, 2, 6, hidden_dims=(8,))
    state, action = torch.randn(2, 5), torch.randn(2, 2)
    expected = model.predict(state, action)
    restored = SuperMimicWorldModel(5, 2, 6, hidden_dims=(8,))
    restored.load_state_dict(model.state_dict())

    actual = restored.predict(state, action)

    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
