"""Tests for TD-MPC2 episode sequence replay."""

import pytest
import torch

from rsl_rl.storage import EpisodeReplayBuffer


def _add_steps(buffer: EpisodeReplayBuffer, env_id: int, offset: float, count: int, terminate: bool = True) -> None:
    for step in range(count):
        obs = torch.tensor([offset + step, offset + step + 0.5])
        next_obs = torch.tensor([offset + step + 1, offset + step + 1.5])
        action = torch.tensor([offset + step])
        buffer.add_step(env_id, obs, action, torch.tensor(float(step)), next_obs, terminate and step == count - 1)
    if not terminate:
        buffer.end_episode(env_id)


def test_sampling_keeps_vectorized_environments_in_separate_episodes() -> None:
    buffer = EpisodeReplayBuffer(capacity=20, sequence_length=2, batch_size=4, device="cpu")
    _add_steps(buffer, env_id=0, offset=100, count=3)
    _add_steps(buffer, env_id=1, offset=200, count=3)

    batch = buffer.sample()

    assert batch.observations.shape == (4, 3, 2)
    assert batch.actions.shape == (4, 2, 1)
    assert batch.rewards.shape == (4, 2, 1)
    assert batch.terminated.shape == (4, 2, 1)
    assert torch.allclose(batch.observations[:, 1:, 0], batch.observations[:, :-1, 0] + 1)
    assert set((batch.observations[:, 0, 0] // 100).tolist()) <= {1.0, 2.0}


def test_timeout_closes_episode_without_marking_transition_terminal() -> None:
    buffer = EpisodeReplayBuffer(capacity=10, sequence_length=2, batch_size=2, device="cpu")
    _add_steps(buffer, env_id=3, offset=10, count=2, terminate=False)

    batch = buffer.sample()

    assert buffer.num_episodes == 1
    assert torch.equal(batch.terminated, torch.zeros_like(batch.terminated))


def test_capacity_evicts_oldest_completed_episode() -> None:
    buffer = EpisodeReplayBuffer(capacity=4, sequence_length=2, batch_size=8, device="cpu")
    _add_steps(buffer, env_id=0, offset=100, count=3)
    _add_steps(buffer, env_id=1, offset=200, count=3)

    batch = buffer.sample()

    assert buffer.size == 4
    assert buffer.num_episodes == 2
    assert torch.all(batch.observations[:, 0, 0] >= 200)


def test_capacity_bounds_in_progress_episodes() -> None:
    buffer = EpisodeReplayBuffer(capacity=3, sequence_length=2, batch_size=1, device="cpu")
    for step in range(5):
        observation = torch.tensor([float(step), float(step + 1)])
        buffer.add_step(0, observation, torch.zeros(1), torch.ones(()), observation + 1.0, terminated=False)

    assert buffer.size == 3
    assert sum(map(len, buffer._current.values())) == 3


def test_sequences_shorter_than_requested_length_are_not_sampled() -> None:
    buffer = EpisodeReplayBuffer(capacity=10, sequence_length=2, batch_size=1, device="cpu")
    _add_steps(buffer, env_id=0, offset=0, count=1)

    with pytest.raises(RuntimeError, match="long enough"):
        buffer.sample()


def test_state_round_trip_restores_partial_episode() -> None:
    buffer = EpisodeReplayBuffer(capacity=10, sequence_length=2, batch_size=2, device="cpu")
    for step in range(2):
        obs = torch.tensor([50.0 + step, 50.5 + step])
        buffer.add_step(4, obs, torch.tensor([50.0 + step]), torch.tensor(1.0), obs + 1.0, terminated=False)
    state = buffer.state_dict()
    restored = EpisodeReplayBuffer(capacity=10, sequence_length=2, batch_size=2, device="cpu")
    restored.load_state_dict(state)
    obs = torch.tensor([52.0, 52.5])
    restored.add_step(4, obs, torch.tensor([52.0]), torch.tensor(1.0), obs + 1.0, terminated=True)

    batch = restored.sample()

    assert restored.num_episodes == 1
    assert torch.all((batch.observations[:, 0, 0] >= 50.0) & (batch.observations[:, 0, 0] <= 51.0))
    assert torch.allclose(batch.observations[:, 1:, 0], batch.observations[:, :-1, 0] + 1.0)
    assert restored.state_dict()["episodes"][0][-1]["terminated"].item() == 1.0
