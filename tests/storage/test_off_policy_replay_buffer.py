"""Tests for flat off-policy transition replay."""

import pytest
import torch

from rsl_rl.storage import OffPolicyReplayBuffer


def _transitions(start: int, count: int) -> tuple[torch.Tensor, ...]:
    ids = torch.arange(start, start + count, dtype=torch.float32).unsqueeze(-1)
    return ids.repeat(1, 2), ids.repeat(1, 3), ids, (ids + 100).repeat(1, 2), ids % 2


def test_add_and_sample_returns_aligned_batch_shapes() -> None:
    buffer = OffPolicyReplayBuffer(capacity=8, obs_dim=2, action_dim=3, device="cpu")
    buffer.add(*_transitions(0, 5))

    batch = buffer.sample(4)

    assert batch.observations.shape == (4, 2)
    assert batch.actions.shape == (4, 3)
    assert batch.rewards.shape == (4, 1)
    assert batch.next_observations.shape == (4, 2)
    assert batch.dones.shape == (4, 1)
    assert torch.equal(batch.next_observations[:, :1], batch.observations[:, :1] + 100)


def test_ring_wraparound_keeps_only_latest_transitions() -> None:
    buffer = OffPolicyReplayBuffer(capacity=3, obs_dim=2, action_dim=3, device="cpu")
    buffer.add(*_transitions(0, 2))
    buffer.add(*_transitions(2, 3))

    batch = buffer.sample(64)

    assert set(batch.observations[:, 0].tolist()) == {2.0, 3.0, 4.0}
    assert torch.equal(batch.next_observations[:, :1], batch.observations[:, :1] + 100)


def test_sampling_empty_buffer_raises() -> None:
    buffer = OffPolicyReplayBuffer(capacity=4, obs_dim=2, action_dim=3, device="cpu")

    with pytest.raises(RuntimeError, match="empty"):
        buffer.sample(1)


def test_state_round_trip_preserves_data_and_cursor() -> None:
    buffer = OffPolicyReplayBuffer(capacity=4, obs_dim=2, action_dim=3, device="cpu")
    buffer.add(*_transitions(0, 3))
    state = buffer.state_dict()
    restored = OffPolicyReplayBuffer(capacity=4, obs_dim=2, action_dim=3, device="cpu")
    restored.load_state_dict(state)
    restored.add(*_transitions(3, 1))

    batch = restored.sample(64)

    assert set(batch.observations[:, 0].tolist()) == {0.0, 1.0, 2.0, 3.0}
    assert restored.size == 4


@pytest.mark.parametrize("count", [0, -1])
def test_capacity_must_be_positive(count: int) -> None:
    with pytest.raises(ValueError, match="capacity"):
        OffPolicyReplayBuffer(capacity=count, obs_dim=2, action_dim=3, device="cpu")
