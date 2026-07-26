import torch

from rsl_rl.algorithms.amp import AmpReplayBuffer, compute_amp_reward


def test_amp_reward_matches_lsgan_quadratic():
    predictions = torch.tensor([[-1.0], [0.0], [1.0], [3.0], [4.0]])

    rewards = compute_amp_reward(predictions, coefficient=0.2)

    assert torch.allclose(
        rewards.squeeze(-1),
        torch.tensor([0.0, 0.15, 0.2, 0.0, 0.0]),
    )


def test_amp_replay_buffer_stores_full_transitions():
    buffer = AmpReplayBuffer(capacity=4, observation_dim=6, device="cpu")
    transitions = torch.arange(18, dtype=torch.float32).reshape(3, 6)

    buffer.add(transitions)

    assert buffer.size == 3
    assert buffer.data[:3].shape == (3, 6)
    assert torch.equal(buffer.data[:3], transitions)
