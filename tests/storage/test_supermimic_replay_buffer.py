import torch

from rsl_rl.storage.supermimic_replay_buffer import SuperMimicReplayBuffer


def _transition(index: int, *, terminated: bool = False) -> dict[str, torch.Tensor]:
    value = float(index)
    return {
        "state": torch.tensor([value, value + 1]),
        "next_state": torch.tensor([value + 1, value + 2]),
        "observation": torch.tensor([value]),
        "next_observation": torch.tensor([value + 1]),
        "action": torch.tensor([value / 10]),
        "reward": torch.tensor(value),
        "clean_observation": torch.tensor([value + 2]),
        "next_clean_observation": torch.tensor([value + 3]),
        "terminated": torch.tensor(terminated),
        "context": torch.tensor([0.25]),
        "goal": torch.tensor([0.5]),
        "reference_bin_id": torch.tensor(2),
    }


def test_samples_contiguous_windows_with_all_transition_fields() -> None:
    replay = SuperMimicReplayBuffer(capacity=32, sequence_length=2, batch_size=8)
    for index in range(4):
        replay.add_step(0, _transition(index, terminated=index == 3))

    batch = replay.sample()

    assert batch.states.shape == (8, 3, 2)
    assert batch.observations.shape == (8, 3, 1)
    assert batch.actions.shape == (8, 2, 1)
    assert batch.clean_observations.shape == (8, 3, 1)
    assert batch.contexts.shape == (8, 2, 1)
    assert batch.goals.shape == (8, 2, 1)
    assert batch.reference_bin_ids.shape == (8, 2)
    assert torch.all(batch.states[:, 1:, 0] == batch.states[:, :-1, 0] + 1)
    assert torch.allclose(batch.actions[:, 1:, 0] - batch.actions[:, :-1, 0], torch.full((8, 1), 0.1))


def test_episode_boundaries_are_not_crossed_and_timeout_can_close_without_terminal() -> None:
    replay = SuperMimicReplayBuffer(capacity=32, sequence_length=2, batch_size=16)
    for index in range(2):
        replay.add_step(0, _transition(index))
    replay.end_episode(0)
    for index in range(10, 12):
        replay.add_step(0, _transition(index))
    replay.end_episode(0)

    batch = replay.sample()

    assert torch.all((batch.states[:, 0, 0] < 2) | (batch.states[:, 0, 0] >= 10))
    assert torch.all(batch.states[:, 1:, 0] == batch.states[:, :-1, 0] + 1)
    assert torch.all(batch.terminated == 0)


def test_capacity_and_checkpoint_round_trip_preserve_sampleable_episodes() -> None:
    replay = SuperMimicReplayBuffer(capacity=4, sequence_length=2, batch_size=2)
    for episode_id in range(3):
        for step in range(2):
            replay.add_step(episode_id, _transition(episode_id * 10 + step))
        replay.end_episode(episode_id)
    state = replay.state_dict()
    restored = SuperMimicReplayBuffer(capacity=4, sequence_length=2, batch_size=2)
    restored.load_state_dict(state)

    assert replay.size == restored.size == 4
    assert restored.num_episodes == 2
    assert restored.sample().states.shape == (2, 3, 2)
