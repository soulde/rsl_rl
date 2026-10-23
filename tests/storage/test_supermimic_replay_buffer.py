import torch

from rsl_rl.storage.supermimic_replay_buffer import SuperMimicReplayBuffer


def _transition(step: int, done: tuple[bool, ...] = (False,)) -> dict[str, torch.Tensor]:
    count = len(done)
    values = torch.arange(count, dtype=torch.float32) * 100 + step
    return {
        "state": values[:, None],
        "next_state": (values + 1)[:, None],
        "observation": values[:, None],
        "next_observation": (values + 1)[:, None],
        "action": (values / 10)[:, None],
        "a_prev": (values / 20)[:, None],
        "reward": values,
        "clean_observation": values[:, None],
        "next_clean_observation": (values + 1)[:, None],
        "terminated": torch.tensor(done),
        "done": torch.tensor(done),
        "context": torch.zeros(count, 1),
        "goal": torch.zeros(count, 1),
        "reference_bin_id": torch.zeros(count, dtype=torch.long),
    }


def test_time_major_ring_samples_window_with_mask_after_episode_boundary() -> None:
    replay = SuperMimicReplayBuffer(capacity=4, num_envs=1, sequence_length=3, batch_size=2, history_length=2)
    replay.add(_transition(0))
    replay.add(_transition(1, done=(True,)))
    replay.add(_transition(2))

    batch = replay.sample_windows()

    assert batch.states.shape == (2, 4, 1)
    assert batch.actions.shape == (2, 3, 1)
    assert batch.mask.shape == (2, 3)
    assert torch.equal(batch.mask[0], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(batch.mask[1], torch.tensor([1.0, 1.0, 0.0]))


def test_start_history_clamps_to_episode_start_and_ring_capacity() -> None:
    replay = SuperMimicReplayBuffer(capacity=3, num_envs=1, sequence_length=2, batch_size=32, history_length=4)
    replay.add(_transition(0))
    replay.add(_transition(1))

    starts, history = replay.sample_starts()

    assert starts["state"].shape == (32, 1)
    assert history["observation"].shape == (32, 4, 1)
    assert torch.all(history["observation"][:, 0] == history["observation"][:, 1])
    assert torch.all(history["observation"][:, 1] == history["observation"][:, 2])
    assert torch.all(history["ep_step"][:, -1] <= 1)

    replay.add(_transition(2))
    replay.add(_transition(3))
    assert replay.size == 3
    assert replay.data["state"][replay._row(torch.tensor([0, 1, 2])), 0, 0].tolist() == [1.0, 2.0, 3.0]


def test_ring_checkpoint_round_trip_preserves_pointer_and_samples() -> None:
    replay = SuperMimicReplayBuffer(capacity=4, num_envs=2, sequence_length=2, batch_size=3, history_length=3)
    for step in range(5):
        replay.add(_transition(step, done=(False, step == 2)))
    state = replay.state_dict()
    restored = SuperMimicReplayBuffer(capacity=4, num_envs=2, sequence_length=2, batch_size=3, history_length=3)
    restored.load_state_dict(state)

    assert restored.size == replay.size == 4
    assert restored.ptr == replay.ptr
    assert torch.equal(restored.data["state"], replay.data["state"])
    assert restored.sample_windows().states.shape == (3, 3, 1)
