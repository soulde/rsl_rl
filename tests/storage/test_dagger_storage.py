import torch

from rsl_rl.storage import BetaSchedule, DaggerStorage


def test_beta_schedule_is_monotonic_and_clamped() -> None:
    schedule = BetaSchedule(start=1.0, end=0.1, decay=0.5)
    values = [schedule(i) for i in range(5)]
    assert values[0] == 1.0
    assert all(0.1 <= value <= 1.0 for value in values)
    assert values == sorted(values, reverse=True)


def test_dagger_storage_aggregates_and_samples_labeled_transitions() -> None:
    storage = DaggerStorage(capacity=5)
    observations = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    teacher_actions = torch.ones(4, 2)
    student_actions = torch.zeros(4, 2)
    dones = torch.tensor([False, False, True, False])
    storage.add(observations, teacher_actions, student_actions, dones, round_index=2)
    dataset = storage.dataset()
    batch = storage.sample(batch_size=3, generator=torch.Generator().manual_seed(4))
    assert len(storage) == 4
    assert dataset["observations"].shape == (4, 3)
    assert dataset["teacher_actions"].shape == (4, 2)
    assert dataset["round"].tolist() == [2, 2, 2, 2]
    assert batch["observations"].shape == (3, 3)
    assert batch["dones"].dtype == torch.bool


def test_dagger_storage_keeps_latest_samples_when_capacity_is_exceeded() -> None:
    storage = DaggerStorage(capacity=3)
    storage.add(torch.tensor([[1.0], [2.0]]), torch.zeros(2, 1))
    storage.add(torch.tensor([[3.0], [4.0]]), torch.zeros(2, 1))
    assert len(storage) == 3
    assert storage.dataset()["observations"].flatten().tolist() == [2.0, 3.0, 4.0]
