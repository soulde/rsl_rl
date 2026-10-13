import torch
import pytest
from tensordict import TensorDict

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


def test_dagger_storage_preserves_tensor_dict_image_groups_when_sampling() -> None:
    storage = DaggerStorage(capacity=5)
    observations = TensorDict(
        {
            "policy": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "depth_camera": torch.arange(4 * 1 * 3 * 5, dtype=torch.float32).reshape(4, 1, 3, 5),
        },
        batch_size=[4],
    )
    teacher_actions = torch.ones(4, 2)

    storage.add(observations, teacher_actions)
    dataset = storage.dataset()
    generator = torch.Generator().manual_seed(7)
    indices = torch.randperm(4, generator=generator)[:3]
    batch = storage.sample(batch_size=3, generator=torch.Generator().manual_seed(7))

    assert isinstance(dataset["observations"], TensorDict)
    assert isinstance(batch["observations"], TensorDict)
    assert dataset["observations"]["depth_camera"].shape == (4, 1, 3, 5)
    assert batch["observations"]["depth_camera"].shape == (3, 1, 3, 5)
    assert torch.equal(batch["observations"]["policy"], dataset["observations"]["policy"][indices])


def test_dagger_storage_applies_fifo_capacity_to_tensor_dict_batches() -> None:
    storage = DaggerStorage(capacity=3)

    def observations(values: list[float]) -> TensorDict:
        policy = torch.tensor(values).unsqueeze(-1)
        depth = policy.reshape(-1, 1, 1, 1).expand(-1, 1, 2, 2).clone()
        return TensorDict({"policy": policy, "depth_camera": depth}, batch_size=[len(values)])

    storage.add(observations([1.0, 2.0]), torch.zeros(2, 1))
    storage.add(observations([3.0, 4.0]), torch.zeros(2, 1))

    assert len(storage) == 3
    assert storage.dataset()["observations"]["policy"].flatten().tolist() == [2.0, 3.0, 4.0]


def test_dagger_storage_clear_resets_tensor_dict_payload() -> None:
    storage = DaggerStorage(capacity=3)
    observations = TensorDict(
        {"depth_camera": torch.zeros(2, 1, 2, 2)}, batch_size=[2]
    )
    storage.add(observations, torch.zeros(2, 1))

    storage.clear()

    assert len(storage) == 0
    assert storage.dataset() == {}


def test_dagger_storage_rejects_changed_tensor_dict_schema_without_mutation() -> None:
    storage = DaggerStorage(capacity=4)
    storage.add(
        TensorDict(
            {"policy": torch.zeros(2, 3), "depth_camera": torch.zeros(2, 1, 4, 5)},
            batch_size=[2],
        ),
        torch.zeros(2, 2),
    )

    with pytest.raises(ValueError, match="TensorDict observation keys must remain consistent"):
        storage.add(TensorDict({"policy": torch.ones(1, 3)}, batch_size=[1]), torch.ones(1, 2))

    assert len(storage) == 2
    assert storage.dataset()["observations"]["policy"].sum() == 0


def test_dagger_storage_rejects_changed_tensor_dict_leaf_shape_without_mutation() -> None:
    storage = DaggerStorage(capacity=4)
    storage.add(
        TensorDict({"depth_camera": torch.zeros(2, 1, 4, 5)}, batch_size=[2]),
        torch.zeros(2, 2),
    )

    with pytest.raises(ValueError, match="TensorDict observation leaf shapes must remain consistent"):
        storage.add(TensorDict({"depth_camera": torch.ones(1, 1, 6, 5)}, batch_size=[1]), torch.ones(1, 2))

    assert len(storage) == 2
    assert storage.dataset()["observations"]["depth_camera"].shape == (2, 1, 4, 5)
