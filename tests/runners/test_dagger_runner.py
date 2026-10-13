import torch
from tensordict import TensorDict

from rsl_rl.runners import DaggerRunner
from rsl_rl.storage import BetaSchedule, DaggerStorage


class CounterEnv:
    def __init__(self) -> None:
        self.steps = 0

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.steps += 1
        return actions + 1, torch.tensor([self.steps % 2 == 0] * actions.shape[0])


def test_runner_collects_teacher_labels_and_mixes_actions() -> None:
    storage = DaggerStorage(capacity=20)
    runner = DaggerRunner(
        student_policy=lambda obs: obs * 0,
        teacher_policy=lambda obs: obs * 2,
        storage=storage,
        beta_schedule=BetaSchedule(start=1.0, end=1.0, decay=1.0),
    )
    env = CounterEnv()

    next_obs = runner.collect(torch.ones(2, 1), env.step, num_steps=3, round_index=0, generator=torch.Generator().manual_seed(0))

    assert next_obs.shape == (2, 1)
    assert len(storage) == 6
    assert torch.all(storage.dataset()["student_actions"] == 0)
    assert torch.all(storage.dataset()["teacher_actions"][:2] == 2)
    assert runner.round_index == 1


def test_runner_update_and_resume_round_state() -> None:
    storage = DaggerStorage(capacity=10)
    storage.add(torch.ones(4, 1), torch.ones(4, 1))
    weight = torch.nn.Parameter(torch.zeros(1, 1))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    runner = DaggerRunner(lambda obs: obs @ weight, lambda obs: obs, storage, BetaSchedule())

    metrics = runner.update(optimizer, batch_size=2, epochs=2)
    state = runner.state_dict()
    restored = DaggerRunner(lambda obs: obs, lambda obs: obs, DaggerStorage(10), BetaSchedule())
    restored.load_state_dict(state)

    assert metrics["loss"] >= 0
    assert restored.round_index == runner.round_index


def test_runner_routes_distinct_tensor_dicts_to_student_and_teacher() -> None:
    storage = DaggerStorage(capacity=8)
    student_seen: list[float] = []
    teacher_seen: list[float] = []
    mixed_actions: list[torch.Tensor] = []

    def student_policy(obs: TensorDict) -> torch.Tensor:
        student_seen.append(float(obs["policy"].mean()))
        return torch.zeros(obs.shape[0], 1)

    def teacher_policy(obs: TensorDict) -> torch.Tensor:
        teacher_seen.append(float(obs["privileged"].mean()))
        return obs["privileged"]

    def env_step(actions: torch.Tensor):
        mixed_actions.append(actions.clone())
        next_student = TensorDict(
            {"policy": actions + 1, "depth_camera": torch.ones(2, 1, 2, 2)},
            batch_size=[2],
        )
        next_teacher = TensorDict({"privileged": actions + 10}, batch_size=[2])
        return next_student, next_teacher, torch.zeros(2, dtype=torch.bool)

    runner = DaggerRunner(
        student_policy,
        teacher_policy,
        storage,
        BetaSchedule(start=0.5, end=0.5, decay=1.0),
    )
    student_obs = TensorDict(
        {"policy": torch.ones(2, 1), "depth_camera": torch.zeros(2, 1, 2, 2)},
        batch_size=[2],
    )
    teacher_obs = TensorDict({"privileged": torch.ones(2, 1) * 2}, batch_size=[2])

    next_student_obs, next_teacher_obs = runner.collect(
        student_obs,
        teacher_obs,
        env_step,
        num_steps=2,
        generator=torch.Generator().manual_seed(0),
    )

    assert student_seen == [1.0, 2.0]
    assert teacher_seen == [2.0, 11.0]
    assert torch.equal(mixed_actions[0], torch.tensor([[2.0], [0.0]]))
    assert isinstance(next_student_obs, TensorDict)
    assert isinstance(next_teacher_obs, TensorDict)
    assert len(storage) == 4
    assert storage.dataset()["teacher_actions"].flatten().tolist() == [2.0, 2.0, 12.0, 10.0]
    assert storage.dataset()["observations"]["depth_camera"].shape == (4, 1, 2, 2)
