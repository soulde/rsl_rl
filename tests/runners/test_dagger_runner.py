import torch

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
