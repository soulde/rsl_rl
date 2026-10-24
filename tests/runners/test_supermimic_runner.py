import torch
from rsl_rl.runners import OffPolicyRunner
from tests.supermimic_helpers import make_mdp, observation, config


def test_reference_schedule_warmup_and_resume_count(tmp_path):
    mdp = make_mdp(tmp_path)

    class Env:
        num_envs = 2
        num_actions = 2
        device = "cpu"
        cfg = {}

        def __init__(self):
            self.unwrapped = self
            self._supermimic_mdp = mdp
            self.frame = 0
            self.bound_sampler = None

        def set_reference_state_sampler(self, sampler, joint_layout):
            self.bound_sampler = (sampler, joint_layout)

        def get_observations(self):
            return observation(mdp, self.frame)

        def reset(self):
            self.frame = 0
            return self.get_observations(), {}

        def step(self, action):
            self.frame += 1
            return self.get_observations(), torch.ones(2) * 0.09, torch.zeros(2, dtype=torch.bool), {}

    cfg = config()
    cfg.update(num_steps_per_env=1, save_interval=0)
    cfg["algorithm"]["wm_only_iters"] = 500
    runner = OffPolicyRunner(Env(), cfg, log_dir=None)
    runner.learn(3)
    assert runner.total_steps == 22  # two envs * (4 warmup + 2 + 3 + 2)
    assert runner.current_learning_iteration == 3
    runner.learn(4)
    assert runner.total_steps == 42  # four additional scheduled iterations, no repeated warmup
    assert runner.current_learning_iteration == 7
    assert cfg["num_steps_per_env"] == 1  # logging must not rewrite the saved schedule config


def test_generic_off_policy_runner_uses_the_shared_schedule_interface(monkeypatch):
    from types import SimpleNamespace
    import rsl_rl.runners.off_policy_runner as module
    from tensordict import TensorDict

    class Algorithm:
        updates = 0
        utd = 0.5
        warmup_steps = 0
        warmup_steps_collected = 0

        @classmethod
        def construct_algorithm(cls, *args):
            return cls()

        def train_mode(self):
            pass

        def act(self, obs):
            return torch.zeros(2, 1)

        def process_env_step(self, *args):
            pass

        def update(self):
            self.updates += 1
            return {"loss": 1.0}

    class Env:
        num_envs = 2
        device = "cpu"
        cfg = {}

        def get_observations(self):
            return TensorDict({"policy": torch.zeros(2, 1)}, batch_size=[2])

        def step(self, actions):
            return self.get_observations(), torch.zeros(2), torch.zeros(2, dtype=torch.bool), {}

    monkeypatch.setattr(module, "resolve_callable", lambda name: Algorithm)
    runner = module.OffPolicyRunner(
        Env(),
        {"algorithm": {"class_name": "test", "learning_starts": 4, "train_every": 2}, "num_steps_per_env": 3},
        log_dir=None,
    )
    runner.learn(2)
    assert runner.total_steps == 8 and runner.alg.updates == 2
    assert runner.current_learning_iteration == 2
