"""Tests for off-policy algorithm orchestration through the native runner API."""

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OffPolicyRunner


class _Policy(torch.nn.Module):
    def forward(self, obs: TensorDict) -> torch.Tensor:
        return torch.zeros(obs.batch_size[0], 1)


class _Algorithm:
    instances = []

    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        instance = cls()
        cls.instances.append(instance)
        return instance

    def __init__(self):
        self.updates = 0
        self.training_mode = None
        self.policy = _Policy()

    def act(self, obs):
        return torch.zeros(obs.batch_size[0], 1)

    def process_env_step(self, obs, rewards, dones, extras):
        pass

    def update(self):
        self.updates += 1
        return {"loss": 1.0}

    def train_mode(self):
        self.training_mode = True

    def eval_mode(self):
        self.training_mode = False

    def save(self):
        return {"updates": self.updates}

    def load(self, loaded_dict, load_cfg, strict):
        self.updates = loaded_dict["updates"]
        return True

    def get_policy(self):
        return self.policy


class _Env(VecEnv):
    def __init__(self):
        self.num_envs = 2
        self.num_actions = 1
        self.max_episode_length = 8
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}
        self.steps = 0

    def get_observations(self):
        return TensorDict({"policy": torch.full((2, 3), float(self.steps))}, batch_size=[2])

    def step(self, actions):
        self.steps += 1
        return self.get_observations(), torch.ones(2), torch.zeros(2), {}


def _cfg() -> dict:
    return {
        "algorithm": {"class_name": "Fake", "learning_starts": 3, "train_every": 1},
        "num_steps_per_env": 2,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    }


def test_runner_warms_up_and_schedules_updates(monkeypatch) -> None:
    from rsl_rl.runners import off_policy_runner

    _Algorithm.instances.clear()
    monkeypatch.setattr(off_policy_runner, "resolve_callable", lambda name: _Algorithm)
    runner = OffPolicyRunner(_Env(), _cfg(), log_dir=None, device="cpu")

    runner.learn(num_learning_iterations=1)

    assert runner.total_steps == 4
    assert _Algorithm.instances[0].updates == 1


def test_runner_save_load_and_inference_policy(tmp_path, monkeypatch) -> None:
    from rsl_rl.runners import off_policy_runner

    _Algorithm.instances.clear()
    monkeypatch.setattr(off_policy_runner, "resolve_callable", lambda name: _Algorithm)
    runner = OffPolicyRunner(_Env(), _cfg(), log_dir=None, device="cpu")
    runner.alg.updates = 7
    runner.current_learning_iteration = 7
    checkpoint = tmp_path / "runner.pt"
    runner.save(str(checkpoint))
    runner.alg.updates = 0
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy()

    assert runner.alg.updates == 7
    assert runner.current_learning_iteration == 7
    assert runner.alg.training_mode is False
    assert isinstance(policy, _Policy)


class _NativeEnv(_Env):
    def __init__(self, obs_dim: int, action_dim: int):
        self.num_envs = 2
        self.num_actions = action_dim
        self.max_episode_length = 8
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}
        self.obs_dim = obs_dim

    def get_observations(self):
        return TensorDict({"policy": torch.randn(2, self.obs_dim)}, batch_size=[2])

    def step(self, actions):
        return self.get_observations(), torch.ones(2), torch.zeros(2, dtype=torch.bool), {}


def _native_cfg(algorithm: str) -> dict:
    cfg = {
        "algorithm": {"class_name": algorithm, "learning_starts": 100, "train_every": 1},
        "num_steps_per_env": 1,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "memory": {"capacity": 16, "batch_size": 2},
    }
    if algorithm == "FastSAC":
        cfg.update(
            actor={"hidden_dims": [8], "log_std_min": -5.0, "log_std_max": 1.0},
            critic={"hidden_dims": [8], "num_atoms": 5, "v_min": -5.0, "v_max": 5.0},
        )
        cfg["algorithm"].update(gamma=0.99, tau=0.01, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4)
        return cfg
    cfg["algorithm"].update(
        horizon=2,
        num_samples=4,
        num_elites=2,
        num_pi_trajs=0,
        iterations=1,
        min_std=0.05,
        max_std=1.0,
        temperature=0.5,
        mpc=True,
        episodic=True,
        tau=0.1,
        lr=1e-3,
    )
    cfg["model"] = {
        "class_name": "TDMPC2WorldModel",
        "latent_dim": 8,
        "enc_dim": 8,
        "mlp_dim": 16,
        "num_enc_layers": 1,
        "num_q": 2,
        "num_bins": 5,
        "vmin": -5.0,
        "vmax": 5.0,
        "simnorm_dim": 4,
        "dropout": 0.0,
    }
    return cfg


def test_native_algorithms_construct_train_checkpoint_and_infer(tmp_path) -> None:
    from rsl_rl.algorithms import FastSAC, TDMPC2

    for name, algorithm_type, obs_dim, action_dim in (
        ("FastSAC", FastSAC, 3, 2),
        ("TDMPC2", TDMPC2, 7, 2),
    ):
        runner = OffPolicyRunner(_NativeEnv(obs_dim, action_dim), _native_cfg(name), log_dir=None, device="cpu")
        assert isinstance(runner.alg, algorithm_type)
        runner.learn(num_learning_iterations=1)
        checkpoint = tmp_path / f"{name}.pt"
        runner.save(str(checkpoint))

        restored = OffPolicyRunner(_NativeEnv(obs_dim, action_dim), _native_cfg(name), log_dir=None, device="cpu")
        restored.load(str(checkpoint))
        policy = restored.get_inference_policy()

        assert restored.total_steps == 2
        assert restored.current_learning_iteration == 0
        inference_obs = TensorDict({"policy": torch.randn(2, obs_dim)}, batch_size=[2])
        assert policy(inference_obs).shape == (2, action_dim)
