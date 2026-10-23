"""SuperMimic model-based imitation learning for vectorized environments."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F

from rsl_rl.datasets import ReturnAwareReferenceSampler
from rsl_rl.env import VecEnv
from rsl_rl.models.supermimic import SuperMimicActor, SuperMimicCritic, SuperMimicWorldModel
from rsl_rl.storage import SuperMimicReplayBuffer, SuperMimicSequenceBatch
from rsl_rl.utils import resolve_obs_groups


class SuperMimic:
    """Learn a differentiable dynamics model and optimize a policy in imagined rollouts."""

    requires_reference_state_initialization = True

    def __init__(
        self,
        model: SuperMimicWorldModel,
        actor: SuperMimicActor,
        critic: SuperMimicCritic,
        replay_buffer: SuperMimicReplayBuffer,
        sampler: ReturnAwareReferenceSampler,
        obs_groups: dict[str, list[str]],
        batch_size: int,
        device: str = "cpu",
        horizon: int = 32,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        learning_starts: int = 0,
        gradient_steps: int = 1,
        train_every: int = 1,
        lr: float = 2e-4,
        grad_clip_norm: float = 1.0,
    ) -> None:
        if batch_size <= 0 or horizon <= 0 or gradient_steps <= 0 or train_every <= 0:
            raise ValueError("SuperMimic batch size, horizon, gradient steps, and train_every must be positive")
        if learning_starts < 0 or not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError("Invalid SuperMimic learning_starts, gamma, or gae_lambda")
        self.device = torch.device(device)
        self.model, self.actor, self.critic = model.to(device), actor.to(device), critic.to(device)
        self.replay_buffer, self.sampler = replay_buffer, sampler
        self.obs_groups = obs_groups
        self.batch_size = batch_size
        self.horizon, self.gamma, self.gae_lambda = horizon, gamma, gae_lambda
        self.learning_starts, self.gradient_steps, self.train_every = learning_starts, gradient_steps, train_every
        self.grad_clip_norm = grad_clip_norm
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.num_updates = 0
        self.num_steps = 0
        self.training = True
        self._transition: dict[str, torch.Tensor] | None = None
        self._episode_returns: torch.Tensor | None = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Choose a deterministic action and cache the current transition fields."""
        self._require_bin_ids(obs)
        actor_obs = self._extract(obs, "actor")
        state = self._extract(obs, "state")
        clean = self._extract(obs, "clean_observation")
        context = self._extract(obs, "context")
        goal = self._extract(obs, "goal")
        bin_ids = obs["reference_bin_id"].to(self.device, dtype=torch.long).reshape(-1)
        if self._episode_returns is None or self._episode_returns.shape[0] != actor_obs.shape[0]:
            self._episode_returns = torch.zeros(actor_obs.shape[0], device=self.device)
        with torch.no_grad():
            if self.training and self.num_steps < self.learning_starts:
                actions = torch.rand(actor_obs.shape[0], self.actor.action_dim, device=self.device) * 2 - 1
            else:
                actions = self.actor(actor_obs, goal if self.actor.goal_dim else None)
        self._transition = {
            "state": state.detach(),
            "observation": actor_obs.detach(),
            "clean_observation": clean.detach(),
            "context": context.detach(),
            "goal": goal.detach(),
            "reference_bin_id": bin_ids.detach(),
            "action": actions.detach(),
        }
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Store transitions, close completed trajectories, and update bin returns."""
        if self._transition is None:
            raise RuntimeError("SuperMimic.process_env_step() called before act().")
        next_observations = obs
        dones = dones.to(self.device).bool().reshape(-1)
        final_observations = extras.get("final_observation", extras.get("final_obs", extras.get("terminal_observation")))
        if final_observations is not None:
            if not isinstance(final_observations, TensorDict):
                raise TypeError("final_observation must be a TensorDict")
            next_observations = obs.clone()
            next_observations[dones] = final_observations[dones]
        timeouts = extras.get("time_outs", torch.zeros_like(dones)).to(self.device).bool().reshape(-1)
        terminated = extras.get("terminated", dones & ~timeouts).to(self.device).bool().reshape(-1)
        next_state = self._extract(next_observations, "state")
        next_actor_obs = self._extract(next_observations, "actor")
        next_clean = self._extract(next_observations, "clean_observation")
        rewards = rewards.to(self.device, dtype=torch.float32).reshape(-1)
        self._episode_returns += rewards
        current = self._transition
        for env_id in range(dones.numel()):
            transition = {
                "state": current["state"][env_id],
                "next_state": next_state[env_id],
                "observation": current["observation"][env_id],
                "next_observation": next_actor_obs[env_id],
                "action": current["action"][env_id],
                "reward": rewards[env_id],
                "clean_observation": current["clean_observation"][env_id],
                "next_clean_observation": next_clean[env_id],
                "terminated": terminated[env_id],
                "context": current["context"][env_id],
                "goal": current["goal"][env_id],
                "reference_bin_id": current["reference_bin_id"][env_id],
            }
            self.replay_buffer.add_step(env_id, transition)
            if bool(dones[env_id]):
                if not bool(terminated[env_id]):
                    self.replay_buffer.end_episode(env_id)
                self.sampler.update_returns(
                    current["reference_bin_id"][env_id : env_id + 1], self._episode_returns[env_id : env_id + 1]
                )
                self._episode_returns[env_id] = 0.0
        self.num_steps += dones.numel()
        self._transition = None

    def update(self) -> dict[str, float]:
        metrics = {}
        for _ in range(self.gradient_steps):
            try:
                batch = self.replay_buffer.sample()
            except RuntimeError as error:
                if "completed episode" in str(error):
                    return metrics or {"model_loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0}
                raise
            batch = self._batch_to_device(batch)
            model_losses = self.model.loss(batch)
            self.model_optimizer.zero_grad(set_to_none=True)
            model_losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.model_optimizer.step()
            actor_loss, critic_loss = self._update_imagined_policy(batch)
            metrics = {
                "model_loss": float(model_losses["total_loss"].detach()),
                "state_loss": float(model_losses["state_loss"].detach()),
                "reward_loss": float(model_losses["reward_loss"].detach()),
                "clean_observation_loss": float(model_losses["clean_observation_loss"].detach()),
                "termination_loss": float(model_losses["termination_loss"].detach()),
                "actor_loss": float(actor_loss.detach()),
                "critic_loss": float(critic_loss.detach()),
            }
            self.num_updates += 1
        return metrics

    def _update_imagined_policy(self, batch: SuperMimicSequenceBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimize policy/value on differentiable imagined sequences with frozen model weights."""
        model_requires_grad = [parameter.requires_grad for parameter in self.model.parameters()]
        critic_requires_grad = [parameter.requires_grad for parameter in self.critic.parameters()]
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)

        state = batch.states[:, 0]
        policy_obs = batch.clean_observations[:, 0]
        context, goal = batch.contexts[:, 0], batch.goals[:, 0]
        survival = torch.ones(state.shape[0], 1, device=self.device)
        observations, rewards, terminals = [], [], []
        actor_return = torch.zeros_like(survival)
        discount = 1.0
        for _ in range(self.horizon):
            observations.append(policy_obs)
            action = self.actor(policy_obs, goal if self.actor.goal_dim else None)
            prediction = self.model.predict(state, action, context, goal)
            reward = prediction["reward"]
            terminal = prediction["termination_logit"].sigmoid()
            actor_return = actor_return + discount * survival * reward
            survival = survival * (1.0 - terminal)
            rewards.append(reward)
            terminals.append(terminal)
            state = prediction["next_state"]
            policy_obs = prediction["clean_observation"]
            discount *= self.gamma
        actor_return = actor_return + discount * survival * self.critic(
            torch.cat((policy_obs, goal), dim=-1) if self.actor.goal_dim else policy_obs
        )
        actor_loss = -actor_return.mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            values = torch.stack(
                [self.critic(torch.cat((item, goal), dim=-1) if self.actor.goal_dim else item) for item in observations]
            )
            last_value = self.critic(torch.cat((policy_obs, goal), dim=-1) if self.actor.goal_dim else policy_obs)
            advantages, returns = self.compute_gae(
                torch.stack(rewards), values, last_value, torch.stack(terminals), self.gamma, self.gae_lambda
            )
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        critic_values = torch.stack(
            [self.critic(torch.cat((item.detach(), goal), dim=-1) if self.actor.goal_dim else item.detach()) for item in observations]
        )
        critic_loss = F.mse_loss(critic_values, returns)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()
        for parameter, requires_grad in zip(self.model.parameters(), model_requires_grad):
            parameter.requires_grad_(requires_grad)
        for parameter, requires_grad in zip(self.critic.parameters(), critic_requires_grad):
            parameter.requires_grad_(requires_grad)
        return actor_loss, critic_loss

    @staticmethod
    def compute_gae(rewards, values, next_value, terminated, gamma: float, gae_lambda: float):
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros_like(next_value)
        for step in reversed(range(rewards.shape[0])):
            next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
            continuation = 1.0 - terminated[step].to(dtype=rewards.dtype)
            delta = rewards[step] + gamma * continuation * next_values - values[step]
            last_advantage = delta + gamma * gae_lambda * continuation * last_advantage
            advantages[step] = last_advantage
        return advantages, advantages + values

    def sample_rsi(self, env_ids: torch.Tensor):
        """Sample aligned reset state and bin IDs for the Isaac Lab task."""
        return self.sampler.sample_reference_states(env_ids)

    def get_rsi_joint_layout(self) -> dict:
        dataset = self.sampler.dataset
        from rsl_rl.datasets.rsi_contract import reference_joint_layout

        return reference_joint_layout(dataset)

    def train_mode(self) -> None:
        self.training = True
        self.model.train()
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.training = False
        self.model.eval()
        self.actor.eval()
        self.critic.eval()

    def get_policy(self):
        return _SuperMimicPolicy(self)

    def save(self) -> dict:
        return {
            "model_state_dict": self.model.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "model_optimizer_state_dict": self.model_optimizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "replay_buffer_state_dict": self.replay_buffer.state_dict(),
            "sampler_state_dict": self.sampler.state_dict(),
            "num_updates": self.num_updates,
            "num_steps": self.num_steps,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        load_cfg = load_cfg or {"model": True, "optimizer": True, "replay_buffer": True, "iteration": True}
        if load_cfg.get("model", True):
            for name, module in (("model", self.model), ("actor", self.actor), ("critic", self.critic)):
                module.load_state_dict(loaded_dict[f"{name}_state_dict"], strict=strict)
        if load_cfg.get("optimizer", True):
            for name in ("model", "actor", "critic"):
                getattr(self, f"{name}_optimizer").load_state_dict(loaded_dict[f"{name}_optimizer_state_dict"])
        if load_cfg.get("replay_buffer", True):
            self.replay_buffer.load_state_dict(loaded_dict["replay_buffer_state_dict"])
            for env_id in tuple(self.replay_buffer._current):
                self.replay_buffer.end_episode(env_id)
            self.sampler.load_state_dict(loaded_dict["sampler_state_dict"])
        self.num_updates = int(loaded_dict.get("num_updates", 0))
        self.num_steps = int(loaded_dict.get("num_steps", 0))
        self._transition = None
        self._episode_returns = None
        return bool(load_cfg.get("iteration", False))

    @classmethod
    def construct_algorithm(cls, obs: TensorDict, env: VecEnv, cfg: dict, device: str):
        algorithm_cfg = dict(cfg["algorithm"])
        algorithm_cfg.pop("class_name", None)
        groups = dict(cfg.get("obs_groups", {"actor": ["policy"]}))
        groups.setdefault("actor", ["policy"])
        groups.setdefault("state", groups["actor"])
        groups.setdefault("clean_observation", groups["actor"])
        groups.setdefault("context", [])
        groups.setdefault("goal", [])
        resolved = {key: value for key, value in groups.items() if key in {"actor", "critic"}}
        groups.update(resolve_obs_groups(obs, resolved, ["actor", "critic"]))
        dims = {key: sum(obs[group].shape[-1] for group in groups[key]) for key in ("actor", "state", "clean_observation", "context", "goal")}
        if dims["actor"] != dims["clean_observation"]:
            raise ValueError("SuperMimic clean_observation must have the same dimension as the actor observation")
        if "reference_bin_id" not in obs:
            raise ValueError("SuperMimic observations must include reference_bin_id from the task reset protocol")
        task = getattr(env, "unwrapped", env)
        dataset = getattr(task, "_supermimic_reference_dataset", None)
        if dataset is None:
            raise ValueError("SuperMimic task must expose _supermimic_reference_dataset before algorithm construction")
        sampler = ReturnAwareReferenceSampler(dataset, len(dataset.reference_bins))
        model = SuperMimicWorldModel(
            dims["state"], env.num_actions, dims["clean_observation"], dims["context"], dims["goal"],
            tuple(cfg.get("model", {}).get("hidden_dims", (256, 256, 256))),
        )
        actor = SuperMimicActor(
            dims["actor"], env.num_actions, dims["goal"], tuple(cfg.get("actor", {}).get("hidden_dims", (256, 256)))
        )
        critic = SuperMimicCritic(
            dims["clean_observation"] + dims["goal"], tuple(cfg.get("critic", {}).get("hidden_dims", (256, 256)))
        )
        algorithm_cfg.pop("rnd_cfg", None)
        memory = cfg.get("memory", {})
        batch_size = int(memory.get("batch_size", 256))
        sequence_length = int(memory.get("sequence_length", algorithm_cfg.get("model_sequence_length", 8)))
        algorithm_cfg.pop("model_sequence_length", None)
        replay = SuperMimicReplayBuffer(memory.get("capacity", 1_000_000), sequence_length, batch_size, device)
        return cls(
            model, actor, critic, replay, sampler, groups, batch_size, device,
            **{key: value for key, value in algorithm_cfg.items() if key in {
                "horizon", "gamma", "gae_lambda", "learning_starts", "gradient_steps", "train_every", "lr", "grad_clip_norm"
            }},
        )

    def _extract(self, obs: TensorDict, group: str) -> torch.Tensor:
        groups = self.obs_groups[group]
        if not groups:
            return torch.zeros(obs.batch_size[0], 0, device=self.device)
        return torch.cat([obs[name].to(self.device, dtype=torch.float32) for name in groups], dim=-1)

    @staticmethod
    def _require_bin_ids(obs: TensorDict) -> None:
        if "reference_bin_id" not in obs:
            raise ValueError("SuperMimic task observations must include reference_bin_id")
        bins = obs["reference_bin_id"]
        if bins.ndim not in (1, 2):
            raise ValueError("reference_bin_id must contain one scalar per environment")

    @staticmethod
    def _batch_to_device(batch: SuperMimicSequenceBatch) -> SuperMimicSequenceBatch:
        return SuperMimicSequenceBatch(**{key: value.to(batch.states.device) for key, value in batch.__dict__.items()})


class _SuperMimicPolicy(nn.Module):
    def __init__(self, algorithm: SuperMimic) -> None:
        super().__init__()
        self.algorithm = algorithm

    def forward(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.algorithm._extract(obs, "actor")
        goal = self.algorithm._extract(obs, "goal")
        return self.algorithm.actor(actor_obs, goal if self.algorithm.actor.goal_dim else None)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        return None
