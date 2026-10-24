# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TD-MPC2 for vectorized state-based continuous-control environments."""

from __future__ import annotations

import copy
import math as python_math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import TDMPC2WorldModel
from rsl_rl.models.tdmpc2 import math
from rsl_rl.storage import EpisodeReplayBuffer
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class TDMPC2:
    """Model-based TD-MPC2 with episodic replay and MPPI planning."""

    def __init__(
        self,
        model: TDMPC2WorldModel,
        replay_buffer: EpisodeReplayBuffer,
        obs_groups: dict[str, list[str]],
        batch_size: int,
        episode_length: int,
        device: str = "cpu",
        horizon: int = 3,
        learning_starts: int | None = None,
        gradient_steps: int | None = None,
        reward_coef: float = 0.1,
        value_coef: float = 0.1,
        termination_coef: float = 1.0,
        consistency_coef: float = 20.0,
        rho: float = 0.5,
        lr: float = 3e-4,
        enc_lr_scale: float = 0.3,
        grad_clip_norm: float = 20.0,
        tau: float = 0.01,
        discount_denom: float = 5.0,
        discount_min: float = 0.95,
        discount_max: float = 0.995,
        mpc: bool = True,
        iterations: int = 6,
        num_samples: int = 512,
        num_elites: int = 64,
        num_pi_trajs: int = 24,
        min_std: float = 0.05,
        max_std: float = 2.0,
        temperature: float = 0.5,
        entropy_coef: float = 1e-4,
        episodic: bool = False,
        train_every: int | None = None,
        utd: float | None = None,
        warmup_steps: int | None = None,
        num_envs: int = 1,
    ) -> None:
        legacy_learning_starts = 0 if learning_starts is None and warmup_steps is None else learning_starts
        legacy_gradient_steps = 1 if gradient_steps is None else gradient_steps
        legacy_train_every = 1 if train_every is None else train_every
        if horizon <= 0 or batch_size <= 0 or num_envs <= 0:
            raise ValueError("TD-MPC2 horizon, batch size, and num_envs must be positive.")
        if legacy_gradient_steps <= 0 or legacy_train_every <= 0:
            raise ValueError("TD-MPC2 legacy update intervals must be positive.")
        utd = legacy_gradient_steps / legacy_train_every if utd is None else float(utd)
        if not python_math.isfinite(utd) or utd <= 0:
            raise ValueError("TD-MPC2 utd must be finite and positive.")
        if warmup_steps is None:
            warmup_steps = python_math.ceil((legacy_learning_starts or 0) / num_envs)
        if warmup_steps < 0:
            raise ValueError("TD-MPC2 warmup_steps cannot be negative.")
        if episode_length <= 0:
            raise ValueError("TD-MPC2 requires a positive vector environment episode length.")
        if legacy_learning_starts is not None and legacy_learning_starts < 0:
            raise ValueError("TD-MPC2 learning_starts cannot be negative.")
        if iterations <= 0 or num_samples <= 0 or not 1 <= num_elites <= num_samples:
            raise ValueError("TD-MPC2 planner iterations and sample counts must be positive, with num_elites <= num_samples.")
        if num_pi_trajs < 0 or num_pi_trajs >= num_samples:
            raise ValueError("TD-MPC2 num_pi_trajs must be non-negative and less than num_samples.")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.replay_buffer = replay_buffer
        self.obs_groups = obs_groups
        self.batch_size = batch_size
        self.episode_length = int(episode_length)
        self.horizon = horizon
        self.num_envs = int(num_envs)
        self.utd = float(utd)
        self.warmup_steps = int(warmup_steps)
        self.learning_starts = self.warmup_steps * self.num_envs
        self.reward_coef = reward_coef
        self.value_coef = value_coef
        self.termination_coef = termination_coef
        self.consistency_coef = consistency_coef
        self.rho = rho
        self.lr = lr
        self.grad_clip_norm = grad_clip_norm
        self.tau = tau
        self.episodic = episodic
        self.mpc = mpc
        self.iterations = iterations
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_pi_trajs = num_pi_trajs
        self.min_std = min_std
        self.max_std = max_std
        self.temperature = temperature
        self.entropy_coef = entropy_coef
        self.discount = min(max(1.0 - discount_denom / self.episode_length, discount_min), discount_max)
        self._scale = torch.ones((), device=self.device)
        trainable_model_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and not name.startswith("policy_head.")
        ]
        self.model_optimizer_parameters = trainable_model_parameters
        encoder_parameters = list(self.model.encoder.parameters())
        encoder_ids = {id(parameter) for parameter in encoder_parameters}
        other_parameters = [parameter for parameter in trainable_model_parameters if id(parameter) not in encoder_ids]
        self.optimizer = torch.optim.Adam(
            [
                {"params": encoder_parameters, "lr": lr * enc_lr_scale},
                {"params": other_parameters, "lr": lr},
            ]
        )
        self.policy_optimizer = torch.optim.Adam(self.model.policy_head.parameters(), lr=lr, eps=1e-5)
        self.num_updates = 0
        self.num_steps = 0
        self.training = True
        self._transition_observations: torch.Tensor | None = None
        self._transition_actions: torch.Tensor | None = None
        self._episode_start: torch.Tensor | None = None
        self._previous_mean: torch.Tensor | None = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Choose actions from warmup sampling, MPPI planning, or the learned prior."""
        observations = self._extract_observations(obs)
        self._transition_observations = observations.detach()
        if self._episode_start is None or self._episode_start.shape[0] != observations.shape[0]:
            self._episode_start = torch.ones(observations.shape[0], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            if self.training and self.num_steps < self.learning_starts:
                actions = self._random_action(observations.shape[0])
            elif self.mpc:
                actions = self._plan(observations, self._episode_start, eval_mode=not self.training)
            else:
                actions, _ = self.model.pi(self.model.encode(observations), deterministic=not self.training)
        self._transition_actions = actions.detach()
        self._episode_start.zero_()
        return actions

    @property
    def warmup_steps_collected(self) -> int:
        """Return replayed vector steps available toward the configured warmup."""
        return min(self.replay_buffer.size // self.num_envs, self.warmup_steps)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Append per-environment transitions and close terminal or timeout episodes."""
        if self._transition_observations is None or self._transition_actions is None:
            raise RuntimeError("TDMPC2.process_env_step() called before act().")
        next_observations = self._extract_observations(obs)
        dones = dones.to(self.device).bool().reshape(-1)
        final_observations = extras.get("final_observation", extras.get("terminal_observation"))
        if final_observations is not None:
            if not isinstance(final_observations, TensorDict):
                raise TypeError("final_observation must be a TensorDict with the same observation groups as env observations.")
            next_observations = next_observations.clone()
            next_observations[dones] = self._extract_observations(final_observations)[dones]
        timeouts = extras.get("time_outs", torch.zeros_like(dones)).to(self.device).bool().reshape(-1)
        terminated = extras.get("terminated", dones & ~timeouts).to(self.device).bool().reshape(-1)
        if bool(terminated.any()) and not self.episodic:
            raise ValueError("TD-MPC2 received true termination flags while episodic mode is disabled.")
        for env_id in range(dones.shape[0]):
            self.replay_buffer.add_step(
                env_id,
                self._transition_observations[env_id],
                self._transition_actions[env_id],
                rewards[env_id],
                next_observations[env_id],
                terminated[env_id],
            )
            if bool(dones[env_id]) and not bool(terminated[env_id]):
                self.replay_buffer.end_episode(env_id)
        self._episode_start = dones
        if self._previous_mean is not None:
            self._previous_mean[dones] = 0.0
        self.num_steps += dones.numel()
        self._transition_observations = None
        self._transition_actions = None

    def update(self, iteration: int | None = None) -> dict[str, float]:
        """Update the world model and policy from sampled episode sequences."""
        try:
            batch = self.replay_buffer.sample()
        except RuntimeError as error:
            if "long enough" in str(error):
                return {"total_loss": 0.0, "pi_loss": 0.0}
            raise
        return self._update_batch(batch.observations, batch.actions, batch.rewards, batch.terminated)

    def _update_batch(
        self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, terminated: torch.Tensor
    ) -> dict[str, float]:
        observations = observations.transpose(0, 1).to(self.device)
        actions = actions.transpose(0, 1).to(self.device)
        rewards = rewards.transpose(0, 1).to(self.device)
        terminated = terminated.transpose(0, 1).to(self.device)
        with torch.no_grad():
            target_latents = self.model.encode(observations[1:])
            next_actions, _ = self.model.pi(target_latents)
            target_values = self.model.Q(target_latents, next_actions, return_type="min", target=True)
            td_targets = rewards + self.discount * (1.0 - terminated) * target_values

        self.model.train()
        latent = self.model.encode(observations[0])
        latents = [latent]
        consistency_loss = torch.zeros((), device=self.device)
        for step in range(self.horizon):
            latent = self.model.next(latent, actions[step])
            consistency_loss = consistency_loss + F.mse_loss(latent, target_latents[step]) * self.rho**step
            latents.append(latent)
        rollout_latents = torch.stack(latents)
        prediction_latents = rollout_latents[:-1]
        q_logits = self.model.Q(prediction_latents, actions, return_type="all")
        reward_logits = self.model.reward(prediction_latents, actions)
        reward_loss = torch.zeros((), device=self.device)
        value_loss = torch.zeros((), device=self.device)
        for step in range(self.horizon):
            weight = self.rho**step
            reward_loss = reward_loss + math.soft_ce(reward_logits[step], rewards[step], self.model.cfg).mean() * weight
            for q_index in range(q_logits.shape[0]):
                value_loss = value_loss + math.soft_ce(q_logits[q_index, step], td_targets[step], self.model.cfg).mean() * weight
        consistency_loss = consistency_loss / self.horizon
        reward_loss = reward_loss / self.horizon
        value_loss = value_loss / (self.horizon * self.model.cfg["num_q"])
        termination_loss = torch.zeros((), device=self.device)
        if self.episodic:
            termination_logits = self.model.termination(rollout_latents[1:], unnormalized=True)
            termination_loss = F.binary_cross_entropy_with_logits(termination_logits, terminated)
        total_loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef * reward_loss
            + self.value_coef * value_loss
            + self.termination_coef * termination_loss
        )
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model_optimizer_parameters, self.grad_clip_norm)
        self.optimizer.step()

        critic_parameters = list(self.model.q_networks.parameters())
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        policy_actions, policy_info = self.model.pi(rollout_latents[:-1].detach())
        policy_values = self.model.Q(rollout_latents[:-1].detach(), policy_actions, return_type="avg")
        with torch.no_grad():
            quantiles = torch.quantile(policy_values.detach().flatten(), torch.tensor([0.05, 0.95], device=self.device))
            scale_value = (quantiles[1] - quantiles[0]).clamp(min=1.0)
            self._scale.lerp_(scale_value, self.tau)
        weights = torch.pow(self.rho, torch.arange(self.horizon, device=self.device)).view(self.horizon, 1, 1)
        policy_entropy = -policy_info["log_prob"]
        policy_loss = -((self.entropy_coef * policy_entropy + policy_values / self._scale) * weights).mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy_head.parameters(), self.grad_clip_norm)
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad(set_to_none=True)
        for parameter in critic_parameters:
            parameter.requires_grad_(True)
        self.model.soft_update_target_q()
        self.model.eval()
        self.num_updates += 1
        return {
            "consistency_loss": float(consistency_loss.detach()),
            "reward_loss": float(reward_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "termination_loss": float(termination_loss.detach()),
            "total_loss": float(total_loss.detach()),
            "grad_norm": float(grad_norm.detach()),
            "pi_loss": float(policy_loss.detach()),
            "pi_entropy": float(policy_entropy.detach().mean()),
            "pi_scale": float(self._scale.detach()),
        }

    def train_mode(self) -> None:
        """Enable training behavior for the world model and planner."""
        self.training = True
        self.model.train()

    def eval_mode(self) -> None:
        """Set the world model to evaluation mode without adding exploration noise."""
        self.training = False
        self.model.eval()

    def save(self) -> dict:
        """Return model, optimizer, replay, and planner state for checkpointing."""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "replay_buffer_state_dict": self.replay_buffer.state_dict(),
            "scale": self._scale.detach().clone(),
            "previous_mean": None if self._previous_mean is None else self._previous_mean.detach().clone(),
            "episode_start": None if self._episode_start is None else self._episode_start.detach().clone(),
            "num_updates": self.num_updates,
            "num_steps": self.num_steps,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Restore selected algorithm state from an `rsl_rl` checkpoint."""
        load_cfg = load_cfg or {"model": True, "optimizer": True, "replay_buffer": True, "iteration": True}
        if load_cfg.get("model", True):
            self.model.load_state_dict(loaded_dict["model_state_dict"], strict=strict)
        if load_cfg.get("optimizer", True):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.policy_optimizer.load_state_dict(loaded_dict["policy_optimizer_state_dict"])
        if load_cfg.get("replay_buffer", True) and "replay_buffer_state_dict" in loaded_dict:
            self.replay_buffer.load_state_dict(loaded_dict["replay_buffer_state_dict"])
            # Native runner checkpoints do not include simulator state. Close partial
            # trajectories at the resume boundary before fresh environment steps arrive.
            for env_id in tuple(self.replay_buffer._current):
                self.replay_buffer.end_episode(env_id)
        if "scale" in loaded_dict:
            self._scale.copy_(loaded_dict["scale"].to(self.device))
        self._previous_mean = None
        self._episode_start = None
        self.num_updates = int(loaded_dict.get("num_updates", 0))
        self.num_steps = int(loaded_dict.get("num_steps", 0))
        return bool(load_cfg.get("iteration", False))

    def get_policy(self) -> TDMPC2Policy:
        """Return the evaluation policy wrapper with planner reset support."""
        return TDMPC2Policy(self)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset planner state for all environments or the indicated done mask."""
        if self._episode_start is None:
            if dones is not None:
                self._episode_start = torch.ones_like(dones, dtype=torch.bool, device=self.device)
            return
        if dones is None:
            self._episode_start.fill_(True)
        else:
            dones = dones.to(self.device).bool().reshape(-1)
            self._episode_start |= dones
            if self._previous_mean is not None:
                self._previous_mean[dones] = 0.0

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> TDMPC2:
        """Construct the state world model, episode replay, and algorithm from native config dictionaries."""
        algorithm_cfg = cfg["algorithm"]
        algorithm_class = resolve_callable(algorithm_cfg.pop("class_name", "TDMPC2"))
        algorithm_options = {
            key: value for key, value in algorithm_cfg.items() if key not in {"rnd_cfg", "num_bins", "vmin", "vmax"}
        }
        model_cfg = cfg["model"]
        model_class = resolve_callable(model_cfg.pop("class_name", "TDMPC2WorldModel"))
        obs_groups = resolve_obs_groups(obs, cfg.get("obs_groups", {}), ["actor"])
        groups = obs_groups["actor"]
        for group in groups:
            if obs[group].ndim != 2:
                raise ValueError(f"TD-MPC2 only supports 1D state observation groups; '{group}' has shape {obs[group].shape}.")
        obs_dim = sum(obs[group].shape[-1] for group in groups)
        episode_length = env.max_episode_length
        if isinstance(episode_length, torch.Tensor):
            episode_length = int(episode_length.max().item())
        episode_length = int(episode_length)
        if episode_length <= 0:
            raise ValueError("TD-MPC2 requires env.max_episode_length to be positive.")
        world_cfg = dict(model_cfg)
        world_cfg.pop("obs", None)
        world_cfg.update(
            {
                "num_bins": model_cfg.get("num_bins", algorithm_cfg.get("num_bins", 101)),
                "vmin": model_cfg.get("vmin", algorithm_cfg.get("vmin", -10.0)),
                "vmax": model_cfg.get("vmax", algorithm_cfg.get("vmax", 10.0)),
                "episodic": algorithm_cfg.get("episodic", False),
                "tau": algorithm_cfg.get("tau", 0.01),
            }
        )
        model = model_class(obs_dim=obs_dim, action_dim=env.num_actions, cfg=world_cfg)
        memory_cfg = cfg.get("memory", {})
        horizon = int(algorithm_cfg.get("horizon", 3))
        batch_size = int(memory_cfg.get("batch_size", 256))
        replay_buffer = EpisodeReplayBuffer(
            capacity=memory_cfg.get("capacity", 1_000_000),
            sequence_length=horizon,
            batch_size=batch_size,
            device=device,
        )
        return algorithm_class(
            model=model,
            replay_buffer=replay_buffer,
            obs_groups=obs_groups,
            batch_size=batch_size,
            episode_length=episode_length,
            device=device,
            num_envs=env.num_envs,
            **algorithm_options,
        )

    def _extract_observations(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group].to(self.device, dtype=torch.float32) for group in self.obs_groups["actor"]], dim=-1)

    def _random_action(self, num_envs: int) -> torch.Tensor:
        return torch.rand(num_envs, self.model.action_dim, device=self.device).mul_(2.0).sub_(1.0)

    def _plan(self, observations: torch.Tensor, t0: torch.Tensor, eval_mode: bool) -> torch.Tensor:
        batch_size = observations.shape[0]
        latent = self.model.encode(observations)
        if self._previous_mean is None or self._previous_mean.shape != (batch_size, self.horizon, self.model.action_dim):
            self._previous_mean = torch.zeros(batch_size, self.horizon, self.model.action_dim, device=self.device)
        mean = torch.zeros_like(self._previous_mean)
        mean[:, :-1] = self._previous_mean[:, 1:]
        mean[t0] = 0.0
        std = torch.full_like(mean, self.max_std)
        samples = self.num_samples
        pi_count = min(self.num_pi_trajs, samples - 1)
        final_elite_actions = None
        for _ in range(self.iterations):
            random_count = samples - pi_count
            random_actions = mean[:, None] + std[:, None] * torch.randn(
                batch_size, random_count, self.horizon, self.model.action_dim, device=self.device
            )
            actions = random_actions.clamp(-1.0, 1.0)
            if pi_count:
                pi_latent = latent[:, None].expand(-1, pi_count, -1).reshape(batch_size * pi_count, -1)
                pi_actions = []
                for _step in range(self.horizon):
                    pi_action, _ = self.model.pi(pi_latent)
                    pi_actions.append(pi_action.reshape(batch_size, pi_count, self.model.action_dim))
                    pi_latent = self.model.next(pi_latent, pi_action)
                actions = torch.cat((torch.stack(pi_actions, dim=2), actions), dim=1)
            values = self._estimate_value(latent, actions)
            elite_values, elite_indices = values.topk(self.num_elites, dim=1)
            final_elite_actions = torch.gather(
                actions,
                1,
                elite_indices[:, :, None, None].expand(-1, -1, self.horizon, self.model.action_dim),
            )
            weights = torch.softmax(self.temperature * (elite_values - elite_values.max(dim=1, keepdim=True).values), dim=1)
            mean = (weights[:, :, None, None] * final_elite_actions).sum(dim=1)
            variance = (weights[:, :, None, None] * (final_elite_actions - mean[:, None]).square()).sum(dim=1)
            std = variance.sqrt().clamp(self.min_std, self.max_std)
        self._previous_mean.copy_(mean)
        elite_indices = torch.multinomial(weights, num_samples=1).squeeze(-1)
        chosen_trajectories = torch.gather(
            final_elite_actions,
            1,
            elite_indices[:, None, None, None].expand(-1, 1, self.horizon, self.model.action_dim),
        ).squeeze(1)
        action = chosen_trajectories[:, 0]
        if not eval_mode:
            action = action + std[:, 0] * torch.randn_like(action)
        return action.clamp(-1.0, 1.0)

    def _estimate_value(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size, num_samples = actions.shape[:2]
        current_latent = latent[:, None].expand(-1, num_samples, -1).reshape(batch_size * num_samples, -1)
        returns = torch.zeros(batch_size * num_samples, 1, device=self.device)
        discount = 1.0
        alive = torch.ones_like(returns)
        for step in range(self.horizon):
            action = actions[:, :, step].reshape(batch_size * num_samples, -1)
            reward = math.two_hot_inv(self.model.reward(current_latent, action), self.model.cfg)
            current_latent = self.model.next(current_latent, action)
            returns = returns + discount * alive * reward
            discount *= self.discount
            if self.episodic:
                alive = alive * (self.model.termination(current_latent) <= 0.5).to(alive.dtype)
        final_action, _ = self.model.pi(current_latent)
        final_value = self.model.Q(current_latent, final_action, return_type="avg")
        returns = returns + discount * alive * final_value
        return returns.reshape(batch_size, num_samples)


class TDMPC2Policy(torch.nn.Module):
    """TensorDict-compatible inference policy retaining MPPI planner state."""

    def __init__(self, algorithm: TDMPC2) -> None:
        super().__init__()
        self.algorithm = algorithm
        self.model = algorithm.model

    def to(self, *args, **kwargs):
        """Move inference model and planner allocations together."""
        super().to(*args, **kwargs)
        self.algorithm.device = next(self.model.parameters()).device
        self.algorithm._scale = self.algorithm._scale.to(self.algorithm.device)
        self.algorithm._episode_start = None
        self.algorithm._previous_mean = None
        return self

    def forward(self, obs: TensorDict) -> torch.Tensor:
        """Plan on configured state observation groups without added exploration noise."""
        return self.algorithm.act(obs)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset planner warm-start state after episode termination."""
        self.algorithm.reset(dones)
