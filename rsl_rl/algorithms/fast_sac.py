# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fast Soft Actor-Critic for vectorized continuous-control environments."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import DistributionalCritic, FastSACActor, FastSACPolicy
from rsl_rl.storage import OffPolicyReplayBuffer
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class FastSAC:
    """Off-policy FastSAC algorithm with distributional critics."""

    def __init__(
        self,
        actor: FastSACActor,
        critic: DistributionalCritic,
        replay_buffer: OffPolicyReplayBuffer,
        obs_groups: dict[str, list[str]],
        all_obs_groups: Sequence[str],
        obs_group_dims: dict[str, int],
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.2,
        target_entropy_ratio: float = 1.0,
        auto_entropy: bool = True,
        learning_starts: int = 5_000,
        train_every: int = 1,
        gradient_steps: int = 1,
        device: str = "cpu",
    ) -> None:
        if batch_size <= 0 or learning_starts < 0 or train_every <= 0 or gradient_steps <= 0:
            raise ValueError("FastSAC batch size and update intervals must be positive; learning_starts cannot be negative.")
        if not 0.0 <= gamma <= 1.0 or not 0.0 < tau <= 1.0:
            raise ValueError("FastSAC requires gamma in [0, 1] and tau in (0, 1].")
        if init_alpha <= 0.0:
            raise ValueError("FastSAC init_alpha must be positive.")
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.target_critic = copy.deepcopy(self.critic).eval()
        self._raw_actor = self.actor
        self._raw_critic = self.critic
        self.replay_buffer = replay_buffer
        self.obs_groups = obs_groups
        self.all_obs_groups = tuple(all_obs_groups)
        self._group_dims = obs_group_dims
        self._group_slices: dict[str, slice] = {}
        start = 0
        for group in self.all_obs_groups:
            group_dim = self._group_dims[group]
            self._group_slices[group] = slice(start, start + group_dim)
            start += group_dim

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learning_starts = learning_starts
        self.train_every = train_every
        self.gradient_steps = gradient_steps
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.auto_entropy = auto_entropy
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(math.log(init_alpha), device=self.device), requires_grad=auto_entropy
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr) if auto_entropy else None
        self.target_entropy = -float(self.actor.action_dim) * target_entropy_ratio
        self.num_updates = 0
        self.num_steps = 0
        self.training = True
        self._transition_observations: torch.Tensor | None = None
        self._transition_actions: torch.Tensor | None = None

    @property
    def alpha(self) -> torch.Tensor:
        """Return the current entropy temperature."""
        return self.log_alpha.exp()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Choose actions and retain the current observations for replay insertion."""
        packed = self._pack_observations(obs)
        actor_obs = self._select_groups(packed, self.obs_groups["actor"])
        self._transition_observations = packed.detach()
        with torch.no_grad():
            if self.training and self.num_steps < self.learning_starts:
                actions = torch.rand(packed.shape[0], self.actor.action_dim, device=self.device).mul_(2.0).sub_(1.0)
            elif self.training:
                actions, _ = self.actor.sample(actor_obs)
            else:
                actions = self.actor.deterministic(actor_obs)
        self._transition_actions = actions.detach()
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Append the completed transition, treating timeouts as bootstrap-able."""
        if self._transition_observations is None or self._transition_actions is None:
            raise RuntimeError("FastSAC.process_env_step() called before act().")
        timeouts = extras.get("time_outs", torch.zeros_like(dones)).to(self.device).bool().reshape(-1, 1)
        terminals = dones.to(self.device).bool().reshape(-1, 1) & ~timeouts
        next_observations = self._pack_observations(obs)
        final_observations = extras.get("final_observation", extras.get("terminal_observation"))
        if final_observations is not None:
            if not isinstance(final_observations, TensorDict):
                raise TypeError("final_observation must be a TensorDict with the same observation groups as env observations.")
            done_mask = dones.to(self.device).bool().reshape(-1)
            next_observations = next_observations.clone()
            next_observations[done_mask] = self._pack_observations(final_observations)[done_mask]
        self.replay_buffer.add(
            self._transition_observations,
            self._transition_actions,
            rewards.to(self.device).reshape(-1, 1),
            next_observations,
            terminals.float(),
        )
        self._transition_observations = None
        self._transition_actions = None
        self.num_steps += rewards.shape[0]

    def update(self) -> dict[str, float]:
        """Perform configured critic, actor, temperature, and target updates."""
        if self.replay_buffer.size < self.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "alpha": float(self.alpha.detach())}
        metric_sums = {"actor_loss": 0.0, "critic_loss": 0.0, "alpha_loss": 0.0}
        for _ in range(self.gradient_steps):
            batch = self.replay_buffer.sample(self.batch_size)
            actor_obs = self._select_groups(batch.observations, self.obs_groups["actor"])
            critic_obs = self._select_groups(batch.observations, self.obs_groups["critic"])
            next_actor_obs = self._select_groups(batch.next_observations, self.obs_groups["actor"])
            next_critic_obs = self._select_groups(batch.next_observations, self.obs_groups["critic"])
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.sample(next_actor_obs)
                target_logits = self.target_critic(next_critic_obs, next_actions)
                target_values = self.target_critic.values(target_logits)
                selected_q = target_values.argmin(dim=0)
                q_indices = torch.arange(self.batch_size, device=self.device)
                target_probabilities = target_logits.softmax(dim=-1)[selected_q, q_indices]
                target_distribution = self.critic.project(
                    target_probabilities,
                    batch.rewards - self.gamma * (1.0 - batch.dones) * self.alpha.detach() * next_log_prob,
                    1.0 - batch.dones,
                    self.gamma,
                )

            logits = self.critic(critic_obs, batch.actions)
            critic_loss = -(
                target_distribution.unsqueeze(0) * F.log_softmax(logits, dim=-1)
            ).sum(dim=-1).mean(dim=1).sum(dim=0)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()

            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            actions, log_prob = self.actor.sample(actor_obs)
            actor_values = self.critic.values(self.critic(critic_obs, actions)).min(dim=0).values
            actor_loss = (self.alpha.detach() * log_prob - actor_values.unsqueeze(-1)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)

            alpha_loss = torch.zeros((), device=self.device)
            if self.auto_entropy:
                alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
                alpha_loss.backward()
                self.alpha_optimizer.step()  # type: ignore[union-attr]

            with torch.no_grad():
                for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                    target.lerp_(source, self.tau)
            metric_sums["actor_loss"] += float(actor_loss.detach())
            metric_sums["critic_loss"] += float(critic_loss.detach())
            metric_sums["alpha_loss"] += float(alpha_loss.detach())
            self.num_updates += 1
        return {key: value / self.gradient_steps for key, value in metric_sums.items()} | {
            "alpha": float(self.alpha.detach())
        }

    def train_mode(self) -> None:
        """Enable training behavior for the actor and critics."""
        self.training = True
        self.actor.train()
        self.critic.train()
        self.target_critic.eval()

    def eval_mode(self) -> None:
        """Use deterministic actions during evaluation."""
        self.training = False
        self.actor.eval()
        self.critic.eval()
        self.target_critic.eval()

    def save(self) -> dict:
        """Return model, optimizer, entropy, and replay states for checkpointing."""
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            "replay_buffer_state_dict": self.replay_buffer.state_dict(),
            "num_updates": self.num_updates,
            "num_steps": self.num_steps,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """Restore selected algorithm state from a runner checkpoint."""
        load_cfg = load_cfg or {
            "actor": True,
            "critic": True,
            "optimizer": True,
            "replay_buffer": True,
            "iteration": True,
        }
        if load_cfg.get("actor", True):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic", True):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
            self.target_critic.load_state_dict(loaded_dict["target_critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer", True):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])
            if self.alpha_optimizer and loaded_dict.get("alpha_optimizer_state_dict") is not None:
                self.alpha_optimizer.load_state_dict(loaded_dict["alpha_optimizer_state_dict"])
        if "log_alpha" in loaded_dict:
            self.log_alpha.data.copy_(loaded_dict["log_alpha"].to(self.device))
        if load_cfg.get("replay_buffer", True) and "replay_buffer_state_dict" in loaded_dict:
            self.replay_buffer.load_state_dict(loaded_dict["replay_buffer_state_dict"])
        self.num_updates = int(loaded_dict.get("num_updates", 0))
        self.num_steps = int(loaded_dict.get("num_steps", 0))
        return bool(load_cfg.get("iteration", False))

    def get_policy(self) -> FastSACPolicy:
        """Return a deterministic TensorDict-compatible inference policy."""
        return FastSACPolicy(self._raw_actor, self.obs_groups["actor"])

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> FastSAC:
        """Construct FastSAC using the standard `rsl_rl` training dictionary."""
        alg_cfg = cfg["algorithm"]
        alg_class = resolve_callable(alg_cfg.pop("class_name", "FastSAC"))
        alg_options = {key: value for key, value in alg_cfg.items() if key != "rnd_cfg"}
        actor_cfg = cfg["actor"]
        actor_class = resolve_callable(actor_cfg.pop("class_name", "FastSACActor"))
        critic_cfg = cfg["critic"]
        critic_class = resolve_callable(critic_cfg.pop("class_name", "DistributionalCritic"))
        memory_cfg = cfg.get("memory", {})
        obs_groups = resolve_obs_groups(obs, cfg.get("obs_groups", {}), ["actor", "critic"])
        all_groups = list(dict.fromkeys(obs_groups["actor"] + obs_groups["critic"]))
        group_dims: dict[str, int] = {}
        for group in all_groups:
            value = obs[group]
            if value.ndim != 2:
                raise ValueError(f"FastSAC only supports 1D observation groups; '{group}' has shape {tuple(value.shape)}.")
            group_dims[group] = value.shape[-1]
        actor_dim = sum(group_dims[group] for group in obs_groups["actor"])
        critic_dim = sum(group_dims[group] for group in obs_groups["critic"])
        actor = actor_class(obs_dim=actor_dim, action_dim=env.num_actions, **actor_cfg)
        critic = critic_class(obs_dim=critic_dim, action_dim=env.num_actions, **critic_cfg)
        replay_buffer = OffPolicyReplayBuffer(
            capacity=memory_cfg.get("capacity", 1_000_000),
            obs_dim=sum(group_dims.values()),
            action_dim=env.num_actions,
            device=device,
        )
        instance = alg_class(
            actor=actor,
            critic=critic,
            replay_buffer=replay_buffer,
            obs_groups=obs_groups,
            all_obs_groups=all_groups,
            obs_group_dims=group_dims,
            batch_size=memory_cfg.get("batch_size", 256),
            device=device,
            **alg_options,
        )
        return instance

    def _pack_observations(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group].to(self.device, dtype=torch.float32) for group in self.all_obs_groups], dim=-1)

    def _select_groups(self, observations: torch.Tensor, groups: Sequence[str]) -> torch.Tensor:
        return torch.cat([observations[..., self._group_slices[group]] for group in groups], dim=-1)
