# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversarial Motion Prior (AMP) on top of PPO."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from .ppo import PPO


class AmpReplayBuffer:
    """Fixed-size replay buffer for policy AMP observations."""

    def __init__(self, capacity: int, observation_dim: int, device: str):
        if capacity <= 0:
            raise ValueError("AMP replay capacity must be positive")
        self.data = torch.empty(capacity, observation_dim, device=device)
        self.capacity = capacity
        self.size = 0
        self.position = 0

    def add(self, observations: torch.Tensor) -> None:
        observations = observations.detach()
        if observations.shape[0] >= self.capacity:
            self.data.copy_(observations[-self.capacity :])
            self.size = self.capacity
            self.position = 0
            return
        indexes = (torch.arange(observations.shape[0], device=observations.device) + self.position) % self.capacity
        self.data[indexes] = observations
        self.position = (self.position + observations.shape[0]) % self.capacity
        self.size = min(self.capacity, self.size + observations.shape[0])

    def sample(self, batch_size: int) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty AMP replay buffer")
        indexes = torch.randint(self.size, (batch_size,), device=self.data.device)
        return self.data[indexes]


class AMP(PPO):
    """PPO with an AMP discriminator and style-reward mixing."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        amp_observation_groups: list[str],
        amp_observation_dim: int,
        collect_reference_motions: Callable[[int], torch.Tensor],
        discriminator_hidden_dims: tuple[int, ...] | list[int] = (1024, 512),
        discriminator_activation: str = "relu",
        discriminator_learning_rate: float = 5.0e-4,
        discriminator_batch_size: int = 4096,
        discriminator_updates: int = 4,
        discriminator_loss_scale: float = 5.0,
        discriminator_logit_regularization_scale: float = 0.05,
        discriminator_gradient_penalty_scale: float = 5.0,
        discriminator_weight_decay_scale: float = 1.0e-4,
        amp_replay_buffer_size: int = 200_000,
        task_reward_scale: float = 0.0,
        style_reward_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(actor, critic, storage, **kwargs)
        self.amp_observation_groups = amp_observation_groups
        self.collect_reference_motions = collect_reference_motions
        self.discriminator_batch_size = discriminator_batch_size
        self.discriminator_updates = discriminator_updates
        self.discriminator_loss_scale = discriminator_loss_scale
        self.discriminator_logit_regularization_scale = discriminator_logit_regularization_scale
        self.discriminator_gradient_penalty_scale = discriminator_gradient_penalty_scale
        self.discriminator_weight_decay_scale = discriminator_weight_decay_scale
        self.task_reward_scale = task_reward_scale
        self.style_reward_scale = style_reward_scale

        self.discriminator = MLP(
            amp_observation_dim,
            1,
            discriminator_hidden_dims,
            discriminator_activation,
        ).to(self.device)
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_learning_rate,
        )
        self.amp_normalizer = EmpiricalNormalization(amp_observation_dim).to(self.device)
        self.amp_replay_buffer = AmpReplayBuffer(
            amp_replay_buffer_size,
            amp_observation_dim,
            self.device,
        )
        self._rollout_amp_observations: list[torch.Tensor] = []
        self.style_rewards = torch.zeros(1, device=self.device)

    def _get_amp_observations(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.amp_observation_groups], dim=-1)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        amp_observations = self._get_amp_observations(obs).detach()
        self._rollout_amp_observations.append(amp_observations)
        with torch.no_grad():
            logits = self.discriminator(self.amp_normalizer(amp_observations))
            self.style_rewards = F.softplus(logits).squeeze(-1)
            combined_rewards = (
                self.task_reward_scale * rewards
                + self.style_reward_scale * self.style_rewards
            )
        extras["amp_style_reward"] = self.style_rewards.mean()
        extras["amp_task_reward"] = rewards.mean()
        super().process_env_step(obs, combined_rewards, dones, extras)

    def update(self) -> dict[str, float]:
        if not self._rollout_amp_observations:
            raise RuntimeError("AMP update requires at least one rollout observation")
        online = torch.cat(self._rollout_amp_observations, dim=0)
        self._rollout_amp_observations.clear()

        ppo_losses = super().update()
        discriminator_losses = self._update_discriminator(online)
        self.amp_replay_buffer.add(online)
        ppo_losses.update(discriminator_losses)
        return ppo_losses

    def _update_discriminator(self, online: torch.Tensor) -> dict[str, float]:
        mean_prediction_loss = 0.0
        mean_gradient_penalty = 0.0
        mean_total_loss = 0.0

        for _ in range(self.discriminator_updates):
            policy_indexes = torch.randint(
                online.shape[0],
                (self.discriminator_batch_size,),
                device=online.device,
            )
            policy_samples = online[policy_indexes]
            replay_samples = (
                self.amp_replay_buffer.sample(self.discriminator_batch_size)
                if self.amp_replay_buffer.size > 0
                else policy_samples
            )
            expert_samples = self.collect_reference_motions(self.discriminator_batch_size).to(self.device)

            with torch.no_grad():
                self.amp_normalizer.update(
                    torch.cat((policy_samples, replay_samples, expert_samples), dim=0)
                )
            policy_samples = self.amp_normalizer(policy_samples)
            replay_samples = self.amp_normalizer(replay_samples)
            expert_samples = self.amp_normalizer(expert_samples).requires_grad_(True)

            policy_logits = self.discriminator(torch.cat((policy_samples, replay_samples), dim=0))
            expert_logits = self.discriminator(expert_samples)
            prediction_loss = 0.5 * (
                F.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))
                + F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))
            )
            loss = prediction_loss

            last_linear = [module for module in self.discriminator.modules() if isinstance(module, nn.Linear)][-1]
            loss = loss + self.discriminator_logit_regularization_scale * last_linear.weight.square().sum()

            expert_gradient = torch.autograd.grad(
                expert_logits,
                expert_samples,
                grad_outputs=torch.ones_like(expert_logits),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradient_penalty = expert_gradient.square().sum(dim=-1).mean()
            loss = loss + self.discriminator_gradient_penalty_scale * gradient_penalty

            all_weights = torch.cat(
                [
                    module.weight.flatten()
                    for module in self.discriminator.modules()
                    if isinstance(module, nn.Linear)
                ]
            )
            loss = loss + self.discriminator_weight_decay_scale * all_weights.square().sum()
            loss = self.discriminator_loss_scale * loss

            self.discriminator_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.discriminator_optimizer.step()

            mean_prediction_loss += prediction_loss.item()
            mean_gradient_penalty += gradient_penalty.item()
            mean_total_loss += loss.item()

        divisor = float(self.discriminator_updates)
        return {
            "amp_discriminator": mean_total_loss / divisor,
            "amp_prediction": mean_prediction_loss / divisor,
            "amp_gradient_penalty": mean_gradient_penalty / divisor,
            "amp_style_reward": self.style_rewards.mean().item(),
        }

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()
        self.amp_normalizer.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()
        self.amp_normalizer.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict.update(
            {
                "amp_discriminator_state_dict": self.discriminator.state_dict(),
                "amp_discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
                "amp_normalizer_state_dict": self.amp_normalizer.state_dict(),
            }
        )
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("amp", True):
            self.discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
            self.discriminator_optimizer.load_state_dict(loaded_dict["amp_discriminator_optimizer_state_dict"])
            self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"], strict=strict)
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "AMP":
        """Construct AMP and connect it to the environment expert dataset."""
        alg_class: type[AMP] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic", "discriminator"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        amp_groups = cfg["obs_groups"]["discriminator"]
        amp_dim = sum(obs[group].shape[-1] for group in amp_groups)
        motion_dataset = env.unwrapped.motion_dataset
        return alg_class(
            actor,
            critic,
            storage,
            amp_observation_groups=amp_groups,
            amp_observation_dim=amp_dim,
            collect_reference_motions=motion_dataset.sample_amp_observations,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
