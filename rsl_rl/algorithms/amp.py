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


def compute_amp_reward(predictions: torch.Tensor) -> torch.Tensor:
    """Compute the bounded least-squares GAN style reward."""
    return torch.clamp(
        1.0 - 0.25 * torch.square(predictions - 1.0),
        min=0.0,
    )


def valid_amp_transition_mask(dones: torch.Tensor) -> torch.Tensor:
    """Return environments whose post-step observation is not an automatic reset."""
    return ~dones.bool()


def resolve_motion_dataset_class(algorithm_cfg: dict):
    """Resolve the configured AMP expert-motion dataset implementation."""
    dataset_format = algorithm_cfg.get("motion_dataset_format", "beyondmimic")
    from rsl_rl.datasets import MotionDataset, SomaMotionDataset

    dataset_classes = {
        "beyondmimic": MotionDataset,
        "soma": SomaMotionDataset,
    }
    try:
        return dataset_classes[dataset_format]
    except KeyError as error:
        raise ValueError(f"Unknown AMP motion dataset format: {dataset_format!r}") from error


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

    def state_dict(self) -> dict:
        """Return only initialized replay data and cursor metadata."""
        return {
            "data": self.data[: self.size].clone(),
            "size": self.size,
            "position": self.position,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore replay data into the configured local capacity."""
        saved_data = state_dict["data"].to(self.data.device)
        if saved_data.shape[0] > self.capacity:
            saved_data = saved_data[-self.capacity :]
        self.size = saved_data.shape[0]
        self.data[: self.size].copy_(saved_data)
        self.position = int(state_dict["position"]) % self.capacity
        if self.size < self.capacity:
            self.position = min(self.position, self.size)


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
        self.amp_state_dim = amp_observation_dim
        self.task_reward_scale = task_reward_scale
        self.style_reward_scale = style_reward_scale

        self.discriminator = MLP(
            2 * self.amp_state_dim,
            1,
            discriminator_hidden_dims,
            discriminator_activation,
        ).to(self.device)
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_learning_rate,
        )
        # One shared scaler is updated from policy/replay/expert states and is
        # applied independently to both sides of every transition.
        self.amp_normalizer = EmpiricalNormalization(self.amp_state_dim).to(self.device)
        self.amp_replay_buffer = AmpReplayBuffer(
            amp_replay_buffer_size,
            2 * self.amp_state_dim,
            self.device,
        )
        self._rollout_amp_transitions: list[torch.Tensor] = []
        self._current_amp_observations: torch.Tensor | None = None
        self.style_rewards = torch.zeros(1, device=self.device)

    def _get_amp_observations(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.amp_observation_groups], dim=-1)

    def _normalize_amp_transitions(self, transitions: torch.Tensor) -> torch.Tensor:
        state, next_state = transitions.split(self.amp_state_dim, dim=-1)
        return torch.cat((self.amp_normalizer(state), self.amp_normalizer(next_state)), dim=-1)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Record the pre-step AMP state before sampling an action."""
        self._current_amp_observations = self._get_amp_observations(obs).detach()
        return super().act(obs)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        if self._current_amp_observations is None:
            raise RuntimeError("AMP process_env_step must follow AMP.act")

        next_amp_observations = self._get_amp_observations(obs).detach().clone()
        terminal_amp_observations = extras.get("terminal_amp_observations")
        if terminal_amp_observations is not None:
            done_mask = dones.bool()
            next_amp_observations[done_mask] = terminal_amp_observations.to(self.device)[done_mask]

        amp_transitions = torch.cat(
            (self._current_amp_observations, next_amp_observations),
            dim=-1,
        )
        valid_mask = valid_amp_transition_mask(dones)
        if valid_mask.any():
            self._rollout_amp_transitions.append(amp_transitions[valid_mask])
        with torch.no_grad():
            predictions = self.discriminator(self._normalize_amp_transitions(amp_transitions))
            self.style_rewards = compute_amp_reward(predictions).squeeze(-1)
            self.style_rewards = self.style_rewards.masked_fill(~valid_mask, 0.0)
            combined_rewards = (
                self.task_reward_scale * rewards
                + self.style_reward_scale * self.style_rewards
            )
        extras["amp_style_reward"] = self.style_rewards.mean()
        extras["amp_task_reward"] = rewards.mean()
        super().process_env_step(obs, combined_rewards, dones, extras)
        self._current_amp_observations = None

    def update(self) -> dict[str, float]:
        online = torch.cat(self._rollout_amp_transitions, dim=0) if self._rollout_amp_transitions else None
        self._rollout_amp_transitions.clear()

        ppo_losses = super().update()
        if online is not None:
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
                states = torch.cat(
                    (
                        *policy_samples.split(self.amp_state_dim, dim=-1),
                        *replay_samples.split(self.amp_state_dim, dim=-1),
                        *expert_samples.split(self.amp_state_dim, dim=-1),
                    ),
                    dim=0,
                )
                self.amp_normalizer.update(states)
            policy_samples = self._normalize_amp_transitions(policy_samples)
            replay_samples = self._normalize_amp_transitions(replay_samples)
            expert_samples = self._normalize_amp_transitions(expert_samples).requires_grad_(True)

            policy_predictions = self.discriminator(torch.cat((policy_samples, replay_samples), dim=0))
            expert_predictions = self.discriminator(expert_samples)
            prediction_loss = 0.5 * (
                F.mse_loss(policy_predictions, -torch.ones_like(policy_predictions))
                + F.mse_loss(expert_predictions, torch.ones_like(expert_predictions))
            )
            loss = prediction_loss

            last_linear = [module for module in self.discriminator.modules() if isinstance(module, nn.Linear)][-1]
            loss = loss + self.discriminator_logit_regularization_scale * last_linear.weight.square().sum()

            expert_gradient = torch.autograd.grad(
                expert_predictions,
                expert_samples,
                grad_outputs=torch.ones_like(expert_predictions),
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
                "amp_replay_buffer_state_dict": self.amp_replay_buffer.state_dict(),
            }
        )
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("amp", True):
            self.discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
            self.discriminator_optimizer.load_state_dict(loaded_dict["amp_discriminator_optimizer_state_dict"])
            self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"], strict=strict)
            if "amp_replay_buffer_state_dict" in loaded_dict:
                self.amp_replay_buffer.load_state_dict(loaded_dict["amp_replay_buffer_state_dict"])
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "AMP":
        """Construct AMP and connect it to the motion dataset.

        The motion dataset is loaded from the path specified in cfg["algorithm"]["motion_dir"].
        """
        alg_class: type[AMP] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic", "discriminator"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        amp_groups = cfg["obs_groups"]["discriminator"]
        amp_dim = sum(obs[group].shape[-1] for group in amp_groups)

        # Load motion dataset from config path
        motion_dir = cfg["algorithm"].pop("motion_dir", None)
        if motion_dir is None:
            raise ValueError("AMP requires 'motion_dir' to be specified in algorithm config")

        # Extract body configuration
        key_body_names = cfg["algorithm"].pop("key_body_names", None)
        body_names = cfg["algorithm"].pop("body_names", None)
        joint_names = cfg["algorithm"].pop("joint_names", None)
        motion_file_pattern = cfg["algorithm"].pop("motion_file_pattern", None)
        motion_files = cfg["algorithm"].pop("motion_files", None)

        motion_dataset_class = resolve_motion_dataset_class(cfg["algorithm"])
        cfg["algorithm"].pop("motion_dataset_format", None)
        motion_dataset = motion_dataset_class(
            motion_dir=motion_dir,
            device=device,
            amp_observation_dim=amp_dim,
            time_between_frames=env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation,
            key_body_names=key_body_names,
            body_names=body_names,
            joint_names=joint_names,
            motion_file_pattern=motion_file_pattern,
            motion_files=motion_files,
        )

        # Extract AMP-specific parameters from config
        amp_params = {
            "discriminator_hidden_dims": cfg["algorithm"].pop("discriminator_hidden_dims", (1024, 512)),
            "discriminator_activation": cfg["algorithm"].pop("discriminator_activation", "relu"),
            "discriminator_learning_rate": cfg["algorithm"].pop("discriminator_learning_rate", 5e-4),
            "discriminator_batch_size": cfg["algorithm"].pop("discriminator_batch_size", 4096),
            "discriminator_updates": cfg["algorithm"].pop("discriminator_updates", 4),
            "discriminator_loss_scale": cfg["algorithm"].pop("discriminator_loss_scale", 5.0),
            "discriminator_logit_regularization_scale": cfg["algorithm"].pop("discriminator_logit_regularization_scale", 0.05),
            "discriminator_gradient_penalty_scale": cfg["algorithm"].pop("discriminator_gradient_penalty_scale", 5.0),
            "discriminator_weight_decay_scale": cfg["algorithm"].pop("discriminator_weight_decay_scale", 1e-4),
            "amp_replay_buffer_size": cfg["algorithm"].pop("amp_replay_buffer_size", 200000),
            "task_reward_scale": cfg["algorithm"].pop("task_reward_scale", 0.0),
            "style_reward_scale": cfg["algorithm"].pop("style_reward_scale", 1.0),
        }

        return alg_class(
            actor,
            critic,
            storage,
            amp_observation_groups=amp_groups,
            amp_observation_dim=amp_dim,
            collect_reference_motions=motion_dataset.sample_amp_observations,
            device=device,
            **cfg["algorithm"],
            **amp_params,
            multi_gpu_cfg=cfg["multi_gpu"],
        )
