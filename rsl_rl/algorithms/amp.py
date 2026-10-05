# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversarial Motion Prior (AMP) on top of PPO."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable
from typing import Any
import warnings
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.datasets.rsi_contract import RsiJointLayout, RsiState, reference_joint_layout, reference_state
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import GaussianDistribution
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
    from rsl_rl.datasets import MotionDataset, SomaMotionDataset, VariableDofMotionDataset

    dataset_classes = {
        "beyondmimic": MotionDataset,
        "soma": SomaMotionDataset,
        "variable_dof": VariableDofMotionDataset,
    }
    try:
        return dataset_classes[dataset_format]
    except KeyError as error:
        raise ValueError(f"Unknown AMP motion dataset format: {dataset_format!r}") from error


def _resolve_rsi_frame_sampler(
    expert_dataset: Any, rsi_dataset: Any | None = None
) -> Callable[[int], dict[str, torch.Tensor]]:
    """Keep RSI sampling independent when a dedicated initialization dataset is configured."""
    if rsi_dataset is not None:
        return rsi_dataset.sample_reference_states
    return expert_dataset.sample_reference_frames


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


class AmpNormalizer(nn.Module):
    """AMP-Go2 running moments with clipped normalization and checkpoint buffers."""

    def __init__(self, observation_dim: int):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, observation_dim, dtype=torch.float64))
        self.register_buffer("_var", torch.ones(1, observation_dim, dtype=torch.float64))
        self.register_buffer("_std", torch.ones(1, observation_dim, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1.0e-4, dtype=torch.float64))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mean = self._mean.to(observations.dtype)
        std = (self._var + 1.0e-4).to(observations.dtype).sqrt()
        return ((observations - mean) / std).clamp(-10.0, 10.0)

    @torch.no_grad()
    def update(self, observations: torch.Tensor) -> None:
        if not self.training or observations.shape[0] == 0:
            return
        observations = observations.to(dtype=torch.float64, device=self._mean.device)
        batch_count = observations.shape[0]
        batch_mean = observations.mean(dim=0, keepdim=True)
        batch_var = observations.var(dim=0, unbiased=False, keepdim=True)
        delta = batch_mean - self._mean
        total_count = self.count + batch_count
        second_moment = self._var * self.count + batch_var * batch_count
        second_moment += delta.square() * self.count * batch_count / total_count
        self._mean.add_(delta * batch_count / total_count)
        self._var.copy_(second_moment / total_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(total_count)


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
        reference_frame_sampler: Callable[[int], dict[str, torch.Tensor]] | None = None,
        reference_state_initialization: bool = False,
        discriminator_hidden_dims: tuple[int, ...] | list[int] = (1024, 512),
        discriminator_activation: str = "lrelu",
        discriminator_learning_rate: float | None = None,
        discriminator_batch_size: int | None = None,
        discriminator_updates: int | None = None,
        discriminator_loss_scale: float = 1.0,
        discriminator_logit_regularization_scale: float = 0.0,
        discriminator_gradient_penalty_scale: float = 10.0,
        discriminator_weight_decay_scale: float = 0.0,
        amp_replay_buffer_size: int = 100_000,
        task_reward_scale: float | None = None,
        style_reward_scale: float | None = None,
        amp_reward_coef: float = 0.2,
        task_reward_lerp: float | None = None,
        min_std: float | list[float] | torch.Tensor | None = None,
        reference_motion_dataset: Any | None = None,
        **kwargs,
    ):
        super().__init__(actor, critic, storage, **kwargs)
        self.amp_observation_groups = amp_observation_groups
        self.collect_reference_motions = collect_reference_motions
        self._reference_frame_sampler = reference_frame_sampler
        self.reference_motion_dataset = reference_motion_dataset
        self.reference_state_initialization = reference_state_initialization
        self.discriminator_batch_size = storage.num_envs * storage.num_transitions_per_env // self.num_mini_batches
        self.discriminator_updates = self.num_learning_epochs * self.num_mini_batches
        legacy_settings = (
            ("discriminator_learning_rate", discriminator_learning_rate, self.learning_rate),
            ("discriminator_batch_size", discriminator_batch_size, self.discriminator_batch_size),
            ("discriminator_updates", discriminator_updates, self.discriminator_updates),
        )
        for name, configured, effective in legacy_settings:
            if configured is not None and configured != effective:
                warnings.warn(
                    f"{name}={configured} is ignored by AMP-Go2 semantics; effective value is {effective} "
                    "and follows PPO's optimizer/minibatches.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        self.discriminator_loss_scale = discriminator_loss_scale
        self.discriminator_logit_regularization_scale = discriminator_logit_regularization_scale
        self.discriminator_gradient_penalty_scale = discriminator_gradient_penalty_scale
        self.discriminator_weight_decay_scale = discriminator_weight_decay_scale
        self.amp_state_dim = amp_observation_dim
        if task_reward_lerp is not None and (task_reward_scale is not None or style_reward_scale is not None):
            raise ValueError("Use task_reward_lerp/amp_reward_coef or legacy reward scales, not both.")
        if task_reward_lerp is None:
            task_reward_lerp = 0.0 if task_reward_scale is None else task_reward_scale
            if style_reward_scale is not None:
                if task_reward_lerp >= 1.0 and style_reward_scale != 0.0:
                    raise ValueError("Legacy style_reward_scale must be zero when task_reward_scale is one.")
                amp_reward_coef = style_reward_scale / (1.0 - task_reward_lerp) if task_reward_lerp < 1.0 else 0.0
        if not 0.0 <= task_reward_lerp <= 1.0 or amp_reward_coef < 0.0:
            raise ValueError("AMP requires task_reward_lerp in [0, 1] and a non-negative amp_reward_coef.")
        self.task_reward_lerp = task_reward_lerp
        self.amp_reward_coef = amp_reward_coef
        self.min_std = None if min_std is None else torch.as_tensor(min_std, device=self.device)
        if self.min_std is not None:
            if not isinstance(self._raw_actor.distribution, GaussianDistribution):
                raise ValueError("AMP min_std requires a state-independent Gaussian actor distribution.")
            distribution = self._raw_actor.distribution
            std_parameter = distribution.std_param if distribution.std_type == "scalar" else distribution.log_std_param
            if self.min_std.ndim > 1 or self.min_std.numel() not in (1, std_parameter.numel()):
                raise ValueError("AMP min_std must be a scalar or one value per action.")
            if not torch.all(torch.isfinite(self.min_std) & (self.min_std > 0.0)):
                raise ValueError("AMP min_std must be positive.")

        self.discriminator = MLP(
            2 * self.amp_state_dim,
            1,
            discriminator_hidden_dims,
            discriminator_activation,
        ).to(self.device)
        # Optimizer weight-decay groups follow the chocolate AMP setup:
        # light decay on the trunk, heavy decay on the output head.
        discriminator_parameters = list(self.discriminator.parameters())
        head_linears = [module for module in self.discriminator.modules() if isinstance(module, nn.Linear)]
        head_param_ids = {id(p) for module in head_linears[-1:] for p in module.parameters()}
        # AMP-Go uses one Adam optimizer and updates its discriminator on each
        # PPO minibatch. Keep separate decay groups while sharing that optimizer.
        self.optimizer.add_param_group(
            {
                "params": [p for p in discriminator_parameters if id(p) not in head_param_ids],
                "weight_decay": 10.0e-4,
                "name": "amp_trunk",
            }
        )
        self.optimizer.add_param_group(
            {
                "params": [p for p in discriminator_parameters if id(p) in head_param_ids],
                "weight_decay": 10.0e-2,
                "name": "amp_head",
            }
        )
        # Normalize both sides using the same moments. Update moments from
        # policy/expert current states after each optimizer step, as AMP-Go2 does.
        self.amp_normalizer = AmpNormalizer(self.amp_state_dim).to(self.device)
        self.amp_replay_buffer = AmpReplayBuffer(
            amp_replay_buffer_size,
            2 * self.amp_state_dim,
            self.device,
        )
        self._amp_normalizer_samples: tuple[torch.Tensor, torch.Tensor] | None = None
        self._current_amp_observations: torch.Tensor | None = None
        self.style_rewards = torch.zeros(1, device=self.device)

    def _get_amp_observations(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.amp_observation_groups], dim=-1)

    def get_rsi_joint_layout(self) -> RsiJointLayout:
        """Return fixed coordinate metadata; independent of the per-reset sampled frame."""
        if getattr(self, "_rsi_joint_layout", None) is None:
            self._rsi_joint_layout = reference_joint_layout(self.reference_motion_dataset)
        return dict(self._rsi_joint_layout)

    def sample_rsi(self, env_ids: torch.Tensor) -> RsiState:
        """Sample aligned reset states from the configured RSI source."""
        if self._reference_frame_sampler is None:
            raise RuntimeError("AMP RSI requested, but no reference-frame sampler was configured")
        if env_ids.ndim != 1:
            raise ValueError(f"AMP RSI env_ids must be one-dimensional, got shape {tuple(env_ids.shape)}")
        layout = self.get_rsi_joint_layout()
        count = int(env_ids.numel())
        return reference_state(self._reference_frame_sampler(count), count, layout)

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
        if terminal_amp_observations is None:
            # Isaac Lab exposes the pre-reset observation as ``final_obs``
            # when compute_final_obs is enabled.  Convert it to the AMP
            # observation contract before the environment's autoreset state
            # is used, matching Chocolate's terminal AMP handling.
            final_obs = extras.get("final_obs")
            if final_obs is not None:
                terminal_amp_observations = torch.cat(
                    [final_obs[group] for group in self.amp_observation_groups],
                    dim=-1,
                )
        done_mask = dones.to(self.device).bool().reshape(-1)
        if terminal_amp_observations is None and done_mask.any():
            raise ValueError(
                "AMP-Go2 terminal transitions require pre-reset AMP states. Provide "
                "extras['terminal_amp_observations'] or extras['final_obs']; "
                "enable compute_final_obs in auto-reset environments."
            )
        if terminal_amp_observations is not None:
            terminal_amp_observations = terminal_amp_observations.to(self.device)
            if terminal_amp_observations.shape == next_amp_observations.shape:
                terminal_states = terminal_amp_observations[done_mask]
            elif terminal_amp_observations.shape == (int(done_mask.sum().item()), self.amp_state_dim):
                terminal_states = terminal_amp_observations
            else:
                raise ValueError("terminal_amp_observations must contain all environments or done environments in mask order.")
            next_amp_observations[done_mask] = terminal_states

        amp_transitions = torch.cat(
            (self._current_amp_observations, next_amp_observations),
            dim=-1,
        )
        valid_mask = valid_amp_transition_mask(dones)
        if terminal_amp_observations is not None:
            # The pre-reset observation makes terminal transitions valid,
            # so reward and train on them as Chocolate AMP does.
            valid_mask = torch.ones_like(valid_mask)
        if valid_mask.any():
            self.amp_replay_buffer.add(amp_transitions[valid_mask])
        with torch.no_grad():
            predictions = self.discriminator(self._normalize_amp_transitions(amp_transitions))
            self.style_rewards = self.amp_reward_coef * compute_amp_reward(predictions).squeeze(-1)
            self.style_rewards = self.style_rewards.masked_fill(~valid_mask, 0.0)
            combined_rewards = (
                self.task_reward_lerp * rewards
                + (1.0 - self.task_reward_lerp) * self.style_rewards
            )
        extras["amp_style_reward"] = self.style_rewards.mean()
        extras["amp_task_reward"] = rewards.mean()
        super().process_env_step(obs, combined_rewards, dones, extras)
        self._current_amp_observations = None

    def update(self) -> dict[str, float]:
        self._amp_normalizer_samples = None
        try:
            losses = super().update()
        finally:
            self._amp_normalizer_samples = None
        losses["amp_style_reward"] = self.style_rewards.mean().item()
        return losses

    def _compute_auxiliary_loss(
        self, batch, original_batch_size: int
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if self.amp_replay_buffer.size == 0:
            return None, {}

        # AMP-Go samples one discriminator minibatch alongside every PPO
        # minibatch, using the PPO minibatch size for both streams.
        batch_size = self.discriminator_batch_size
        policy_samples = self.amp_replay_buffer.sample(batch_size)
        expert_samples = self.collect_reference_motions(batch_size).to(self.device)
        self._amp_normalizer_samples = (
            policy_samples[:, :self.amp_state_dim].detach(),
            expert_samples[:, :self.amp_state_dim].detach(),
        )
        policy_samples = self._normalize_amp_transitions(policy_samples)
        expert_samples = self._normalize_amp_transitions(expert_samples).requires_grad_(True)

        policy_predictions = self.discriminator(policy_samples)
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
        if self.discriminator_weight_decay_scale:
            all_weights = torch.cat(
                [module.weight.flatten() for module in self.discriminator.modules() if isinstance(module, nn.Linear)]
            )
            loss = loss + self.discriminator_weight_decay_scale * all_weights.square().sum()
        loss = self.discriminator_loss_scale * loss
        return loss, {
            "amp_discriminator": loss.detach(),
            "amp_prediction": prediction_loss.detach(),
            "amp_gradient_penalty": gradient_penalty.detach(),
            "amp_policy_prediction": policy_predictions.mean().detach(),
            "amp_expert_prediction": expert_predictions.mean().detach(),
        }

    def _clip_gradients(self) -> None:
        """AMP-Go2 clips mean-network/critic gradients, leaving std and discriminator untouched."""
        distribution = self._raw_actor.distribution
        distribution_parameters = set() if distribution is None else {id(p) for p in distribution.parameters()}
        actor_parameters = [p for p in self.actor.parameters() if id(p) not in distribution_parameters]
        nn.utils.clip_grad_norm_(actor_parameters, self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

    def _after_optimizer_step(self) -> None:
        """Match reference std clamping and post-step current-state moment updates."""
        with torch.no_grad():
            if self.min_std is not None:
                distribution = self._raw_actor.distribution
                if distribution.std_type == "scalar":
                    distribution.std_param.copy_(torch.maximum(distribution.std_param, self.min_std))
                else:
                    distribution.log_std_param.copy_(torch.maximum(distribution.log_std_param, self.min_std.log()))
        if self._amp_normalizer_samples is not None:
            for states in self._amp_normalizer_samples:
                self.amp_normalizer.update(states)
            self._amp_normalizer_samples = None

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
                "amp_normalizer_state_dict": self.amp_normalizer.state_dict(),
                "amp_replay_buffer_state_dict": self.amp_replay_buffer.state_dict(),
            }
        )
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("amp", True):
            self.discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
            self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"], strict=strict)
            if "amp_replay_buffer_state_dict" in loaded_dict:
                self.amp_replay_buffer.load_state_dict(loaded_dict["amp_replay_buffer_state_dict"])
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "AMP":
        """Construct AMP and connect it to the motion dataset.

        Expert motions use ``motion_dir``. When RSI is enabled, ``rsi_motion_dir`` may
        independently select its reset-state source; it can contain single-frame clips.
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
        rsi_motion_dir = cfg["algorithm"].pop("rsi_motion_dir", None)
        reference_state_initialization = cfg["algorithm"].pop("reference_state_initialization", False)

        # Extract body configuration
        key_body_names = cfg["algorithm"].pop("key_body_names", None)
        body_names = cfg["algorithm"].pop("body_names", None)
        joint_names = cfg["algorithm"].pop("joint_names", None)
        legacy_profile = cfg["algorithm"].pop("observation_profile", None)
        if legacy_profile is not None:
            warnings.warn(
                "'observation_profile' is deprecated and ignored; AMP motion observations now use one layout.",
                DeprecationWarning,
                stacklevel=2,
            )
        legacy_root_height = cfg["algorithm"].pop("amp_include_root_height", None)
        if legacy_root_height:
            raise ValueError("amp_include_root_height=True is incompatible with the unified AMP observation layout.")
        quaternion_format = cfg["algorithm"].pop("motion_quaternion_format", "wxyz")
        motion_file_pattern = cfg["algorithm"].pop("motion_file_pattern", None)
        motion_files = cfg["algorithm"].pop("motion_files", None)

        motion_dataset_class = resolve_motion_dataset_class(cfg["algorithm"])
        rsi_joint_names = cfg["algorithm"].pop("rsi_joint_names", None)
        dataset_options = {}
        if rsi_joint_names is not None:
            from rsl_rl.datasets import VariableDofMotionDataset

            if motion_dataset_class is not VariableDofMotionDataset:
                raise ValueError("rsi_joint_names requires motion_dataset_format='variable_dof'")
            dataset_options["rsi_joint_names"] = rsi_joint_names
        cfg["algorithm"].pop("motion_dataset_format", None)
        motion_dataset = motion_dataset_class(
            motion_dir=motion_dir,
            device=device,
            amp_observation_dim=amp_dim,
            time_between_frames=env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation,
            key_body_names=key_body_names,
            body_names=body_names,
            joint_names=joint_names,
            quaternion_format=quaternion_format,
            motion_file_pattern=motion_file_pattern,
            motion_files=motion_files,
            **dataset_options,
        )
        rsi_motion_dataset = None
        if reference_state_initialization and rsi_motion_dir:
            rsi_motion_dataset = motion_dataset_class(
                motion_dir=rsi_motion_dir,
                device=device,
                amp_observation_dim=amp_dim,
                time_between_frames=env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation,
                key_body_names=key_body_names,
                body_names=body_names,
                joint_names=joint_names,
                quaternion_format=quaternion_format,
                require_transitions=False,
                **dataset_options,
            )

        # Extract AMP-specific parameters from config
        amp_params = {
            "discriminator_hidden_dims": cfg["algorithm"].pop("discriminator_hidden_dims", (1024, 512)),
            "discriminator_activation": cfg["algorithm"].pop("discriminator_activation", "lrelu"),
            "discriminator_learning_rate": cfg["algorithm"].pop("discriminator_learning_rate", None),
            "discriminator_batch_size": cfg["algorithm"].pop("discriminator_batch_size", None),
            "discriminator_updates": cfg["algorithm"].pop("discriminator_updates", None),
            "discriminator_loss_scale": cfg["algorithm"].pop("discriminator_loss_scale", 1.0),
            "discriminator_logit_regularization_scale": cfg["algorithm"].pop(
                "discriminator_logit_regularization_scale", 0.0
            ),
            "discriminator_gradient_penalty_scale": cfg["algorithm"].pop("discriminator_gradient_penalty_scale", 10.0),
            "discriminator_weight_decay_scale": cfg["algorithm"].pop("discriminator_weight_decay_scale", 0.0),
            "amp_replay_buffer_size": cfg["algorithm"].pop("amp_replay_buffer_size", 100000),
            "reference_state_initialization": reference_state_initialization,
            "task_reward_scale": cfg["algorithm"].pop("task_reward_scale", None),
            "style_reward_scale": cfg["algorithm"].pop("style_reward_scale", None),
            "amp_reward_coef": cfg["algorithm"].pop("amp_reward_coef", 0.2),
            "task_reward_lerp": cfg["algorithm"].pop("task_reward_lerp", None),
            "min_std": cfg["algorithm"].pop("min_std", None),
        }

        return alg_class(
            actor,
            critic,
            storage,
            amp_observation_groups=amp_groups,
            amp_observation_dim=amp_dim,
            collect_reference_motions=motion_dataset.sample_amp_observations,
            reference_frame_sampler=_resolve_rsi_frame_sampler(motion_dataset, rsi_motion_dataset),
            reference_motion_dataset=motion_dataset,
            device=device,
            **cfg["algorithm"],
            **amp_params,
            multi_gpu_cfg=cfg["multi_gpu"],
        )
