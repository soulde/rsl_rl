# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with a scheduled teacher behavior-cloning objective.

This algorithm is separate from :class:`PPO` and the DAgger runners. It keeps
the ordinary PPO and DAgger APIs intact while supporting a frozen bank of
teacher policies selected by an optional routing observation.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from itertools import chain

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import (
    load_actor_checkpoint,
    resolve_callable,
    resolve_obs_groups,
)


class DistillationPPO(PPO):
    """PPO trainer with teacher-action regression and optional teacher routing.

    The PPO policy and entropy terms ramp from zero to ``max_ppo_coef``. The
    critic loss stays active throughout training and the teacher loss is scaled
    by ``dagger_loss_coef * (1 - ppo_coef)``. A negative routing index marks a
    transition whose teacher is no longer tracking the reference; its teacher
    loss is masked, matching PHP's task-level DistillationPPO wrapper.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        teachers: Sequence[MLPModel],
        teacher_checkpoints: Sequence[str],
        env: VecEnv | None = None,
        routing_obs_group: str | None = None,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adamw",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        min_learning_rate: float = 1.0e-5,
        timeout_bootstrap_from_final_obs: bool = False,
        reshuffle_minibatches_each_epoch: bool = False,
        ppo_start_iteration: int = 0,
        dagger_end_iteration: int = 10_000,
        distillation_warmup_steps: int = 0,
        max_ppo_coef: float = 0.9,
        dagger_loss_coef: float = 10.0,
        distill_loss_type: str = "mse",
        mask_zero_teacher_actions: bool = False,
        weight_decay: float = 0.0,
        depth_weight_decay: float = 1.0e-2,
        teacher_labels: Sequence[str] = (),
    ) -> None:
        if not teachers:
            raise ValueError("DistillationPPO requires at least one frozen teacher model")
        if len(teachers) != len(teacher_checkpoints):
            raise ValueError("teacher model and checkpoint counts must match")
        if teacher_labels and len(teachers) != len(teacher_labels):
            raise ValueError("teacher model and label counts must match")
        if actor.is_recurrent or critic.is_recurrent:
            raise ValueError("DistillationPPO currently supports feed-forward actor and critic models")
        if dagger_end_iteration <= ppo_start_iteration:
            raise ValueError("dagger_end_iteration must be greater than ppo_start_iteration")
        if distillation_warmup_steps < 0:
            raise ValueError("distillation_warmup_steps must be non-negative")
        if not 0.0 <= max_ppo_coef <= 1.0:
            raise ValueError("max_ppo_coef must be in [0, 1]")
        if dagger_loss_coef < 0.0 or weight_decay < 0.0 or depth_weight_decay < 0.0:
            raise ValueError("loss and weight-decay coefficients must be non-negative")
        if distill_loss_type not in ("mse", "huber"):
            raise ValueError("distill_loss_type must be 'mse' or 'huber'")

        super().__init__(
            actor,
            critic,
            storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
            min_learning_rate=min_learning_rate,
            timeout_bootstrap_from_final_obs=timeout_bootstrap_from_final_obs,
            reshuffle_minibatches_each_epoch=reshuffle_minibatches_each_epoch,
        )

        self.teachers = torch.nn.ModuleList(teachers).to(device)
        self.env = env
        self.teacher_checkpoints = list(teacher_checkpoints)
        self.teacher_labels = list(teacher_labels)
        self.routing_obs_group = routing_obs_group
        self.ppo_start_iteration = ppo_start_iteration
        self.dagger_end_iteration = dagger_end_iteration
        self.distillation_warmup_steps = distillation_warmup_steps
        self.max_ppo_coef = max_ppo_coef
        self.dagger_loss_coef = dagger_loss_coef
        self.distill_loss_type = distill_loss_type
        self.mask_zero_teacher_actions = mask_zero_teacher_actions
        self.depth_weight_decay = depth_weight_decay
        self.ppo_coef = 0.0
        self._warmup_active = False
        self.current_learning_iteration = 0
        self._std_group_index: int | None = None
        self._build_distillation_optimizer(optimizer, learning_rate, weight_decay)
        self._set_std_learning_rate()

        for teacher in self.teachers:
            teacher.eval()
            teacher.requires_grad_(False)

    def _build_distillation_optimizer(self, optimizer_name: str, learning_rate: float, weight_decay: float) -> None:
        """Use separate AdamW groups for depth features and policy std."""
        if optimizer_name.lower() != "adamw":
            raise ValueError("DistillationPPO uses AdamW to match the PHP training recipe")

        std_parameters = []
        distribution = getattr(self.actor, "distribution", None)
        if distribution is not None:
            std_parameters = [parameter for parameter in distribution.parameters() if parameter.requires_grad]
        std_ids = {id(parameter) for parameter in std_parameters}

        depth_module = getattr(self.actor, "cnns", None)
        depth_parameters = list(depth_module.parameters()) if depth_module is not None else []
        depth_ids = {id(parameter) for parameter in depth_parameters}
        actor_parameters = [
            parameter
            for parameter in self.actor.parameters()
            if parameter.requires_grad and id(parameter) not in std_ids | depth_ids
        ]
        actor_ids = {id(parameter) for parameter in actor_parameters}
        critic_parameters = [
            parameter
            for parameter in self.critic.parameters()
            if parameter.requires_grad and id(parameter) not in std_ids | depth_ids | actor_ids
        ]

        groups = [
            {"params": actor_parameters, "lr": learning_rate, "weight_decay": weight_decay},
            {"params": critic_parameters, "lr": learning_rate, "weight_decay": weight_decay},
        ]
        if depth_parameters:
            groups.append({"params": depth_parameters, "lr": learning_rate, "weight_decay": self.depth_weight_decay})
        if std_parameters:
            self._std_group_index = len(groups)
            groups.append({"params": std_parameters, "lr": 0.0, "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(groups, lr=learning_rate)

    def _set_schedule(self) -> None:
        iteration = self.current_learning_iteration
        self._warmup_active = iteration < self.distillation_warmup_steps
        if iteration < self.ppo_start_iteration:
            self.ppo_coef = 0.0
        elif iteration >= self.dagger_end_iteration:
            self.ppo_coef = self.max_ppo_coef
        else:
            fraction = (iteration - self.ppo_start_iteration) / (
                self.dagger_end_iteration - self.ppo_start_iteration
            )
            self.ppo_coef = min(fraction, self.max_ppo_coef)
        self._set_std_learning_rate()

    def _set_std_learning_rate(self) -> None:
        if self._std_group_index is not None:
            self.optimizer.param_groups[self._std_group_index]["lr"] = (
                self.learning_rate if self.ppo_coef > 0.1 else 0.0
            )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample student actions and record routed, frozen-teacher targets."""
        # Match PHP's ``learn()`` ordering: update the PPO/DAgger coefficient
        # before the first action of each rollout, then keep it fixed through
        # that rollout's optimization step. RSL-RL clears storage after update,
        # so ``step == 0`` marks the next rollout boundary.
        if self.storage.step == 0:
            self._set_schedule()
        actions = super().act(obs)
        with torch.no_grad():
            route = self._get_teacher_indices(obs)
            teacher_outputs = torch.stack([teacher(obs) for teacher in self.teachers], dim=0)
            if route is None:
                labels = teacher_outputs[0]
            else:
                safe_route = route.clamp(min=0)
                gather_index = safe_route.view(1, -1, 1).expand(1, -1, teacher_outputs.shape[-1])
                labels = teacher_outputs.gather(0, gather_index).squeeze(0)
                labels = labels.masked_fill((route < 0).unsqueeze(-1), 0.0)
            self.transition.privileged_actions = labels.detach()
            student_mean = self.actor(obs, stochastic_output=False)
            self._update_ppo_gate_stats(student_mean, labels)
        # PHP collects student distribution statistics but lets the teacher
        # drive the environment during its optional initial DAgger warmup.
        return labels if self._warmup_active else actions

    def _get_teacher_indices(self, obs: TensorDict) -> torch.Tensor | None:
        if self.routing_obs_group is None:
            if len(self.teachers) > 1:
                raise ValueError("multiple teachers require a routing_obs_group")
            return None
        if self.routing_obs_group not in obs:
            raise KeyError(f"routing observation group {self.routing_obs_group!r} is missing")
        values = obs[self.routing_obs_group].reshape(obs.batch_size[0], -1)
        if values.shape[1] != 1:
            raise ValueError(f"routing observation must be one-dimensional per sample, got {values.shape[1]} values")
        route = values[:, 0].long()
        if torch.any(values[:, 0] != route.to(values.dtype)):
            raise ValueError("teacher routing indices must be integer-valued")
        if torch.any(route >= len(self.teachers)):
            raise ValueError(f"teacher routing index exceeds the configured teacher count ({len(self.teachers)})")
        return route

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if self.transition.privileged_actions is None:
            raise RuntimeError("DistillationPPO did not record teacher actions before the environment step")
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        metrics = super().update()
        if self._warmup_active:
            # PHP's warmup update is DAgger-only, even if the PPO coefficient
            # schedule has already started.
            for name in ("value", "surrogate", "entropy"):
                if name in metrics:
                    metrics[name] = 0.0
        metrics["warmup_active"] = float(self._warmup_active)
        self.current_learning_iteration += 1
        return metrics

    def _compose_policy_value_loss(
        self,
        surrogate_loss: torch.Tensor,
        value_loss: torch.Tensor,
        mean_entropy: torch.Tensor,
        *,
        surrogate_per_sample: torch.Tensor | None = None,
        entropy_per_sample: torch.Tensor | None = None,
        batch=None,
        original_batch_size: int | None = None,
    ) -> torch.Tensor:
        if self._warmup_active:
            # Keep PPO and critic parameters out of the graph entirely. A
            # zero-valued graph would still create zero gradients and let
            # AdamW apply weight decay during the PHP DAgger-only phase.
            return value_loss.new_zeros(())
        if surrogate_per_sample is None or entropy_per_sample is None or batch is None:
            return self.value_loss_coef * value_loss + self.ppo_coef * (
                surrogate_loss - self.entropy_coef * mean_entropy
            )
        gate = self._ppo_gate(batch.observations, original_batch_size)
        entropy = entropy_per_sample.reshape(-1)
        coef = gate * self.ppo_coef
        ppo_terms = coef * (surrogate_per_sample.reshape(-1) - self.entropy_coef * entropy)
        return self.value_loss_coef * value_loss + ppo_terms.mean()

    def _compute_kl_mean(self, kl: torch.Tensor, batch, original_batch_size: int) -> torch.Tensor:
        """Match PHP's epsilon-stabilized Gaussian KL and gate-weighted mean."""
        del kl  # Recompute from distribution parameters to retain PHP's log epsilon.
        if batch.old_distribution_params is None:
            raise RuntimeError("DistillationPPO minibatch is missing old Gaussian parameters")
        old_mean, old_std = batch.old_distribution_params
        new_mean, new_std = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
        kl = torch.sum(
            torch.log(new_std / old_std[:original_batch_size] + 1.0e-5)
            + (
                old_std[:original_batch_size].square()
                + (old_mean[:original_batch_size] - new_mean).square()
            )
            / (2.0 * new_std.square())
            - 0.5,
            dim=-1,
        )
        gate = self._ppo_gate(batch.observations, original_batch_size)
        weights = gate.reshape(-1)
        return (kl.reshape(-1) * weights).sum() / weights.sum().clamp(min=1e-6)

    def _should_adapt_learning_rate(self) -> bool:
        return not self._warmup_active and self.ppo_coef > 0.1

    def _compute_auxiliary_loss(self, batch, original_batch_size: int):
        if batch.privileged_actions is None:
            raise RuntimeError("DistillationPPO rollout storage has no teacher-action targets")
        prediction = self.actor(batch.observations, stochastic_output=False)[:original_batch_size]
        target = batch.privileged_actions[:original_batch_size]
        if self.distill_loss_type == "mse":
            per_action = F.mse_loss(prediction, target, reduction="none")
        else:
            per_action = F.huber_loss(prediction, target, reduction="none")
        per_sample = per_action.mean(dim=-1)

        gate = self._ppo_gate(batch.observations, original_batch_size)
        coefficient = 1.0 if self._warmup_active else 1.0 - gate * self.ppo_coef
        if self.mask_zero_teacher_actions:
            # PHP's PhpDistillationPPO identifies lost-expert samples from the
            # all-zero teacher label emitted by its routing-aware teacher.
            # Keep this opt-in because other routed distillation users may
            # legitimately train on an all-zero teacher action.
            keep = ~torch.all(target == 0.0, dim=-1)
            per_sample = per_sample * keep.to(per_sample.dtype)
        elif self.routing_obs_group is not None:
            route = self._get_teacher_indices(batch.observations)[:original_batch_size]
            per_sample = per_sample * (route >= 0).to(per_sample.dtype)
        behavior_loss = (coefficient * per_sample).mean()
        auxiliary_loss = self.dagger_loss_coef * behavior_loss
        return auxiliary_loss, {
            "behavior": behavior_loss.detach(),
            "ppo_coef": torch.tensor(self.ppo_coef, device=prediction.device),
            "ppo_gate_mean": gate.mean().detach(),
        }

    @staticmethod
    def _ppo_gate(observations: TensorDict, original_batch_size: int | None = None) -> torch.Tensor:
        if "ppo_gate" not in observations:
            size = observations.batch_size[0] if original_batch_size is None else original_batch_size
            return torch.ones(size, device=observations.device)
        values = observations["ppo_gate"].reshape(observations.batch_size[0], -1)
        if values.shape[1] != 1:
            raise ValueError(f"ppo_gate must contain one scalar per sample, got {values.shape[1]} values")
        gate = values[:original_batch_size, 0] if original_batch_size is not None else values[:, 0]
        if not torch.isfinite(gate).all() or torch.any(gate < 0.0) or torch.any(gate > 1.0):
            raise ValueError("ppo_gate values must be finite and in [0, 1]")
        return gate

    def _update_ppo_gate_stats(self, student_mean: torch.Tensor, teacher_actions: torch.Tensor) -> None:
        if self.env is None:
            return
        env = getattr(self.env, "unwrapped", self.env)
        command_manager = getattr(env, "command_manager", None)
        if command_manager is None:
            return
        command = command_manager.get_term("motion")
        update = getattr(command, "observe_dagger_residual", None)
        if update is None:
            return
        residual = (student_mean - teacher_actions).pow(2).mean(dim=-1)
        keep = ~torch.all(teacher_actions == 0.0, dim=-1)
        update(residual.detach(), keep.detach())

    def _after_optimizer_step(self) -> None:
        self._set_std_learning_rate()

    def _clip_gradients(self) -> None:
        """Match PHP distillation's single global actor-and-critic norm clip."""
        torch.nn.utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)

    def train_mode(self) -> None:
        super().train_mode()
        self.teachers.eval()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.teachers.eval()

    def save(self) -> dict:
        saved = super().save()
        saved.update(
            {
                # OnPolicyRunner writes its last completed iteration index to
                # ``iter`` after this method returns. The algorithm counter is
                # advanced immediately after update(), so subtract one here to
                # keep standalone metadata and PHP-style resume scheduling on
                # that same index (iteration 0 remains 0 before the first update).
                "distillation_iteration": max(self.current_learning_iteration - 1, 0),
                "ppo_coef": self.ppo_coef,
                "teacher_checkpoints": self.teacher_checkpoints,
                "teacher_labels": self.teacher_labels,
                # Frozen experts are part of the training state: storing only
                # their source paths makes resume depend on external files
                # remaining unchanged, unlike the student/critic state above.
                "teacher_state_dict": self.teachers.state_dict(),
            }
        )
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        checkpoint_labels = list(loaded_dict.get("teacher_labels", []) or [])
        if self.teacher_labels and checkpoint_labels and self.teacher_labels != checkpoint_labels:
            raise ValueError(
                f"Checkpoint teacher labels {checkpoint_labels} do not match configured labels {self.teacher_labels}"
            )
        if not self.teacher_labels:
            self.teacher_labels = checkpoint_labels
        if self.teacher_labels and len(self.teacher_labels) != len(self.teachers):
            raise ValueError("Checkpoint teacher labels do not match the configured teacher bank size")
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        load_teachers = load_cfg is None or load_cfg.get("teachers", load_cfg.get("actor", False))
        teacher_state = loaded_dict.get("teacher_state_dict")
        if load_teachers and teacher_state is not None:
            self.teachers.load_state_dict(teacher_state, strict=strict)
        if load_cfg is None or load_cfg.get("iteration", False):
            self.current_learning_iteration = int(
                loaded_dict.get("iter", loaded_dict.get("distillation_iteration", 0))
            )
        self._set_schedule()
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> DistillationPPO:
        """Construct the student PPO models and the frozen teacher bank."""
        algorithm_cfg = cfg["algorithm"]
        alg_class = resolve_callable(algorithm_cfg.pop("class_name"))
        actor_cfg = dict(cfg["actor"])
        critic_cfg = dict(cfg["critic"])
        teacher_cfg = dict(cfg["teacher"])
        actor_class = resolve_callable(actor_cfg.pop("class_name"))
        critic_class = resolve_callable(critic_cfg.pop("class_name"))
        teacher_class = resolve_callable(teacher_cfg.pop("class_name"))

        checkpoint_value = teacher_cfg.pop("checkpoints", teacher_cfg.pop("checkpoint", None))
        checkpoints = _normalize_checkpoints(checkpoint_value)
        teacher_labels = list(teacher_cfg.pop("labels", []) or [])
        resume_payload = None
        if not checkpoints:
            resume_path = cfg.get("resume_checkpoint_path")
            if resume_path is None:
                raise ValueError(
                    "configure teacher.checkpoint(s), or provide resume_checkpoint_path for a checkpoint "
                    "containing teacher_state_dict"
                )
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            if not isinstance(resume_payload, dict) or not isinstance(
                resume_payload.get("teacher_state_dict"), dict
            ):
                raise ValueError(
                    f"Resume checkpoint {resume_path!r} does not contain teacher_state_dict; "
                    "configure the original teacher checkpoint(s) to resume this legacy checkpoint"
                )
            teacher_state_keys = resume_payload["teacher_state_dict"].keys()
            teacher_indices = sorted(
                {
                    int(key.partition(".")[0])
                    for key in teacher_state_keys
                    if key.partition(".")[0].isdigit()
                }
            )
            if not teacher_indices or teacher_indices != list(range(len(teacher_indices))):
                raise ValueError(
                    f"Resume checkpoint {resume_path!r} has an invalid teacher_state_dict module layout"
                )
            checkpoints = _normalize_checkpoints(resume_payload.get("teacher_checkpoints"))
            if len(checkpoints) != len(teacher_indices):
                # Paths are descriptive metadata only when resuming from the
                # embedded teacher bank; they need not still exist.
                checkpoints = ["<embedded checkpoint>"] * len(teacher_indices)
            saved_labels = list(resume_payload.get("teacher_labels", []) or [])
            if teacher_labels and saved_labels != teacher_labels:
                raise ValueError(
                    f"Resume checkpoint teacher labels {saved_labels} do not match configured labels {teacher_labels}"
                )
            if not teacher_labels:
                teacher_labels = saved_labels
            if teacher_labels and len(teacher_labels) != len(teacher_indices):
                raise ValueError("Resume checkpoint teacher labels do not match its teacher bank size")
        routing_obs_group = algorithm_cfg.pop("routing_obs_group", None)
        share_cnn_encoders = algorithm_cfg.pop("share_cnn_encoders", False)

        default_sets = ["actor", "critic", "teacher"]
        if algorithm_cfg.get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        if routing_obs_group is not None and routing_obs_group not in obs:
            raise KeyError(f"routing observation group {routing_obs_group!r} is not present in the environment")
        algorithm_cfg = resolve_rnd_config(algorithm_cfg, obs, cfg["obs_groups"], env)
        algorithm_cfg = resolve_symmetry_config(algorithm_cfg, env)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        if share_cnn_encoders:
            critic_cfg["cnns"] = actor.cnns
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        teacher_obs_groups = {"teacher": list(cfg["obs_groups"]["teacher"])}
        base_teacher = teacher_class(obs, teacher_obs_groups, "teacher", env.num_actions, **teacher_cfg).to(device)
        teacher_count = len(teacher_indices) if resume_payload is not None else len(checkpoints)
        teachers = [base_teacher]
        teachers.extend(copy.deepcopy(base_teacher) for _ in range(teacher_count - 1))
        if resume_payload is None:
            for teacher, path in zip(teachers, checkpoints, strict=True):
                loaded = torch.load(path, map_location=device, weights_only=False)
                if not isinstance(loaded, dict):
                    raise ValueError(f"Teacher checkpoint must contain a state dictionary: {path}")
                load_actor_checkpoint(teacher, loaded, strict=True)
        for teacher in teachers:
            teacher.eval()
            teacher.requires_grad_(False)

        storage = RolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            store_privileged_actions=True,
        )
        algorithm = alg_class(
            actor,
            critic,
            storage,
            teachers=teachers,
            teacher_checkpoints=checkpoints,
            teacher_labels=teacher_labels,
            routing_obs_group=routing_obs_group,
            env=env,
            device=device,
            multi_gpu_cfg=cfg["multi_gpu"],
            **algorithm_cfg,
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm


def _normalize_checkpoints(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]
