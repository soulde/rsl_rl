"""Isaac Lab compatible DAgger runner with separate student and teacher observations."""

from __future__ import annotations

import copy
import os
import time
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.runners.dagger_runner import DaggerRunner
from rsl_rl.storage import BetaSchedule, DaggerStorage
from rsl_rl.utils import load_actor_checkpoint, resolve_callable

from .cnn_export import export_cnn_policy_pair


class DaggerOnPolicyRunner:
    """Collect teacher labels on student-visited states and train a deterministic student policy."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Build the student and frozen teacher from the configured model classes."""
        self.env, self.cfg, self.device, self.log_dir = env, train_cfg, torch.device(device), log_dir
        self.num_envs = int(env.num_envs)
        obs = env.get_observations()
        self.obs_groups = train_cfg["obs_groups"]
        self.routing_obs_group = train_cfg.get("algorithm", {}).get("routing_obs_group")
        if "student" not in self.obs_groups or "teacher" not in self.obs_groups:
            raise ValueError("obs_groups must define 'student' and 'teacher' observation sets")
        self._validate_groups(obs)

        student_cfg = dict(train_cfg["student"])
        student_class = resolve_callable(student_cfg.pop("class_name"))
        _remove_legacy_model_fields(student_cfg)
        teacher_cfg = dict(train_cfg["teacher"])
        teacher_checkpoints = _normalize_checkpoints(
            teacher_cfg.pop("checkpoints", None), teacher_cfg.pop("checkpoint", None)
        )
        teacher_class = resolve_callable(teacher_cfg.pop("class_name"))
        _remove_legacy_model_fields(teacher_cfg)
        model_groups = {"student": list(self.obs_groups["student"]), "teacher": list(self.obs_groups["teacher"])}
        self.student = student_class(obs, model_groups, "student", env.num_actions, **student_cfg).to(self.device)
        base_teacher = teacher_class(obs, model_groups, "teacher", env.num_actions, **teacher_cfg).to(self.device)
        self.teachers = torch.nn.ModuleList(
            [base_teacher, *(copy.deepcopy(base_teacher) for _ in teacher_checkpoints[1:])]
        )
        for teacher, teacher_checkpoint in zip(self.teachers, teacher_checkpoints):
            self._load_teacher(teacher, teacher_checkpoint)
        for teacher in self.teachers:
            teacher.eval()
            teacher.requires_grad_(False)
        # Keep the historical attribute for downstream integrations that
        # inspect the single-teacher DAgger runner.
        self.teacher = self.teachers[0]
        if self.routing_obs_group is not None:
            if self.routing_obs_group not in self.obs_groups:
                raise ValueError(
                    f"routing_obs_group {self.routing_obs_group!r} must name an entry in obs_groups"
                )
            if len(self.teachers) > 1 and len(teacher_checkpoints) != len(self.teachers):
                raise ValueError("each routed DAgger teacher must have a checkpoint")

        algo_cfg = train_cfg["algorithm"]
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=float(algo_cfg.get("learning_rate", 1e-3)))
        self.storage = DaggerStorage(int(algo_cfg.get("storage_capacity", 32768)), device="cpu")
        self.alg = DaggerRunner(
            self._student_actions,
            self._teacher_actions,
            self.storage,
            BetaSchedule(**algo_cfg.get("beta", {})),
            skip_beta_sampling_at_endpoints=bool(algo_cfg.get("skip_beta_sampling_at_endpoints", False)),
        )
        self.student_stochastic_output = bool(algo_cfg.get("student_stochastic_output", False))
        self.current_learning_iteration = 0
        self.writer = None
        self.git_status_repos: list[str] = []

    def _validate_groups(self, obs: TensorDict) -> None:
        group_sets = ["student", "teacher"]
        if self.routing_obs_group is not None:
            group_sets.append(self.routing_obs_group)
        for group_set in group_sets:
            if group_set not in self.obs_groups:
                raise ValueError(f"obs_groups does not define required set {group_set!r}")
            groups = self.obs_groups[group_set]
            missing = [key for key in groups if key not in obs]
            if not groups or missing:
                raise ValueError(f"Invalid {group_set} observation groups; missing keys: {missing}")
        if self.routing_obs_group is not None and len(self.obs_groups[self.routing_obs_group]) != 1:
            raise ValueError("the DAgger routing observation group must contain one observation key")

    def _select(self, obs: TensorDict, set_name: str) -> TensorDict:
        keys = list(self.obs_groups[set_name])
        if set_name == "teacher" and self.routing_obs_group is not None:
            keys.extend(self.obs_groups[self.routing_obs_group])
        return obs.select(*keys)

    def _student_actions(self, obs: TensorDict) -> torch.Tensor:
        obs = obs.to(self.device)
        if self.student_stochastic_output:
            return self.student(obs, stochastic_output=True)
        return self.student(obs)

    def _student_mean_actions(self, obs: TensorDict) -> torch.Tensor:
        return self.student(obs.to(self.device), stochastic_output=False)

    def _teacher_actions(self, obs: TensorDict) -> torch.Tensor:
        obs = obs.to(self.device)
        if self.routing_obs_group is None:
            return self.teacher(obs)

        route_key = self.obs_groups[self.routing_obs_group][0]
        route_values = obs[route_key].reshape(obs.batch_size[0], -1)
        if route_values.shape[1] != 1:
            raise ValueError(f"DAgger routing observations must be scalar, got {route_values.shape[1]} values")
        route = route_values[:, 0].long()
        if torch.any(route_values[:, 0] != route.to(route_values.dtype)):
            raise ValueError("DAgger teacher routing indices must be integer-valued")
        if torch.any(route >= len(self.teachers)):
            raise ValueError(f"DAgger teacher routing index exceeds teacher count ({len(self.teachers)})")
        teacher_obs = obs.select(*self.obs_groups["teacher"])
        all_actions = torch.stack([teacher(teacher_obs) for teacher in self.teachers], dim=0)
        active = route >= 0
        safe_route = route.clamp(min=0)
        gather_index = safe_route.view(1, -1, 1).expand(1, -1, all_actions.shape[-1])
        labels = all_actions.gather(0, gather_index).squeeze(0)
        return labels.masked_fill(~active.unsqueeze(-1), 0.0)

    def _load_teacher(self, teacher: MLPModel, path: str | None) -> None:
        if path is None:
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        load_actor_checkpoint(teacher, checkpoint, strict=True)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Collect and optimize the student for the requested number of rounds."""
        if num_learning_iterations < 0:
            raise ValueError("num_learning_iterations must be non-negative")
        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf"):
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        initial_obs = self.env.get_observations()
        student_obs, teacher_obs = self._select(initial_obs, "student"), self._select(initial_obs, "teacher")
        self.student.train()
        self.teacher.eval()
        if self.log_dir is not None and self.writer is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            if self.git_status_repos:
                self.writer.add_text("Git/repositories", "\n".join(self.git_status_repos), 0)
        algo_cfg = self.cfg["algorithm"]
        end_iteration = self.current_learning_iteration + num_learning_iterations
        for iteration in range(self.current_learning_iteration, end_iteration):
            start = time.perf_counter()
            student_obs, teacher_obs = self.alg.collect(
                student_obs,
                teacher_obs,
                self._env_step,
                num_steps=int(algo_cfg.get("collection_steps", self.cfg["num_steps_per_env"])),
                round_index=self.alg.round_index,
            )
            collect_time = time.perf_counter() - start
            update_start = time.perf_counter()
            has_epoch_minibatch_cfg = "num_mini_batches" in algo_cfg and "num_learning_epochs" in algo_cfg
            if ("num_mini_batches" in algo_cfg) != ("num_learning_epochs" in algo_cfg):
                raise ValueError("num_mini_batches and num_learning_epochs must be configured together")
            if has_epoch_minibatch_cfg:
                metrics = self.alg.update_minibatches(
                    self.optimizer,
                    student_policy=self._student_mean_actions,
                    num_mini_batches=int(algo_cfg["num_mini_batches"]),
                    num_epochs=int(algo_cfg.get("num_learning_epochs", 1)),
                    max_grad_norm=algo_cfg.get("max_grad_norm"),
                    parameters=self.student.parameters(),
                )
            else:
                metrics = self.alg.update(
                    self.optimizer,
                    batch_size=min(int(algo_cfg.get("batch_size", 256)), len(self.storage)),
                    epochs=int(algo_cfg.get("updates_per_round", 1)),
                )
            if algo_cfg.get("clear_storage_after_update", False):
                self.storage.clear()
            self.current_learning_iteration = iteration + 1
            if self.writer is not None:
                self.writer.add_scalar("Loss/behavior_cloning", metrics["loss"], iteration + 1)
                self.writer.add_scalar("DAgger/beta", self.alg.beta_schedule(iteration), iteration + 1)
                self.writer.add_scalar("Performance/collection_seconds", collect_time, iteration + 1)
                self.writer.add_scalar("Performance/update_seconds", time.perf_counter() - update_start, iteration + 1)
            if self.log_dir is not None and (iteration + 1) % int(self.cfg.get("save_interval", 100)) == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration + 1}.pt"))
            print(
                f"DAgger iteration {iteration + 1}: loss={metrics['loss']:.6f}, "
                f"collect={collect_time:.2f}s, update={time.perf_counter() - update_start:.2f}s"
            )
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def _env_step(self, actions: torch.Tensor) -> tuple[TensorDict, TensorDict, torch.Tensor]:
        observations, _rewards, dones, _extras = self.env.step(actions.to(self.env.device))
        return self._select(observations, "student"), self._select(observations, "teacher"), dones

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save student, optimizer, DAgger schedule state, and iteration."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "student_state_dict": self.student.state_dict(),
                "teacher_state_dict": self.teacher.state_dict(),
                "teacher_state_dicts": [teacher.state_dict() for teacher in self.teachers],
                "optimizer_state_dict": self.optimizer.state_dict(),
                "dagger_state_dict": self.alg.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )
        if self.cfg.get("export_policy_on_save", False):
            try:
                export_cnn_policy_pair(
                    self.student, path, opset_version=self.cfg.get("onnx_export_opset_version", 11)
                )
            except Exception as error:
                print(f"[WARN] Failed to export split CNN policy for {path}: {error}")

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict | None:
        """Restore selected state from a runner checkpoint."""
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        load_cfg = load_cfg or {"student": True, "teacher": True, "optimizer": True, "dagger": True, "iteration": True}
        if load_cfg.get("student", True):
            self.student.load_state_dict(checkpoint["student_state_dict"], strict=strict)
        if load_cfg.get("teacher", True):
            if "teacher_state_dicts" in checkpoint:
                saved_teachers = checkpoint["teacher_state_dicts"]
                if len(saved_teachers) != len(self.teachers):
                    raise ValueError(
                        f"checkpoint has {len(saved_teachers)} teachers, but this config has {len(self.teachers)}"
                    )
                for teacher, state in zip(self.teachers, saved_teachers, strict=True):
                    teacher.load_state_dict(state, strict=strict)
            elif "teacher_state_dict" in checkpoint:
                self.teacher.load_state_dict(checkpoint["teacher_state_dict"], strict=strict)
        if load_cfg.get("optimizer", True) and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_cfg.get("dagger", True) and "dagger_state_dict" in checkpoint:
            self.alg.load_state_dict(checkpoint["dagger_state_dict"])
        if load_cfg.get("iteration", True):
            self.current_learning_iteration = int(checkpoint.get("iter", 0))
        return checkpoint.get("infos")

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the student in evaluation mode on the requested device."""
        self.student.eval()
        return self.student.to(device or self.device)

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Record repository paths for the training launcher logging interface."""
        self.git_status_repos.append(repo_file_path)

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the deterministic student policy as a TorchScript module."""
        os.makedirs(path, exist_ok=True)
        model = self.student.as_jit().to("cpu")
        torch.jit.script(model).save(os.path.join(path, filename))

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the deterministic student policy in the RSL-RL ONNX format."""
        os.makedirs(path, exist_ok=True)
        model = self.student.as_onnx(verbose=verbose).to("cpu")
        model.eval()
        torch.onnx.export(
            model,
            model.get_dummy_inputs(),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=model.input_names,
            output_names=model.output_names,
        )


def _remove_legacy_model_fields(model_cfg: dict) -> None:
    """Drop Isaac Lab's PPO-only compatibility fields before constructing a model."""
    for key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
        model_cfg.pop(key, None)


def _normalize_checkpoints(multiple: object, single: object) -> list[str | None]:
    value = multiple if multiple else single
    if value is None:
        return [None]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError("teacher checkpoint(s) must be a path string or a sequence of paths")
