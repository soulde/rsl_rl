"""Isaac Lab compatible DAgger runner with separate student and teacher observations."""

from __future__ import annotations

import os
import time
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.runners.dagger_runner import DaggerRunner
from rsl_rl.storage import BetaSchedule, DaggerStorage
from rsl_rl.utils import resolve_callable


class DaggerOnPolicyRunner:
    """Collect teacher labels on student-visited states and train a deterministic student policy."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Build the student and frozen teacher from the configured model classes."""
        self.env, self.cfg, self.device, self.log_dir = env, train_cfg, torch.device(device), log_dir
        self.num_envs = int(env.num_envs)
        obs = env.get_observations()
        self.obs_groups = train_cfg["obs_groups"]
        if "student" not in self.obs_groups or "teacher" not in self.obs_groups:
            raise ValueError("obs_groups must define 'student' and 'teacher' observation sets")
        self._validate_groups(obs)

        student_cfg = dict(train_cfg["student"])
        student_class = resolve_callable(student_cfg.pop("class_name"))
        _remove_legacy_model_fields(student_cfg)
        teacher_cfg = dict(train_cfg["teacher"])
        teacher_checkpoint = teacher_cfg.pop("checkpoint", None)
        teacher_class = resolve_callable(teacher_cfg.pop("class_name"))
        _remove_legacy_model_fields(teacher_cfg)
        model_groups = {"student": list(self.obs_groups["student"]), "teacher": list(self.obs_groups["teacher"])}
        self.student = student_class(obs, model_groups, "student", env.num_actions, **student_cfg).to(self.device)
        self.teacher = teacher_class(obs, model_groups, "teacher", env.num_actions, **teacher_cfg).to(self.device)
        if teacher_checkpoint:
            self._load_teacher(teacher_checkpoint)
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        algo_cfg = train_cfg["algorithm"]
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=float(algo_cfg.get("learning_rate", 1e-3)))
        self.storage = DaggerStorage(int(algo_cfg.get("storage_capacity", 32768)), device="cpu")
        self.alg = DaggerRunner(
            self._student_actions,
            self._teacher_actions,
            self.storage,
            BetaSchedule(**algo_cfg.get("beta", {})),
        )
        self.current_learning_iteration = 0
        self.writer = None
        self.git_status_repos: list[str] = []

    def _validate_groups(self, obs: TensorDict) -> None:
        for group_set in ("student", "teacher"):
            groups = self.obs_groups[group_set]
            missing = [key for key in groups if key not in obs]
            if not groups or missing:
                raise ValueError(f"Invalid {group_set} observation groups; missing keys: {missing}")

    def _select(self, obs: TensorDict, set_name: str) -> TensorDict:
        return obs.select(*self.obs_groups[set_name])

    def _student_actions(self, obs: TensorDict) -> torch.Tensor:
        return self.student(obs.to(self.device))

    def _teacher_actions(self, obs: TensorDict) -> torch.Tensor:
        return self.teacher(obs.to(self.device))

    def _load_teacher(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint.get("actor_state_dict", checkpoint.get("model_state_dict", checkpoint))
        if not isinstance(state, dict):
            raise ValueError(f"Teacher checkpoint {path} does not contain a model state dictionary")
        self.teacher.load_state_dict(state)

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
            metrics = self.alg.update(
                self.optimizer,
                batch_size=min(int(algo_cfg.get("batch_size", 256)), len(self.storage)),
                epochs=int(algo_cfg.get("updates_per_round", 1)),
            )
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
                "optimizer_state_dict": self.optimizer.state_dict(),
                "dagger_state_dict": self.alg.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict | None:
        """Restore selected state from a runner checkpoint."""
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        load_cfg = load_cfg or {"student": True, "optimizer": True, "dagger": True, "iteration": True}
        if load_cfg.get("student", True):
            self.student.load_state_dict(checkpoint["student_state_dict"], strict=strict)
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
