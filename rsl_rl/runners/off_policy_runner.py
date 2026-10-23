# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner for off-policy algorithms using the standard `rsl_rl` interfaces."""

from __future__ import annotations

import os
import time

import torch

from rsl_rl.env import VecEnv
from rsl_rl.utils import resolve_callable
from rsl_rl.utils.logger import Logger


class OffPolicyRunner:
    """Collect vectorized experience and schedule off-policy updates."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.cfg["algorithm"].setdefault("rnd_cfg", None)
        obs = self.env.get_observations()
        algorithm_class = resolve_callable(self.cfg["algorithm"]["class_name"])
        self.alg = algorithm_class.construct_algorithm(obs, self.env, self.cfg, self.device)
        if getattr(self.alg, "requires_reference_state_initialization", False):
            task = getattr(self.env, "unwrapped", None)
            reset = getattr(self.env, "reset", None)
            sample_rsi = getattr(self.alg, "sample_rsi", None)
            get_layout = getattr(self.alg, "get_rsi_joint_layout", None)
            if task is None or not callable(reset):
                raise RuntimeError(
                    "SuperMimic requires an Isaac Lab task exposing env.unwrapped and reset() for reference-state initialization"
                )
            if not callable(sample_rsi) or not callable(get_layout):
                raise RuntimeError(
                    "Reference-state initialization requires sample_rsi() and get_rsi_joint_layout() APIs"
                )
            task._supermimic_rsi_sampler = sample_rsi
            task._supermimic_rsi_joint_layout = get_layout()
            task._supermimic_num_reference_bins = int(self.alg.sampler.num_bins)
            reset()
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=False,
            gpu_world_size=1,
            gpu_global_rank=0,
            device=self.device,
        )
        self.current_learning_iteration = 0
        self.total_steps = 0

    def learn(self, num_learning_iterations: int) -> None:
        """Collect `num_steps_per_env` vectorized steps per learning iteration."""
        if num_learning_iterations < 0:
            raise ValueError("num_learning_iterations cannot be negative.")
        num_steps_per_env = int(self.cfg["num_steps_per_env"])
        if num_steps_per_env <= 0:
            raise ValueError("num_steps_per_env must be positive.")
        learning_starts = int(self.cfg["algorithm"].get("learning_starts", getattr(self.alg, "learning_starts", 0)))
        train_every = int(self.cfg["algorithm"].get("train_every", getattr(self.alg, "train_every", 1)))
        if learning_starts < 0 or train_every <= 0:
            raise ValueError("learning_starts cannot be negative and train_every must be positive.")

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        start_iteration = self.current_learning_iteration
        total_iterations = start_iteration + num_learning_iterations
        for iteration in range(start_iteration, total_iterations):
            start_time = time.time()
            loss_sums: dict[str, float] = {}
            update_count = 0
            for _ in range(num_steps_per_env):
                actions = self.alg.act(obs)
                next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                next_obs = next_obs.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                self.alg.process_env_step(next_obs, rewards, dones, extras)
                self.logger.process_env_step(rewards, dones, extras)
                self.total_steps += self.env.num_envs
                env_step = self.total_steps // self.env.num_envs
                if self.total_steps >= learning_starts and env_step % train_every == 0:
                    losses = self.alg.update()
                    for key, value in losses.items():
                        loss_sums[key] = loss_sums.get(key, 0.0) + float(value)
                    update_count += 1
                obs = next_obs

            elapsed = time.time() - start_time
            losses = {key: value / max(update_count, 1) for key, value in loss_sums.items()}
            self.current_learning_iteration = iteration
            self.logger.log(
                it=iteration,
                start_it=start_iteration,
                total_it=total_iterations,
                collect_time=elapsed,
                learn_time=0.0,
                loss_dict=losses,
                learning_rate=float(getattr(self.alg, "learning_rate", 0.0)),
                action_std=torch.zeros(1, device=self.device),
                rnd_weight=None,
            )
            save_interval = int(self.cfg.get("save_interval", 0))
            if self.logger.writer is not None and save_interval > 0 and iteration % save_interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{iteration}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore[arg-type]
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the algorithm state and current runner iteration."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict | None:
        """Restore algorithm and runner state from a checkpoint."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = int(loaded_dict["iter"])
        self.total_steps = int(loaded_dict.get("num_steps", self.total_steps))
        return loaded_dict.get("infos")

    def get_inference_policy(self, device: str | None = None):
        """Return the algorithm's deterministic policy in evaluation mode."""
        self.alg.eval_mode()
        return self.alg.get_policy().to(device or self.device)
