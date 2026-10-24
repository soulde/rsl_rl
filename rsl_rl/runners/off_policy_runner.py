# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runner for off-policy algorithms using the standard `rsl_rl` interfaces."""

from __future__ import annotations

import math
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
        initialize_env = getattr(self.alg, "initialize_env", None)
        if callable(initialize_env):
            initialize_env(self.env)
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
        """Run the shared warmup and UTD schedule for off-policy algorithms."""
        if num_learning_iterations < 0:
            raise ValueError("num_learning_iterations cannot be negative.")
        if num_learning_iterations == 0:
            return
        utd = float(self.alg.utd)
        warmup_steps = int(self.alg.warmup_steps)
        if not math.isfinite(utd) or utd <= 0 or warmup_steps < 0:
            raise ValueError("Algorithm utd must be finite and positive and warmup_steps cannot be negative.")

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        self.logger.cfg = dict(self.cfg)

        def collect(count):
            nonlocal obs
            for _ in range(count):
                actions = self.alg.act(obs)
                next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                next_obs = next_obs.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                self.alg.process_env_step(next_obs, rewards, dones, extras)
                self.logger.process_env_step(rewards, dones, extras)
                self.total_steps += self.env.num_envs
                obs = next_obs

        warmup_collected = int(self.alg.warmup_steps_collected)
        if warmup_collected < warmup_steps:
            collect(warmup_steps - warmup_collected)
        start = self.current_learning_iteration
        end = start + num_learning_iterations
        for iteration in range(start, end):
            begin = time.time()
            collection_steps = math.floor((iteration + 1) / utd) - math.floor(iteration / utd)
            collect(collection_steps)
            collect_time = time.time() - begin
            update_begin = time.time()
            losses = self.alg.update()
            learn_time = time.time() - update_begin
            self.logger.cfg["num_steps_per_env"] = collection_steps
            self.logger.tot_timesteps = self.total_steps - collection_steps * self.env.num_envs
            self.current_learning_iteration = iteration + 1
            self.logger.log(
                it=iteration,
                start_it=start,
                total_it=end,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=losses,
                learning_rate=float(getattr(self.alg, "learning_rate", 0.0)),
                action_std=torch.zeros(1, device=self.device),
                rnd_weight=None,
            )
            interval = int(self.cfg.get("save_interval", 0))
            checkpoint_iteration = iteration + 1
            if self.logger.writer is not None and interval > 0 and checkpoint_iteration % interval == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{checkpoint_iteration}.pt"))
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
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

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the algorithm policy using its deployment interface."""
        model = self.alg.get_policy().as_jit().cpu().eval()
        os.makedirs(path, exist_ok=True)
        torch.jit.script(model).save(os.path.join(path, filename))

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export an explicit-history policy with a dynamic batch dimension."""
        model = self.alg.get_policy().as_onnx(verbose=verbose).cpu().eval()
        os.makedirs(path, exist_ok=True)
        torch.onnx.export(
            model,
            model.get_dummy_inputs(),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=model.input_names,
            output_names=model.output_names,
            dynamic_axes={name: {0: "batch"} for name in model.input_names + model.output_names},
            dynamo=False,
        )
