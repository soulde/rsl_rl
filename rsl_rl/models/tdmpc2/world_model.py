# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TD-MPC2 implicit world model for vectorized state observations."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from rsl_rl.models.tdmpc2 import init, layers, math


class TDMPC2WorldModel(nn.Module):
    """Encode state observations and predict latent dynamics, reward, and value."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: dict) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("TD-MPC2 observation and action dimensions must be positive.")
        self.cfg = {
            "latent_dim": 512,
            "enc_dim": 256,
            "mlp_dim": 512,
            "num_enc_layers": 2,
            "num_q": 5,
            "num_bins": 101,
            "vmin": -10.0,
            "vmax": 10.0,
            "simnorm_dim": 8,
            "dropout": 0.01,
            "log_std_min": -10.0,
            "log_std_max": 2.0,
            "tau": 0.01,
            "episodic": False,
        }
        self.cfg.update(cfg)
        if self.cfg["num_q"] <= 0:
            raise ValueError("TD-MPC2 requires at least one Q function.")
        if self.cfg["num_bins"] < 2:
            raise ValueError("TD-MPC2 categorical reward/value support requires at least two bins.")
        if self.cfg["num_bins"] > 1 and self.cfg["vmin"] >= self.cfg["vmax"]:
            raise ValueError("TD-MPC2 requires vmin < vmax.")
        latent_dim = int(self.cfg["latent_dim"])
        mlp_dim = int(self.cfg["mlp_dim"])
        simnorm_dim = int(self.cfg["simnorm_dim"])
        if latent_dim % simnorm_dim:
            raise ValueError("TD-MPC2 latent_dim must be divisible by simnorm_dim.")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder = layers.encoder(
            obs_dim, int(self.cfg["enc_dim"]), latent_dim, int(self.cfg["num_enc_layers"]), simnorm_dim
        )
        self.dynamics = layers.mlp(
            latent_dim + action_dim,
            [mlp_dim, mlp_dim],
            latent_dim,
            activation=layers.SimNorm(simnorm_dim),
        )
        num_bins = max(int(self.cfg["num_bins"]), 1)
        self.reward_head = layers.mlp(latent_dim + action_dim, [mlp_dim, mlp_dim], num_bins)
        self.termination_head = (
            layers.mlp(latent_dim, [mlp_dim, mlp_dim], 1) if self.cfg["episodic"] else None
        )
        self.policy_head = layers.mlp(latent_dim, [mlp_dim, mlp_dim], 2 * action_dim)
        self.q_networks = nn.ModuleList(
            [layers.mlp(latent_dim + action_dim, [mlp_dim, mlp_dim], num_bins, dropout=float(self.cfg["dropout"]))
             for _ in range(int(self.cfg["num_q"]))]
        )
        self.apply(init.weight_init)
        init.zero_([self.reward_head[-1].weight])
        self.target_q = copy.deepcopy(self.q_networks)
        for parameter in self.target_q.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("log_std_min", torch.tensor(float(self.cfg["log_std_min"])))
        self.register_buffer("log_std_max", torch.tensor(float(self.cfg["log_std_max"])))

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Map flat state observations to normalized latent states."""
        if observations.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected observation dimension {self.obs_dim}, got {observations.shape[-1]}.")
        return self.encoder(observations.float())

    def next(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict the next latent state."""
        return self.dynamics(torch.cat((latent, actions), dim=-1))

    def reward(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict categorical reward logits."""
        return self.reward_head(torch.cat((latent, actions), dim=-1))

    def termination(self, latent: torch.Tensor, unnormalized: bool = False) -> torch.Tensor:
        """Predict termination probability, or ones when episodic learning is disabled."""
        if self.termination_head is None:
            return torch.ones(*latent.shape[:-1], 1, device=latent.device, dtype=latent.dtype)
        logits = self.termination_head(latent)
        return logits if unnormalized else torch.sigmoid(logits)

    def pi(self, latent: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sample or return the mean action from the latent Gaussian policy prior."""
        mean, raw_log_std = self.policy_head(latent).chunk(2, dim=-1)
        log_std = math.bounded_log_std(raw_log_std, self.log_std_min, self.log_std_max)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros_like(mean[..., :1])
        else:
            noise = torch.randn_like(mean)
            action = torch.tanh(mean + noise * log_std.exp())
            log_prob = math.gaussian_logprob(noise, log_std) - torch.log(1.0 - action.square() + 1e-6).sum(
                dim=-1, keepdim=True
            )
        return action, {"mean": torch.tanh(mean), "log_std": log_std, "log_prob": log_prob}

    def Q(
        self,
        latent: torch.Tensor,
        actions: torch.Tensor,
        return_type: str = "min",
        target: bool = False,
    ) -> torch.Tensor:
        """Return ensemble Q logits or decoded values."""
        if return_type not in {"min", "avg", "all"}:
            raise ValueError("return_type must be one of 'min', 'avg', or 'all'.")
        networks = self.target_q if target else self.q_networks
        inputs = torch.cat((latent, actions), dim=-1)
        logits = torch.stack([network(inputs) for network in networks])
        if return_type == "all":
            return logits
        values = torch.stack([math.two_hot_inv(value, self.cfg) for value in logits])
        return values.min(dim=0).values if return_type == "min" else values.mean(dim=0)

    def soft_update_target_q(self) -> None:
        """Polyak-average online Q parameters into the target ensemble."""
        with torch.no_grad():
            for target, source in zip(self.target_q.parameters(), self.q_networks.parameters()):
                target.lerp_(source, float(self.cfg["tau"]))

    def train(self, mode: bool = True) -> TDMPC2WorldModel:
        """Set module training mode while keeping the target ensemble in evaluation mode."""
        super().train(mode)
        self.target_q.eval()
        return self
