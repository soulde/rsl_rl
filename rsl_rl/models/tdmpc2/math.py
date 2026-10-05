# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distributional and Gaussian math helpers for TD-MPC2."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symlog(values: torch.Tensor) -> torch.Tensor:
    """Apply the symmetric logarithm transform."""
    return values.sign() * torch.log1p(values.abs())


def symexp(values: torch.Tensor) -> torch.Tensor:
    """Invert :func:`symlog`."""
    return values.sign() * torch.expm1(values.abs())


def two_hot(values: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Encode scalar targets as linearly interpolated symlog support bins."""
    num_bins = int(cfg["num_bins"])
    if num_bins == 1:
        return symlog(values)
    if num_bins < 1:
        return values
    vmin, vmax = float(cfg["vmin"]), float(cfg["vmax"])
    if vmin >= vmax:
        raise ValueError("TD-MPC2 value support requires vmin < vmax.")
    transformed = symlog(values.reshape(-1)).clamp(vmin, vmax)
    spacing = (vmax - vmin) / (num_bins - 1)
    positions = (transformed - vmin) / spacing
    lower = positions.floor().long()
    upper = (lower + 1).clamp(max=num_bins - 1)
    upper_weight = positions - lower
    upper_weight = torch.where(lower == upper, torch.zeros_like(upper_weight), upper_weight)
    encoded = torch.zeros(transformed.shape[0], num_bins, dtype=values.dtype, device=values.device)
    encoded.scatter_add_(1, lower.unsqueeze(-1), (1.0 - upper_weight).unsqueeze(-1))
    encoded.scatter_add_(1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return encoded


def two_hot_inv(logits: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Decode categorical logits to scalar values via expected symlog support."""
    num_bins = int(cfg["num_bins"])
    if num_bins == 1:
        return symexp(logits)
    if num_bins < 1:
        return logits
    support = torch.linspace(cfg["vmin"], cfg["vmax"], num_bins, dtype=logits.dtype, device=logits.device)
    return symexp((logits.softmax(dim=-1) * support).sum(dim=-1, keepdim=True))


def soft_ce(logits: torch.Tensor, target: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Return cross entropy against a scalar or distributional target."""
    target_distribution = two_hot(target, cfg)
    return -(target_distribution * F.log_softmax(logits, dim=-1)).sum(dim=-1, keepdim=True)


def bounded_log_std(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Map raw policy output into the configured log standard-deviation range."""
    return low + 0.5 * (high - low) * (torch.tanh(values) + 1.0)


def gaussian_logprob(noise: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Compute diagonal Gaussian log probability for standard-normal noise."""
    return (-0.5 * noise.square() - log_std - 0.9189385175704956).sum(dim=-1, keepdim=True)
