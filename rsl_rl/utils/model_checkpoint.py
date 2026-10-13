# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for loading actor weights from common RL checkpoint formats."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


_ACTOR_STATE_KEYS = ("actor_state_dict", "actor_model_state_dict", "model_state_dict", "state_dict", "model")
_ACTOR_PREFIXES = ("module.", "actor_module.module.", "actor_module.", "actor.")


def load_actor_checkpoint(model: nn.Module, checkpoint: Mapping, *, strict: bool = True) -> None:
    """Load an actor model from RSL-RL or Holosoma/PHP checkpoint layouts.

    Holosoma PPO checkpoints store the actor under ``actor_model_state_dict``
    with an ``actor_module.module.`` prefix and a direct ``std`` parameter.
    RSL-RL models use ``mlp.*`` and ``distribution.std_param`` respectively.
    """
    state = checkpoint
    for key in _ACTOR_STATE_KEYS:
        value = state.get(key) if isinstance(state, Mapping) else None
        if isinstance(value, Mapping):
            state = value
            break
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint does not contain an actor model state dictionary")

    target_keys = set(model.state_dict())
    normalized = {}
    for source_key, value in state.items():
        if not isinstance(source_key, str) or not isinstance(value, torch.Tensor):
            continue
        key = source_key
        changed = True
        while changed:
            changed = False
            for prefix in _ACTOR_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break
        candidates = [key]
        if not key.startswith("mlp."):
            candidates.append(f"mlp.{key}")
        if key == "std":
            candidates.append("distribution.std_param")
        elif key == "log_std":
            candidates.append("distribution.log_std_param")
        matches = [candidate for candidate in candidates if candidate in target_keys]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous actor checkpoint key {source_key!r}: {matches}")
        if matches:
            target_key = matches[0]
            if target_key in normalized:
                raise ValueError(f"Multiple checkpoint entries map to actor parameter {target_key!r}")
            normalized[target_key] = value

    if not normalized:
        raise ValueError("Checkpoint does not contain actor parameters compatible with the configured model")
    model.load_state_dict(normalized, strict=strict)

    normalizer_state = checkpoint.get("actor_obs_normalizer_state_dict")
    normalizer = getattr(model, "obs_normalizer", None)
    if normalizer_state is not None and normalizer is not None and not isinstance(normalizer, nn.Identity):
        normalizer.load_state_dict(normalizer_state, strict=strict)
