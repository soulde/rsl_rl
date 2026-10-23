# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TD-MPC2 network initialization helpers."""

import torch.nn as nn


def weight_init(module: nn.Module) -> None:
    """Initialize linear layers with the TD-MPC2 truncated normal scheme."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def zero_(parameters: list[nn.Parameter]) -> None:
    """Set the supplied parameters to zero in place."""
    for parameter in parameters:
        parameter.data.zero_()
