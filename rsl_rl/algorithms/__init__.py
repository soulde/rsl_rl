# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .amp import AMP
from .distillation import Distillation
from .distillation_ppo import DistillationPPO
from .fast_sac import FastSAC
from .fast_sac import FastSAC
from .ppo import PPO
from .tdmpc2 import TDMPC2
from .supermimic import SuperMimic

__all__ = ["AMP", "PPO", "Distillation", "DistillationPPO", "FastSAC", "TDMPC2", "SuperMimic"]
__all__ = ["AMP", "FastSAC", "PPO", "Distillation", "TDMPC2"]
