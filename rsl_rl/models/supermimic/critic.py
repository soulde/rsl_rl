"""Privileged invariant-state critic for imagined tracking trajectories."""

from torch import nn
from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import RunningNorm
from ..common import run_mlp


class SuperMimicCritic(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), mixed_precision=True):
        super().__init__()
        self.mixed_precision = mixed_precision
        self.norm = RunningNorm(input_dim)
        self.network = MLP(input_dim, 1, hidden_dims, activation="elu")

    def forward(self, features):
        return run_mlp(self.network, self.norm(features), self.mixed_precision and self.training).squeeze(-1)
