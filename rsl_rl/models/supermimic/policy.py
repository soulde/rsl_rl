"""Normalized motion encoder and shared history decoder for SuperMimic."""

import copy
import torch
from torch import nn
from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import RunningNorm
from ..common import run_mlp


class SuperMimicActor(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        goal_dim,
        history_length=10,
        hidden_dims=(2048, 1024, 512, 256),
        goal_latent_dim=32,
        encoder_hidden_dims=(2048, 1024, 512, 256),
        mixed_precision=True,
    ):
        super().__init__()
        self.observation_dim, self.action_dim, self.goal_dim = observation_dim, action_dim, goal_dim
        self.history_length = history_length
        self.mixed_precision = mixed_precision
        self.goal_encoder = MLP(goal_dim, goal_latent_dim, encoder_hidden_dims, activation="elu")
        self.network = MLP(
            observation_dim * history_length + goal_latent_dim, action_dim, hidden_dims, activation="elu"
        )
        self.obs_norm = RunningNorm(observation_dim)
        self.goal_norm = RunningNorm(goal_dim)
        with torch.no_grad():
            self.network[-1].weight.mul_(0.01)
            self.network[-1].bias.zero_()

    def forward(self, observation, goal):
        mixed = self.mixed_precision and self.training
        latent = torch.tanh(run_mlp(self.goal_encoder, self.goal_norm(goal), mixed))
        history = self.obs_norm(observation).flatten(start_dim=1)
        return run_mlp(self.network, torch.cat((latent, history), -1), mixed)

    def as_jit(self):
        return _SuperMimicActorExport(self)

    def as_onnx(self, verbose=False):
        return _SuperMimicActorExport(self)


class _SuperMimicActorExport(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.goal_encoder = copy.deepcopy(actor.goal_encoder)
        self.network = copy.deepcopy(actor.network)
        self.obs_norm = copy.deepcopy(actor.obs_norm)
        self.goal_norm = copy.deepcopy(actor.goal_norm)
        self.history_length, self.observation_dim, self.goal_dim = (
            actor.history_length,
            actor.observation_dim,
            actor.goal_dim,
        )
        self.input_names = ["observation_history", "goal"]
        self.output_names = ["actions"]
        self.eval()

    def forward(self, observation_history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        latent = torch.tanh(self.goal_encoder(self.goal_norm(goal)))
        history = self.obs_norm(observation_history).flatten(start_dim=1)
        return self.network(torch.cat((latent, history), -1))

    @torch.jit.ignore
    def get_dummy_inputs(self):
        return torch.zeros(1, self.history_length, self.observation_dim), torch.zeros(1, self.goal_dim)
