"""Differentiable state-space dynamics used by SuperMimic."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rsl_rl.storage import SuperMimicSequenceBatch


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(width, hidden_dim), nn.ELU()))
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class SuperMimicWorldModel(nn.Module):
    """Predict next state, reward, clean observation, and termination from transitions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        observation_dim: int,
        context_dim: int = 0,
        goal_dim: int = 0,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
    ) -> None:
        super().__init__()
        if min(state_dim, action_dim, observation_dim) <= 0 or min(context_dim, goal_dim) < 0:
            raise ValueError(
                "World-model state/action/observation dimensions must be positive; context/goal nonnegative."
            )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.context_dim = context_dim
        self.goal_dim = goal_dim
        input_dim = state_dim + action_dim + context_dim + goal_dim
        self.trunk = _mlp(input_dim, hidden_dims, hidden_dims[-1] if hidden_dims else input_dim)
        feature_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.next_state_head = nn.Linear(feature_dim, state_dim)
        self.reward_head = nn.Linear(feature_dim, 1)
        self.clean_observation_head = nn.Linear(feature_dim, observation_dim)
        self.termination_head = nn.Linear(feature_dim, 1)

    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict one step while keeping gradients through every model operation."""
        context = self._optional_input(context, state, self.context_dim, "context")
        goal = self._optional_input(goal, state, self.goal_dim, "goal")
        features = torch.cat((state, action, context, goal), dim=-1)
        latent = self.trunk(features)
        return {
            # Residual dynamics make the learned model predict a state delta,
            # matching the reference implementation and easing long rollouts.
            "next_state": state + self.next_state_head(latent),
            "reward": self.reward_head(latent),
            "clean_observation": self.clean_observation_head(latent),
            "termination_logit": self.termination_head(latent),
        }

    def loss(self, batch: SuperMimicSequenceBatch) -> dict[str, torch.Tensor]:
        """Compute one-step supervised losses over all transitions in a sequence batch."""
        batch_steps = batch.actions.shape[0] * batch.actions.shape[1]
        state = batch.states[:, :-1].reshape(batch_steps, self.state_dim)
        action = batch.actions.reshape(batch_steps, self.action_dim)
        context = batch.contexts.reshape(batch_steps, self.context_dim)
        goal = batch.goals.reshape(batch_steps, self.goal_dim)
        predictions = self.predict(state, action, context, goal)
        next_state = batch.states[:, 1:].reshape(batch_steps, self.state_dim)
        reward = batch.rewards.reshape(batch_steps, 1)
        clean_observation = batch.clean_observations[:, 1:].reshape(batch_steps, self.observation_dim)
        terminated = batch.terminated.reshape(batch_steps, 1).to(dtype=torch.float32)
        mask = batch.mask.reshape(batch_steps, 1).to(dtype=state.dtype)
        denom = mask.sum().clamp_min(1.0)

        def masked_mean(value: torch.Tensor) -> torch.Tensor:
            return (value * mask).sum() / (denom * value.shape[-1])

        state_loss = masked_mean((predictions["next_state"] - next_state).square())
        reward_loss = masked_mean((predictions["reward"] - reward).square())
        clean_observation_loss = masked_mean((predictions["clean_observation"] - clean_observation).square())
        termination_loss = masked_mean(
            F.binary_cross_entropy_with_logits(predictions["termination_logit"], terminated, reduction="none")
        )
        return {
            "total_loss": state_loss + reward_loss + clean_observation_loss + termination_loss,
            "state_loss": state_loss,
            "reward_loss": reward_loss,
            "clean_observation_loss": clean_observation_loss,
            "termination_loss": termination_loss,
        }

    @staticmethod
    def _optional_input(value: torch.Tensor | None, reference: torch.Tensor, width: int, name: str) -> torch.Tensor:
        if value is None:
            if width:
                raise ValueError(f"{name} input is required when {name}_dim is nonzero")
            return reference.new_zeros((*reference.shape[:-1], 0))
        if value.shape[:-1] != reference.shape[:-1] or value.shape[-1] != width:
            raise ValueError(f"{name} input must have shape {(*reference.shape[:-1], width)}")
        return value
