"""Simulator-independent conditional variational autoencoder models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rsl_rl.modules import MLP


@dataclass
class VAEOutput:
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor


class ConditionalVAE(nn.Module):
    """A compact conditional VAE operating on the last tensor dimension.

    The model intentionally has no environment or dataset dependencies. Inputs may
    have arbitrary leading dimensions (for example batch and time); conditioning
    is concatenated on the last dimension and can be omitted when ``condition_dim``
    is zero.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        condition_dim: int = 0,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0 or condition_dim < 0:
            raise ValueError("input_dim and latent_dim must be positive; condition_dim cannot be negative")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.beta = beta
        encoder_input = input_dim + condition_dim
        self.encoder = MLP(encoder_input, hidden_dims[-1], hidden_dims[:-1] or (hidden_dims[-1],))
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)
        self.decoder = MLP(latent_dim + condition_dim, input_dim, hidden_dims[::-1] or (hidden_dims[-1],))

    def _condition(self, x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if self.condition_dim == 0:
            if condition is not None:
                raise ValueError("condition must be None when condition_dim=0")
            return x.new_empty(*x.shape[:-1], 0)
        if condition is None or condition.shape[-1] != self.condition_dim:
            raise ValueError(f"condition must have last dimension {self.condition_dim}")
        while condition.ndim < x.ndim:
            condition = condition.unsqueeze(-2)
        return condition.expand(*x.shape[:-1], self.condition_dim)

    def encode(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"x must have last dimension {self.input_dim}")
        flat = x.reshape(-1, self.input_dim)
        cond = self._condition(x, condition).reshape(-1, self.condition_dim)
        hidden = self.encoder(torch.cat((flat, cond), dim=-1))
        return (
            self.mu(hidden).reshape(*x.shape[:-1], self.latent_dim),
            self.logvar(hidden).reshape(*x.shape[:-1], self.latent_dim),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if not sample:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"z must have last dimension {self.latent_dim}")
        cond = self._condition(z, condition).reshape(-1, self.condition_dim)
        decoded = self.decoder(torch.cat((z.reshape(-1, self.latent_dim), cond), dim=-1))
        return decoded.reshape(*z.shape[:-1], self.input_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None, sample: bool = True) -> VAEOutput:
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar, sample=sample)
        return VAEOutput(self.decode(z, condition), mu, logvar, z)

    def loss(self, target: torch.Tensor, output: VAEOutput, reduction: str = "mean") -> dict[str, torch.Tensor]:
        reconstruction = (output.reconstruction - target).pow(2)
        reconstruction = reconstruction.mean(dim=-1)
        kl = -0.5 * (1 + output.logvar - output.mu.pow(2) - output.logvar.exp()).sum(dim=-1)
        if reduction == "mean":
            reconstruction, kl = reconstruction.mean(), kl.mean()
        elif reduction != "none":
            raise ValueError("reduction must be 'mean' or 'none'")
        return {"reconstruction": reconstruction, "kl": kl, "total": reconstruction + self.beta * kl}
