"""Simulator-independent conditional diffusion building blocks."""

from __future__ import annotations

import math

import torch
from torch import nn

from rsl_rl.modules import MLP


class DiffusionSchedule(nn.Module):
    """Linear beta schedule and forward/reverse diffusion utilities."""

    def __init__(self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        if num_steps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("require num_steps >= 2 and 0 < beta_start < beta_end < 1")
        betas = torch.linspace(beta_start, beta_end, num_steps)
        alphas = 1.0 - betas
        self.num_steps = num_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

    def add_noise(
        self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0) if noise is None else noise
        if timesteps.ndim != 1 or timesteps.shape[0] != x0.shape[0]:
            raise ValueError("timesteps must be one value per batch element")
        shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
        noisy = self.sqrt_alphas_cumprod[timesteps].reshape(shape) * x0
        noisy = noisy + self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(shape) * noise
        return noisy, noise


class ConditionalDiffusionModel(nn.Module):
    """Conditional epsilon/x0 predictor with a DDPM sampler."""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int = 0,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        num_steps: int = 1000,
        prediction_type: str = "epsilon",
    ) -> None:
        super().__init__()
        if prediction_type not in {"epsilon", "x0"}:
            raise ValueError("prediction_type must be 'epsilon' or 'x0'")
        self.input_dim, self.condition_dim, self.prediction_type = input_dim, condition_dim, prediction_type
        self.schedule = DiffusionSchedule(num_steps)
        time_dim = hidden_dims[0]
        self.time_projection = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU())
        self.denoiser = MLP(input_dim + condition_dim + time_dim, input_dim, hidden_dims)

    def _time_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
        angles = timesteps.float().unsqueeze(-1) * frequencies
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))

    def forward(
        self, x_t: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x_t.shape[-1] != self.input_dim or timesteps.shape[0] != x_t.shape[0]:
            raise ValueError("x_t last dimension and timesteps batch dimension do not match the model")
        if self.condition_dim:
            if condition is None or condition.shape[-1] != self.condition_dim:
                raise ValueError(f"condition must have last dimension {self.condition_dim}")
            while condition.ndim < x_t.ndim:
                condition = condition.unsqueeze(-2)
            cond = condition.expand(*x_t.shape[:-1], self.condition_dim)
        else:
            cond = x_t.new_empty(*x_t.shape[:-1], 0)
        time = self.time_projection(self._time_embedding(timesteps, self.time_projection[0].in_features))
        time = time.reshape(x_t.shape[0], *([1] * (x_t.ndim - 2)), -1).expand(*x_t.shape[:-1], -1)
        inputs = torch.cat((x_t, cond, time), dim=-1).reshape(-1, self.input_dim + self.condition_dim + time.shape[-1])
        return self.denoiser(inputs).reshape_as(x_t)

    def training_loss(
        self, x0: torch.Tensor, condition: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        timesteps = torch.randint(self.schedule.num_steps, (x0.shape[0],), device=x0.device)
        noisy, noise = self.schedule.add_noise(x0, timesteps)
        target = noise if self.prediction_type == "epsilon" else x0
        error = (self(noisy, timesteps, condition) - target).pow(2).mean(dim=-1)
        if mask is not None:
            mask = mask.to(dtype=error.dtype)
            error = error * mask
            return error.sum() / mask.sum().clamp_min(1)
        return error.mean()

    @torch.no_grad()
    def sample(
        self, shape: tuple[int, ...], condition: torch.Tensor | None = None, device: torch.device | None = None
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device or next(self.parameters()).device)
        for step in reversed(range(self.schedule.num_steps)):
            t = torch.full((shape[0],), step, device=x.device, dtype=torch.long)
            prediction = self(x, t, condition)
            beta = self.schedule.betas[step]
            x = (x - beta * prediction / self.schedule.sqrt_one_minus_alphas_cumprod[step]) / torch.sqrt(1 - beta)
            if step:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
        return x
