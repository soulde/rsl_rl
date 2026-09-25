import torch

from rsl_rl.models import ConditionalDiffusionModel, DiffusionSchedule


def test_diffusion_schedule_add_noise_preserves_shape() -> None:
    schedule = DiffusionSchedule(num_steps=8)
    x0 = torch.randn(4, 5, 3)
    timesteps = torch.tensor([0, 1, 4, 7])

    noisy, noise = schedule.add_noise(x0, timesteps)

    assert noisy.shape == x0.shape
    assert noise.shape == x0.shape
    assert torch.all(schedule.alphas_cumprod[1:] < schedule.alphas_cumprod[:-1])


def test_conditional_diffusion_loss_accepts_token_mask() -> None:
    model = ConditionalDiffusionModel(input_dim=3, condition_dim=2, hidden_dims=(16,), num_steps=8)
    x0 = torch.randn(2, 4, 3)
    condition = torch.randn(2, 2)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)

    loss = model.training_loss(x0, condition=condition, mask=mask)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_conditional_diffusion_sample_has_requested_shape() -> None:
    model = ConditionalDiffusionModel(input_dim=3, condition_dim=2, hidden_dims=(16,), num_steps=4)

    sample = model.sample((2, 5, 3), condition=torch.randn(2, 2))

    assert sample.shape == (2, 5, 3)
