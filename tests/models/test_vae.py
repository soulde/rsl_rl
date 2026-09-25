import torch

from rsl_rl.models import ConditionalVAE


def test_conditional_vae_returns_reconstruction_and_kl_loss() -> None:
    model = ConditionalVAE(input_dim=6, latent_dim=3, condition_dim=2, hidden_dims=(16,))
    x = torch.randn(4, 6)
    condition = torch.randn(4, 2)

    output = model(x, condition)
    losses = model.loss(x, output, reduction="mean")

    assert output.reconstruction.shape == x.shape
    assert output.mu.shape == (4, 3)
    assert output.logvar.shape == (4, 3)
    assert losses["total"].ndim == 0
    assert losses["reconstruction"].ndim == 0
    assert losses["kl"].ndim == 0


def test_conditional_vae_can_decode_deterministic_latent() -> None:
    model = ConditionalVAE(input_dim=5, latent_dim=2, condition_dim=1, hidden_dims=(8,))
    z = torch.zeros(3, 2)
    condition = torch.ones(3, 1)

    decoded = model.decode(z, condition)

    assert decoded.shape == (3, 5)
