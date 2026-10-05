"""Tests for TD-MPC2 state-observation world-model components."""

import torch

from rsl_rl.models.tdmpc2 import TDMPC2WorldModel
from rsl_rl.models.tdmpc2.math import two_hot, two_hot_inv


def _cfg(episodic: bool = True) -> dict:
    return {
        "latent_dim": 16,
        "enc_dim": 16,
        "mlp_dim": 32,
        "num_enc_layers": 2,
        "num_q": 3,
        "num_bins": 11,
        "vmin": -5.0,
        "vmax": 5.0,
        "simnorm_dim": 4,
        "dropout": 0.0,
        "log_std_min": -5.0,
        "log_std_max": 1.0,
        "tau": 0.1,
        "episodic": episodic,
    }


def test_world_model_predicts_latent_reward_termination_policy_and_q_shapes() -> None:
    model = TDMPC2WorldModel(obs_dim=7, action_dim=2, cfg=_cfg())
    observations = torch.randn(5, 7)
    actions = torch.rand(5, 2) * 2 - 1
    latent = model.encode(observations)
    next_latent = model.next(latent, actions)
    sampled_actions, info = model.pi(latent)

    assert latent.shape == (5, 16)
    assert next_latent.shape == (5, 16)
    assert model.reward(latent, actions).shape == (5, 11)
    assert model.termination(latent).shape == (5, 1)
    assert sampled_actions.shape == (5, 2)
    assert torch.all(sampled_actions.abs() <= 1.0)
    assert info["log_prob"].shape == (5, 1)
    assert model.Q(latent, actions, return_type="all").shape == (3, 5, 11)
    assert model.Q(latent, actions).shape == (5, 1)


def test_non_episodic_model_omits_termination_head() -> None:
    model = TDMPC2WorldModel(obs_dim=7, action_dim=2, cfg=_cfg(episodic=False))

    assert model.termination_head is None
    assert torch.equal(model.termination(torch.randn(3, 16)), torch.ones(3, 1))


def test_target_q_soft_update_moves_toward_online_parameters() -> None:
    model = TDMPC2WorldModel(obs_dim=7, action_dim=2, cfg=_cfg())
    before = next(model.target_q.parameters()).detach().clone()
    with torch.no_grad():
        next(model.q_networks.parameters()).add_(1.0)

    model.soft_update_target_q()

    after = next(model.target_q.parameters()).detach()
    assert torch.allclose(after, before + 0.1, atol=1e-6)


def test_two_hot_round_trip_is_finite() -> None:
    values = torch.tensor([[-2.0], [0.0], [3.0]])

    encoded = two_hot(values, _cfg())
    decoded = two_hot_inv(torch.log(encoded + 1e-8), _cfg())

    assert encoded.shape == (3, 11)
    assert torch.allclose(encoded.sum(dim=-1), torch.ones(3))
    assert torch.isfinite(decoded).all()
