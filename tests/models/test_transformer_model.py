from __future__ import annotations

import tempfile
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.models import TransformerModel


def _model_inputs() -> tuple[TensorDict, dict[str, list[str]]]:
    obs = TensorDict(
        {
            "state": torch.randn(4, 6),
            "history": torch.randn(4, 5, 3),
            "commands": torch.randn(4, 7, 2),
        },
        batch_size=[4],
    )
    return obs, {"actor": ["state", "history", "commands"], "critic": ["state"]}


def test_transformer_model_supports_mixed_vector_and_sequence_groups() -> None:
    """Mixed input ranks are encoded and fused in configured group order."""
    obs, groups = _model_inputs()
    model = TransformerModel(
        obs,
        groups,
        "actor",
        output_dim=4,
        sequence_groups=["history", "commands"],
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        hidden_dims=[16],
    )

    assert model(obs).shape == (4, 4)
    assert model.get_latent(obs).shape == (4, 22)


def test_transformer_model_supports_sequence_only_with_causal_mask() -> None:
    """Sequence-only models can use causal history attention."""
    obs = TensorDict({"history": torch.randn(3, 6, 5)}, batch_size=[3])
    model = TransformerModel(
        obs,
        {"actor": ["history"]},
        "actor",
        output_dim=2,
        sequence_groups=["history"],
        causal_groups=["history"],
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        hidden_dims=[16],
    )

    assert model(obs).shape == (3, 2)


@pytest.mark.parametrize("pooling", ["mean", "max", "last"])
def test_transformer_model_supports_configured_pooling(pooling: str) -> None:
    """All public pooling modes return a single latent per input sequence."""
    obs = TensorDict({"history": torch.randn(3, 6, 5)}, batch_size=[3])
    model = TransformerModel(
        obs,
        {"actor": ["history"]},
        "actor",
        output_dim=2,
        sequence_groups=["history"],
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        pooling=pooling,
        hidden_dims=[8],
    )

    assert model.get_latent(obs).shape == (3, 8)


def test_transformer_model_validates_sequence_padding_mask_shape() -> None:
    """Per-group padding masks must match that sequence group's batch and time axes."""
    obs, groups = _model_inputs()
    model = TransformerModel(
        obs,
        groups,
        "actor",
        output_dim=1,
        sequence_groups=["history", "commands"],
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        hidden_dims=[8],
    )

    with pytest.raises(ValueError, match=r"history.*padding mask"):
        model(obs, sequence_padding_masks={"history": torch.zeros(4, 4, dtype=torch.bool)})


def test_transformer_model_exposes_gaussian_distribution() -> None:
    """The output distribution follows RSL-RL's Gaussian policy interface."""
    obs, groups = _model_inputs()
    model = TransformerModel(
        obs,
        groups,
        "actor",
        output_dim=3,
        sequence_groups=["history", "commands"],
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        hidden_dims=[16],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
    )

    deterministic = model(obs)
    sampled = model(obs, stochastic_output=True)

    assert deterministic.shape == sampled.shape == (4, 3)
    assert model.output_mean.shape == (4, 3)
    assert model.output_std.shape == (4, 3)
    assert model.output_entropy.shape == (4,)
    assert model.get_output_log_prob(sampled).shape == (4,)


def test_transformer_model_updates_observation_normalization() -> None:
    """Empirical vector normalization updates change the encoded latent."""
    obs = TensorDict({"state": torch.randn(16, 4) + 10.0}, batch_size=[16])
    model = TransformerModel(
        obs,
        {"actor": ["state"]},
        "actor",
        output_dim=1,
        sequence_groups=[],
        obs_normalization=True,
        hidden_dims=[8],
    )
    before = model.get_latent(obs)

    model.update_normalization(obs)

    after = model.get_latent(obs)
    assert not torch.allclose(before, after)


def test_jit_export_matches_eager_output_for_ordered_groups() -> None:
    """The exported policy consumes the configured tensor order and matches eager inference."""
    obs, groups = _model_inputs()
    model = TransformerModel(
        obs,
        groups,
        "actor",
        output_dim=2,
        sequence_groups=["history", "commands"],
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        hidden_dims=[8],
    ).eval()
    inputs = [obs[group] for group in model.obs_groups]

    scripted = torch.jit.script(model.as_jit())

    assert torch.allclose(model(obs), scripted(inputs), atol=1e-5)


def test_onnx_export_preserves_group_inputs() -> None:
    """The ONNX adapter provides named inputs and valid deterministic output."""
    import onnx

    obs, groups = _model_inputs()
    model = TransformerModel(
        obs,
        groups,
        "actor",
        output_dim=2,
        sequence_groups=["history", "commands"],
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        hidden_dims=[8],
    ).eval()
    onnx_model = model.as_onnx()
    onnx_model.eval()

    with tempfile.NamedTemporaryFile(suffix=".onnx") as output_file:
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),
            output_file.name,
            opset_version=18,
            input_names=onnx_model.input_names,
            output_names=onnx_model.output_names,
        )
        graph = onnx.load(output_file.name)
        onnx.checker.check_model(graph)

    assert [value.name for value in graph.graph.input] == model.obs_groups
    assert [value.name for value in graph.graph.output] == ["actions"]
