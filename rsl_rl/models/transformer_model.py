# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transformer-based models for vector and sequence observation groups."""

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState, SinusoidalPositionalEncoding
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class TransformerModel(nn.Module):
    """Encode rank-3 observation sequences and fuse them with rank-2 vector observations."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        sequence_groups: tuple[str, ...] | list[str],
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.0,
        transformer_activation: str = "gelu",
        pooling: str = "mean",
        causal_groups: tuple[str, ...] | list[str] = (),
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize encoders, fusion head, and optional output distribution.

        Args:
            obs: Example TensorDict used to infer observation shapes.
            obs_groups: Mapping from observation sets to ordered observation group names.
            obs_set: Observation set consumed by this model.
            output_dim: Output width before applying an optional distribution.
            sequence_groups: Selected groups expected to have shape ``[batch, time, features]``.
            hidden_dims: Hidden layer widths of the output MLP.
            activation: MLP activation name.
            embedding_dim: Width used by each sequence Transformer.
            num_layers: Number of Transformer encoder layers per sequence group.
            num_heads: Attention head count per encoder layer.
            feedforward_dim: Transformer feedforward width.
            dropout: Transformer dropout probability.
            transformer_activation: PyTorch Transformer activation (``relu`` or ``gelu``).
            pooling: Sequence pooling mode: ``mean``, ``max``, or ``last``.
            causal_groups: Sequence groups that may attend only to current and earlier tokens.
            obs_normalization: Whether to empirically normalize vector and sequence features.
            distribution_cfg: Optional output distribution configuration.
        """
        super().__init__()
        if obs_set not in obs_groups:
            raise KeyError(f"observation set '{obs_set}' is not defined")
        self._example_obs_shapes = {group: tuple(obs[group].shape) for group in obs_groups[obs_set]}
        self.obs_groups = list(obs_groups[obs_set])
        self.sequence_groups = list(sequence_groups)
        self.vector_groups = [group for group in self.obs_groups if group not in self.sequence_groups]
        if len(set(self.sequence_groups)) != len(self.sequence_groups):
            raise ValueError("sequence_groups must not contain duplicates")
        if not set(self.sequence_groups).issubset(self.obs_groups):
            raise ValueError("all sequence_groups must be selected by obs_groups[obs_set]")
        if not set(causal_groups).issubset(self.sequence_groups):
            raise ValueError("causal_groups must be a subset of sequence_groups")
        if pooling not in {"mean", "max", "last"}:
            raise ValueError(f"unsupported pooling '{pooling}'; expected 'mean', 'max', or 'last'")
        if embedding_dim <= 0 or num_layers <= 0 or num_heads <= 0 or feedforward_dim <= 0:
            raise ValueError("embedding_dim, num_layers, num_heads, and feedforward_dim must be positive")
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        if transformer_activation not in {"relu", "gelu"}:
            raise ValueError("transformer_activation must be 'relu' or 'gelu'")

        self.sequence_group_set = set(self.sequence_groups)
        self.causal_groups = set(causal_groups)
        self.pooling = pooling
        self.embedding_dim = embedding_dim
        self.sequence_input_dims: list[int] = []
        self.vector_input_dims: list[int] = []
        batch_size = None
        for group in self.obs_groups:
            if group not in obs:
                raise KeyError(f"observation group '{group}' is missing")
            tensor = obs[group]
            expected_rank = 3 if group in self.sequence_group_set else 2
            if tensor.ndim != expected_rank:
                raise ValueError(
                    f"observation group '{group}' must have rank {expected_rank}, got shape {tuple(tensor.shape)}"
                )
            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise ValueError("all observation groups must have the same batch dimension")
            if group in self.sequence_group_set:
                self.sequence_input_dims.append(tensor.shape[-1])
            else:
                self.vector_input_dims.append(tensor.shape[-1])

        self.obs_dim = sum(self.vector_input_dims)
        self.obs_normalization = obs_normalization
        self.obs_normalizer = (
            EmpiricalNormalization(self.obs_dim) if obs_normalization and self.obs_dim else nn.Identity()
        )
        self.sequence_normalizers = nn.ModuleList(
            [EmpiricalNormalization(dim) if obs_normalization else nn.Identity() for dim in self.sequence_input_dims]
        )
        self.sequence_projections = nn.ModuleList([nn.Linear(dim, embedding_dim) for dim in self.sequence_input_dims])
        self.position_encodings = nn.ModuleList(
            [SinusoidalPositionalEncoding(embedding_dim) for _ in self.sequence_input_dims]
        )
        self.sequence_encoders = nn.ModuleList()
        for _ in self.sequence_input_dims:
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
                activation=transformer_activation,
                norm_first=True,
            )
            self.sequence_encoders.append(
                nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
            )

        self.latent_dim = self.obs_dim + len(self.sequence_groups) * embedding_dim
        if distribution_cfg is not None:
            distribution_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim
        self.mlp = MLP(self.latent_dim, mlp_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        sequence_padding_masks: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute deterministic or stochastic model output."""
        if masks is not None and not self.is_recurrent:
            obs = unpad_trajectories(obs, masks)
        latent = self.get_latent(obs, sequence_padding_masks=sequence_padding_masks)
        output = self.mlp(latent)
        if self.distribution is None:
            return output
        if stochastic_output:
            self.distribution.update(output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(output)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        sequence_padding_masks: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return concatenated normalized vector features and pooled sequence features."""
        del masks, hidden_state
        sequence_padding_masks = sequence_padding_masks or {}
        unknown_masks = set(sequence_padding_masks) - self.sequence_group_set
        if unknown_masks:
            raise ValueError(f"padding masks supplied for non-sequence groups: {sorted(unknown_masks)}")

        vector_values = [obs[group] for group in self.vector_groups]
        if vector_values:
            vectors = torch.cat(vector_values, dim=-1)
            vectors = self.obs_normalizer(vectors)
        else:
            vectors = None

        pooled_sequences: list[torch.Tensor] = []
        for index, group in enumerate(self.sequence_groups):
            sequence = obs[group]
            if sequence.ndim != 3 or sequence.shape[-1] != self.sequence_input_dims[index]:
                raise ValueError(f"observation group '{group}' has incompatible sequence shape {tuple(sequence.shape)}")
            padding_mask = sequence_padding_masks.get(group)
            if padding_mask is not None:
                expected = sequence.shape[:2]
                if padding_mask.shape != expected or padding_mask.dtype != torch.bool:
                    raise ValueError(f"{group} padding mask must be bool with shape {tuple(expected)}")
                if torch.any(padding_mask.all(dim=1)):
                    raise ValueError(f"{group} padding mask cannot mask every token in a sequence")
            sequence = self.sequence_normalizers[index](sequence)
            sequence = self.sequence_projections[index](sequence)
            sequence = self.position_encodings[index](sequence)
            causal_mask = None
            if group in self.causal_groups:
                causal_mask = torch.triu(
                    torch.ones(sequence.shape[1], sequence.shape[1], dtype=torch.bool, device=sequence.device),
                    diagonal=1,
                )
            encoded = self.sequence_encoders[index](sequence, mask=causal_mask, src_key_padding_mask=padding_mask)
            pooled_sequences.append(self._pool(encoded, padding_mask))

        features = ([] if vectors is None else [vectors]) + pooled_sequences
        return torch.cat(features, dim=-1)

    def _pool(self, sequence: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if self.pooling == "mean":
            if padding_mask is None:
                return sequence.mean(dim=1)
            keep = (~padding_mask).unsqueeze(-1).to(sequence.dtype)
            return (sequence * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        if self.pooling == "max":
            if padding_mask is None:
                return sequence.max(dim=1).values
            return sequence.masked_fill(padding_mask.unsqueeze(-1), torch.finfo(sequence.dtype).min).max(dim=1).values
        if padding_mask is None:
            return sequence[:, -1]
        positions = torch.arange(sequence.shape[1], device=sequence.device).expand_as(padding_mask)
        last = positions.masked_fill(padding_mask, -1).max(dim=1).values
        return sequence[torch.arange(sequence.shape[0], device=sequence.device), last]

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset internal recurrent state (no-op for this feed-forward model)."""
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        """Return ``None`` because the model carries no state between transitions."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach recurrent state (no-op for this feed-forward model)."""
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        """Mean of the current output distribution."""
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        """Standard deviation of the current output distribution."""
        return self.distribution.std  # type: ignore[union-attr]

    @property
    def output_entropy(self) -> torch.Tensor:
        """Entropy of the current output distribution."""
        return self.distribution.entropy  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Raw parameters of the current output distribution."""
        return self.distribution.params  # type: ignore[union-attr]

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute output log probabilities."""
        return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two distribution parameterizations."""
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore[union-attr]

    def update_normalization(self, obs: TensorDict) -> None:
        """Update vector and per-sequence empirical normalization statistics."""
        if not self.obs_normalization:
            return
        vector_values = [obs[group] for group in self.vector_groups]
        if vector_values and isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.update(torch.cat(vector_values, dim=-1))
        for index, group in enumerate(self.sequence_groups):
            normalizer = self.sequence_normalizers[index]
            if isinstance(normalizer, EmpiricalNormalization):
                normalizer.update(obs[group].reshape(-1, obs[group].shape[-1]))

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible deterministic model accepting an ordered tensor list."""
        return _TorchTransformerModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-compatible deterministic model accepting an ordered tensor list."""
        return _OnnxTransformerModel(self, verbose)


class _TorchTransformerModel(nn.Module):
    """Export wrapper taking selected observation tensors as an ordered list."""

    sequence_group_indices: list[int]
    vector_group_indices: list[int]
    causal_group_indices: list[int]

    def __init__(self, model: TransformerModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.sequence_normalizers = copy.deepcopy(model.sequence_normalizers)
        self.sequence_projections = copy.deepcopy(model.sequence_projections)
        self.position_encodings = copy.deepcopy(model.position_encodings)
        self.sequence_encoders = copy.deepcopy(model.sequence_encoders)
        self.mlp = copy.deepcopy(model.mlp)
        self.input_shapes = [tuple(model._example_obs_shapes[group]) for group in model.obs_groups]
        self.sequence_group_indices = [model.obs_groups.index(group) for group in model.sequence_groups]
        self.vector_group_indices = [model.obs_groups.index(group) for group in model.vector_groups]
        self.causal_group_indices = [model.obs_groups.index(group) for group in model.causal_groups]
        self.pooling = model.pooling
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from tensors ordered like the configured observation groups."""
        vectors = [inputs[index] for index in self.vector_group_indices]
        if vectors:
            vector_features = self.obs_normalizer(torch.cat(vectors, dim=-1))
        else:
            vector_features = torch.zeros(inputs[0].shape[0], 0, device=inputs[0].device, dtype=inputs[0].dtype)

        sequences: list[torch.Tensor] = []
        for encoder_index, modules in enumerate(
            zip(self.sequence_normalizers, self.sequence_projections, self.position_encodings, self.sequence_encoders)
        ):
            normalizer, projection, position_encoding, encoder = modules
            group_index = self.sequence_group_indices[encoder_index]
            x = normalizer(inputs[group_index])
            x = projection(x)
            x = position_encoding(x)
            if group_index in self.causal_group_indices:
                mask = torch.triu(
                    torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1
                )
            else:
                mask = torch.zeros(0, 0, dtype=torch.bool, device=x.device)
            x = encoder(x) if mask.numel() == 0 else encoder(x, mask=mask)
            if self.pooling == "max":
                sequences.append(x.max(dim=1).values)
            elif self.pooling == "last":
                sequences.append(x[:, -1])
            else:
                sequences.append(x.mean(dim=1))

        features: list[torch.Tensor] = [vector_features]
        features.extend(sequences)
        features = torch.cat(features, dim=-1)
        return self.deterministic_output(self.mlp(features))

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op)."""
        pass


class _OnnxTransformerModel(nn.Module):
    """ONNX wrapper exposing one input tensor per selected observation group."""

    def __init__(self, model: TransformerModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.policy = _TorchTransformerModel(model)
        self.group_names = model.obs_groups
        self.input_shapes = [tuple(model._example_obs_shapes[group]) for group in model.obs_groups]

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference from positional observation tensors."""
        return self.policy(list(inputs))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return one dummy tensor per configured observation group."""
        return tuple(torch.zeros((1, *shape[1:])) for shape in self.input_shapes)

    @property
    def input_names(self) -> list[str]:
        """Stable ONNX input names matching the observation groups."""
        return self.group_names

    @property
    def output_names(self) -> list[str]:
        """The ONNX output name."""
        return ["actions"]
