# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a CNN policy as an image encoder and a vector policy head."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn

from rsl_rl.models import CNNModel


class _CNNEncoder(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)


class _CNNPolicyHead(nn.Module):
    def __init__(self, model: CNNModel):
        super().__init__()
        self.normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.vector_obs_dim = model.obs_dim
        if model.distribution is None:
            self.deterministic_output = nn.Identity()
        else:
            self.deterministic_output = model.distribution.as_deterministic_output_module()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        vector_obs = observations[:, : self.vector_obs_dim]
        depth_latent = observations[:, self.vector_obs_dim :]
        latent = torch.cat((self.normalizer(vector_obs), depth_latent), dim=-1)
        return self.deterministic_output(self.mlp(latent))


def export_cnn_policy_pair(model: nn.Module, checkpoint_path: str | Path, opset_version: int = 11) -> Path:
    """Export a single-image ``CNNModel`` beside a checkpoint.

    The output directory uses the checkpoint stem and contains
    ``depth_backbone.onnx`` (image to latent), ``student.onnx`` (vector plus
    latent to action mean), and ``io.json`` describing the input ordering.
    """
    if not isinstance(model, CNNModel):
        raise TypeError(f"Expected CNNModel for split export, got {type(model).__name__}")
    if len(model.obs_groups_2d) != 1:
        raise ValueError(f"Split export requires exactly one image group, got {model.obs_groups_2d}")

    image_group = model.obs_groups_2d[0]
    channels = model.obs_channels_2d[0]
    height, width = model.obs_dims_2d[0]
    export_dir = Path(checkpoint_path).with_suffix("")
    export_dir.mkdir(parents=True, exist_ok=True)

    encoder = _CNNEncoder(copy.deepcopy(model.cnns[image_group])).eval().cpu()
    head = _CNNPolicyHead(model).eval().cpu()
    image = torch.zeros(1, channels, height, width)
    vector_dim = model.obs_dim + int(model.cnns[image_group].output_dim)
    vector = torch.zeros(1, vector_dim)
    torch.onnx.export(
        encoder,
        image,
        export_dir / "depth_backbone.onnx",
        input_names=["depth_image"],
        output_names=["depth_latent"],
        opset_version=opset_version,
        dynamo=False,
    )
    torch.onnx.export(
        head,
        vector,
        export_dir / "student.onnx",
        input_names=["obs"],
        output_names=["actions"],
        opset_version=opset_version,
        dynamo=False,
    )
    io_description = {
        "image_group": image_group,
        "image_shape": [channels, height, width],
        "vector_groups": model.obs_groups,
        "vector_dim": model.obs_dim,
        "latent_dim": int(model.cnns[image_group].output_dim),
        "head_input_order": [*model.obs_groups, "depth_latent"],
    }
    (export_dir / "io.json").write_text(json.dumps(io_description, indent=2) + "\n", encoding="utf-8")
    return export_dir
