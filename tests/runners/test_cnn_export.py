from __future__ import annotations

import torch
from tensordict import TensorDict

import onnx

from rsl_rl.models import CNNModel
from rsl_rl.runners.cnn_export import export_cnn_policy_pair


def test_split_cnn_export_honors_requested_opset(tmp_path) -> None:
    obs = TensorDict(
        {
            "policy": torch.randn(2, 6),
            "depth": torch.randn(2, 1, 16, 16),
        },
        batch_size=[2],
    )
    model = CNNModel(
        obs,
        {"actor": ["policy", "depth"]},
        "actor",
        4,
        hidden_dims=[16],
        cnn_cfg={
            "depth": {
                "output_channels": [8],
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "activation": "elu",
                "flatten": True,
                "output_dim": 8,
                "output_activation": "elu",
            }
        },
    )

    export_dir = export_cnn_policy_pair(model, tmp_path / "model_1.pt", opset_version=13)

    for filename in ("depth_backbone.onnx", "student.onnx"):
        exported = onnx.load(export_dir / filename)
        onnx.checker.check_model(exported)
        assert [(entry.domain, entry.version) for entry in exported.opset_import] == [("", 13)]
