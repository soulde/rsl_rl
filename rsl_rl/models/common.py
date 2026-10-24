"""Small reusable helpers shared by neural models."""

import torch


def run_mlp(module, values, mixed_precision=False):
    """Run CUDA MLP matmuls in BF16 while returning FP32 values."""
    if mixed_precision and values.is_cuda:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return module(values).float()
    return module(values)
