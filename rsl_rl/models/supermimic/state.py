"""Robot-independent SuperMimic state and context conventions."""

import torch
from .rotations import mat_to_6d, mat_to_quat, quat_to_mat, yaw_mat


def flatten_state(state):
    p, rotation, velocity, omega, q, qd = state
    return torch.cat((p, mat_to_quat(rotation), velocity, omega, q, qd), -1)


def unflatten_state(state):
    n = (state.shape[-1] - 13) // 2
    return (
        state[..., :3],
        quat_to_mat(state[..., 3:7]),
        state[..., 7:10],
        state[..., 10:13],
        state[..., 13 : 13 + n],
        state[..., 13 + n :],
    )


def invariant_features(state):
    p, rotation, velocity, omega, q, qd = state
    rt = rotation.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(yaw_mat(rotation).transpose(-1, -2) @ rotation),
            (rt @ velocity.unsqueeze(-1)).squeeze(-1),
            (rt @ omega.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


def normalize_context(context):
    return torch.cat(((context[..., :1] - 0.75) / 0.45, context[..., 1:-3] / 0.01, context[..., -3:] / 0.05), -1)
