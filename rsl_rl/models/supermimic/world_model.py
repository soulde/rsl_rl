"""Normalized frame-local residual dynamics with SO(3) integration."""

import torch
from torch import nn
from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import RunningNorm
from ..common import run_mlp
from .state import invariant_features, unflatten_state, normalize_context
from .rotations import yaw_mat, so3_exp, so3_log


class SuperMimicWorldModel(nn.Module):
    def __init__(self, num_joints, context_dim, hidden_dims=(2048, 2048, 2048), mixed_precision=True):
        super().__init__()
        self.num_joints = num_joints
        self.mixed_precision = mixed_precision
        self.state_dim = 13 + 2 * num_joints
        self.in_dim = 13 + 2 * num_joints + num_joints + context_dim
        self.out_dim = 12 + 2 * num_joints
        self.net = MLP(self.in_dim, self.out_dim, hidden_dims, activation="silu", layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(self.out_dim, eps=1e-4)
        self.register_buffer(
            "error_floor",
            torch.tensor(
                [0.003] * 3 + [0.005] * 3 + [0.02] * 3 + [0.05] * 3 + [0.003] * num_joints + [0.05] * num_joints
            ),
        )

    def inputs(self, state, action, normalized_context):
        return torch.cat((invariant_features(state), action, normalized_context), -1)

    def targets(self, state, next_state):
        p, r, v, w, q, qd = state
        p1, r1, v1, w1, q1, qd1 = next_state
        rt = r.transpose(-1, -2)
        return torch.cat(
            (
                (yaw_mat(r).transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1),
                so3_log(rt @ r1),
                (rt @ (v1 - v).unsqueeze(-1)).squeeze(-1),
                (rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    @torch.no_grad()
    def update_normalization(self, state, next_state, action, context):
        self.in_norm.update(self.inputs(state, action, normalize_context(context)))
        self.out_norm.update(self.targets(state, next_state))

    def step(self, state, action, normalized_context):
        p, r, v, w, q, qd = state
        delta = run_mlp(
            self.net,
            self.in_norm(self.inputs(state, action, normalized_context)),
            self.mixed_precision and self.training,
        )
        delta = delta * self.out_norm.std() + self.out_norm.mean
        dp, dr, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        return (
            p + (yaw_mat(r) @ dp.unsqueeze(-1)).squeeze(-1),
            r @ so3_exp(dr),
            v + (r @ dv.unsqueeze(-1)).squeeze(-1),
            w + (r @ dw.unsqueeze(-1)).squeeze(-1),
            q + dq,
            qd + dqd,
        )

    def loss(self, batch):
        state = unflatten_state(batch.states[:, 0])
        std = torch.maximum(self.out_norm.std(), self.error_floor)
        errors = []
        for step in range(batch.actions.shape[1]):
            state = self.step(state, batch.actions[:, step], normalize_context(batch.contexts[:, step]))
            target = unflatten_state(batch.states[:, step + 1])
            errors.append((self.targets(target, state) / std).square().mean(-1))
        error = torch.stack(errors, 1)
        total = (error * batch.mask).sum() / batch.actions.shape[0]
        last_mask = batch.mask[:, -1]
        return {
            "total_loss": total,
            "state_loss": total / batch.actions.shape[1],
            "last_step_error": (error[:, -1] * last_mask).sum() / last_mask.sum().clamp_min(1.0),
        }
