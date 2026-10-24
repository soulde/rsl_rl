"""Analytic tracking heads used identically by simulation and imagined rollouts.

Adapted from the successful SuperMimic reference; robot topology and orders are
constructor inputs, never imports from a training project.
"""

import torch
from .rotations import mat_to_6d, quat_to_mat, rot_angle_sq, yaw_mat
from .state import invariant_features


class AnalyticTrackingMDP:
    def __init__(self, dataset, kinematics, default_q, control_dt=0.02, goal_frames=10):
        self.dataset, self.kin = dataset, kinematics
        self.device = dataset.device
        self.default_q = torch.as_tensor(default_q, device=self.device)
        self.num_joints = len(kinematics.joint_names)
        self.control_dt, self.goal_frames = control_dt, goal_frames
        self.goal_dim = goal_frames * (15 + 2 * self.num_joints)
        self.observation_dim = 3 * self.num_joints + 12
        self.context_dim = self.num_joints + 4
        self.critic_dim = (
            (13 + 2 * self.num_joints)
            + self.num_joints
            + 3 * len(kinematics.body_names)
            + 9
            + self.context_dim
            + self.goal_dim
        )
        self.max_step_reward = 5 * control_dt
        n = self.num_joints
        self.obs_noise = torch.tensor(
            [0.01] * n + [0.5] * n + [0.2] * 3 + [0.05] * 3 + [0.0] * n + [0.05] * 6, device=self.device
        )
        mid, half = (kinematics.q_lo + kinematics.q_hi) / 2, (kinematics.q_hi - kinematics.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - 0.9 * half, mid + 0.9 * half
        self.clip_lengths = torch.tensor([len(m["joint_pos"]) for m in dataset.motions], device=self.device)
        self.clip_offsets = torch.cat((
            torch.zeros(1, device=self.device, dtype=torch.long),
            self.clip_lengths.cumsum(0)[:-1],
        ))
        positions = torch.cat([m["body_pos_w"][:, 0] for m in dataset.motions])
        quaternions = torch.cat([m["body_quat_xyzw"][:, 0] for m in dataset.motions])
        self.reference_state = (
            positions,
            quat_to_mat(quaternions[:, [3, 0, 1, 2]]),
            torch.cat([m["body_lin_vel_w"][:, 0] for m in dataset.motions]),
            torch.cat([m["body_ang_vel_w"][:, 0] for m in dataset.motions]),
            torch.cat([m["joint_pos"] for m in dataset.motions]),
            torch.cat([m["joint_vel"] for m in dataset.motions]),
        )

    def reference(self, motion_ids, frame_ids):
        frame_ids = frame_ids.clamp_min(0).minimum(self.clip_lengths[motion_ids] - 1)
        index = self.clip_offsets[motion_ids] + frame_ids
        return tuple(value[index] for value in self.reference_state)

    def goal(self, motion_ids, frame_ids, num_steps=1):
        result = []
        for step in range(num_steps):
            current = (frame_ids + step).minimum(self.clip_lengths[motion_ids] - 1)
            p, r, *_ = self.reference(motion_ids, current)
            future = current[:, None] + torch.arange(1, self.goal_frames + 1, device=self.device)
            ids = motion_ids[:, None].expand_as(future)
            fp, fr, fv, fw, fq, fqd = self.reference(ids, future)
            ht = yaw_mat(r).transpose(-1, -2)[:, None]
            rt = fr.transpose(-1, -2)
            values = torch.cat(
                (
                    (ht @ (fp - p[:, None]).unsqueeze(-1)).squeeze(-1),
                    mat_to_6d(ht @ fr),
                    (rt @ fv.unsqueeze(-1)).squeeze(-1),
                    (rt @ fw.unsqueeze(-1)).squeeze(-1),
                    fq,
                    0.1 * fqd,
                ),
                -1,
            )
            result.append(values.flatten(-2))
        return torch.stack(result)

    def observation(self, state, previous_action, context, motion_ids, frame_ids, noisy=False):
        _, r, _, w, q, qd = state
        ref_r = self.reference(motion_ids, frame_ids)[1]
        rt = r.transpose(-1, -2)
        obs = torch.cat(
            (
                q - self.default_q - context[..., 1:-3],
                qd,
                (rt @ w.unsqueeze(-1)).squeeze(-1),
                -r[..., 2, :],
                previous_action,
                mat_to_6d(rt @ ref_r),
            ),
            -1,
        )
        return self.add_noise(obs) if noisy else obs

    def add_noise(self, observation):
        return observation + (2 * torch.rand_like(observation) - 1) * self.obs_noise

    def _tracking(self, state, motion_ids, frame_ids):
        p, r, *_ = state
        ref = self.reference(motion_ids, frame_ids)
        rp, rr, *_ = ref
        bp, br, bv, bw = self.kin.forward(*state)
        ref_bp, ref_br, ref_bv, ref_bw = self.kin.forward(*ref)
        dr = yaw_mat(r @ rr.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        aligned_p = anchor.unsqueeze(-2) + (dr.unsqueeze(-3) @ (ref_bp - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
        aligned_r = dr.unsqueeze(-3) @ ref_br
        return ref, aligned_p - bp, aligned_r, br, ref_bv - bv, ref_bw - bw

    def reward(self, state, motion_ids, frame_ids, action, previous_action):
        ref, ep, rr, br, ev, ew = self._tracking(state, motion_ids, frame_ids)
        p, r, _, _, q, _ = state
        track = (
            torch.exp(-ep.square().sum(-1).mean(-1) / 0.3**2)
            + torch.exp(-rot_angle_sq(rr, br).mean(-1) / 0.4**2)
            + torch.exp(-ev.square().sum(-1).mean(-1) / 1.0**2)
            + torch.exp(-ew.square().sum(-1).mean(-1) / 3.14**2)
            + 0.5 * torch.exp(-(ref[0] - p).square().sum(-1) / 0.3**2)
            + 0.5 * torch.exp(-rot_angle_sq(ref[1], r) / 0.4**2)
        )
        rate = (action - previous_action).square().sum(-1)
        limit = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        return (track - 0.1 * rate - 2 * limit) * self.control_dt

    def termination(self, state, motion_ids, frame_ids, soft=False):
        ref = self.reference(motion_ids, frame_ids)
        ez = (ref[0][..., 2] - state[0][..., 2]).abs()
        et = (ref[1][..., 2, 2] - state[1][..., 2, 2]).abs()
        if soft:
            return 1 - (1 - torch.sigmoid((ez - 0.25) / 0.02)) * (1 - torch.sigmoid((et - 0.8) / 0.04))
        return (ez > 0.25) | (et > 0.8)

    def critic_features(self, state, motion_ids, frame_ids, normalized_context, goal):
        ref, ep, *_ = self._tracking(state, motion_ids, frame_ids)
        p, r, _, _, q, _ = state
        ht = yaw_mat(r).transpose(-1, -2)
        return torch.cat(
            (
                invariant_features(state),
                ref[4] - q,
                (ht.unsqueeze(-3) @ ep.unsqueeze(-1)).squeeze(-1).flatten(-2),
                (ht @ (ref[0] - p).unsqueeze(-1)).squeeze(-1),
                mat_to_6d(r.transpose(-1, -2) @ ref[1]),
                normalized_context,
                goal,
            ),
            -1,
        )

    def valid_starts(self, motion_ids, frame_ids, horizon):
        return frame_ids + horizon <= self.clip_lengths[motion_ids] - 1
