"""Coordinate transforms for the shared AMP observation contract."""

import torch


def world_to_heading_frame(vectors: torch.Tensor, root_quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Express world vectors in root-heading axes, preserving the world vertical.

    ``vectors`` has shape ``(..., 3)`` or ``(..., points, 3)`` and root quaternions
    have shape ``(..., 4)``. Quaternions are unit XYZW rotations from body to world.
    Only inverse yaw is applied; roll and pitch are not removed. This expresses
    absolute physical velocities in new axes, not derivatives of moving axes.
    """
    x, y, z, w = root_quat_xyzw.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = yaw.cos(), yaw.sin()
    while c.ndim < vectors.ndim - 1:
        c, s = c.unsqueeze(-1), s.unsqueeze(-1)
    vx, vy, vz = vectors.unbind(-1)
    return torch.stack((c * vx + s * vy, -s * vx + c * vy, vz), dim=-1)


def amp_features_to_heading_frame(
    features: torch.Tensor, root_quat_xyzw: torch.Tensor, joint_dim: int
) -> torch.Tensor:
    """Convert AMP velocities/keypoints while retaining all joint feature coordinates.

    Input order is root linear velocity (3), angular velocity (3), joint position
    and velocity (``joint_dim`` total), then root-relative keypoints (3 per point).
    Positions must already have the root position subtracted. Each transition
    endpoint must be converted using its own root orientation.
    """
    keypoint_start = 6 + joint_dim
    keypoint_width = features.shape[-1] - keypoint_start
    if joint_dim < 0 or keypoint_width < 0 or keypoint_width % 3:
        raise ValueError("AMP feature width must be 6 + joint_dim + 3 * keypoint_count")
    shape = features.shape[:-1]
    return torch.cat(
        (
            world_to_heading_frame(features[..., :6].reshape(*shape, 2, 3), root_quat_xyzw).flatten(-2),
            features[..., 6:keypoint_start],
            world_to_heading_frame(
                features[..., keypoint_start:].reshape(*shape, keypoint_width // 3, 3), root_quat_xyzw
            ).flatten(-2),
        ),
        dim=-1,
    )
