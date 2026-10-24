"""Differentiable URDF link kinematics in a configured joint/body order."""

from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from .rotations import axis_angle_to_mat


class UrdfKinematics:
    """Compute link-origin transforms and velocities, including fixed links."""

    def __init__(self, urdf_path, joint_names, body_names, device="cpu"):
        tree = ET.parse(Path(urdf_path)).getroot()
        nodes = {node.find("child").attrib["link"]: node for node in tree.findall("joint")}
        roots = [node.attrib["name"] for node in tree.findall("link") if node.attrib["name"] not in nodes]
        if len(roots) != 1:
            raise ValueError("URDF must have exactly one root link")
        self.root_body = roots[0]
        self.joint_names, self.body_names = tuple(joint_names), tuple(body_names)
        self.device = torch.device(device)
        order = [self.root_body]
        self.parent, self.joint_ids, axes, positions, rotations = (
            [-1],
            [-1],
            [[0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0]],
            [torch.eye(3)],
        )
        pending = dict(nodes)
        limits = {}
        while pending:
            progressed = False
            for child, node in list(pending.items()):
                parent = node.find("parent").attrib["link"]
                if parent not in order:
                    continue
                origin, axis = node.find("origin"), node.find("axis")
                xyz = [float(x) for x in (origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
                rpy = [float(x) for x in (origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
                ax = [float(x) for x in (axis.attrib.get("xyz", "1 0 0") if axis is not None else "1 0 0").split()]
                name, kind = node.attrib["name"], node.attrib["type"]
                if kind not in ("fixed", "revolute", "continuous"):
                    raise ValueError(f"Unsupported joint type: {name}: {kind}")
                if kind != "fixed" and name not in self.joint_names:
                    raise ValueError(f"Movable joint absent from configured profile: {name}")
                jid = -1 if kind == "fixed" else self.joint_names.index(name)
                self.parent.append(order.index(parent))
                self.joint_ids.append(jid)
                ax_tensor = torch.tensor(ax)
                axes.append((ax_tensor / ax_tensor.norm().clamp_min(1e-12)).tolist())
                positions.append(xyz)
                r = torch.eye(3)
                for i in (2, 1, 0):
                    r = r @ axis_angle_to_mat(torch.eye(3)[i], torch.tensor(rpy[i]))
                rotations.append(r)
                limit = node.find("limit")
                if jid >= 0:
                    limits[name] = (
                        (-torch.pi, torch.pi)
                        if limit is None
                        else (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
                    )
                order.append(child)
                del pending[child]
                progressed = True
            if not progressed:
                raise ValueError("URDF link graph is disconnected or cyclic")
        if any(name not in order for name in body_names):
            raise ValueError("Configured tracking bodies are missing from URDF")
        self.tracked_ids = [order.index(name) for name in body_names]
        self.axes = torch.tensor(axes, device=device)
        self.offset_p = torch.tensor(positions, device=device)
        self.offset_r = torch.stack(rotations).to(device)
        self.q_lo = torch.tensor([limits[name][0] for name in joint_names], device=device)
        self.q_hi = torch.tensor([limits[name][1] for name in joint_names], device=device)

    def forward(self, p, rotation, velocity, omega, q, qd):
        positions, rotations, velocities, omegas = [p], [rotation], [velocity], [omega]
        for i in range(1, len(self.parent)):
            parent, joint = self.parent[i], self.joint_ids[i]
            rp = rotations[parent]
            offset = (rp @ self.offset_p[i].unsqueeze(-1)).squeeze(-1)
            pos = positions[parent] + offset
            rot = rp @ self.offset_r[i]
            vel = velocities[parent] + torch.cross(omegas[parent], offset, dim=-1)
            om = omegas[parent]
            if joint >= 0:
                om = om + (rot @ self.axes[i].unsqueeze(-1)).squeeze(-1) * qd[..., joint : joint + 1]
                rot = rot @ axis_angle_to_mat(self.axes[i], q[..., joint])
            positions.append(pos)
            rotations.append(rot)
            velocities.append(vel)
            omegas.append(om)
        return (
            torch.stack([positions[i] for i in self.tracked_ids], -2),
            torch.stack([rotations[i] for i in self.tracked_ids], -3),
            torch.stack([velocities[i] for i in self.tracked_ids], -2),
            torch.stack([omegas[i] for i in self.tracked_ids], -2),
        )
