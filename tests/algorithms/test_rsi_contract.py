from types import SimpleNamespace

import torch

from rsl_rl.algorithms.amp import AMP


def test_sample_rsi_has_only_fixed_state_fields_and_wxyz_quaternions():
    raw = dict(root_pos=torch.zeros(2, 3), root_quat_xyzw=torch.tensor([[0.6, 0., 0., 0.8]]).repeat(2, 1),
               root_lin_vel=torch.ones(2, 3), root_ang_vel=torch.zeros(2, 3),
               joint_pos=torch.tensor([[0.8, 0.6, 0., 0., 0.2]]).repeat(2, 1), joint_vel=torch.ones(2, 4),
               amp_obs=torch.zeros(2, 4), joint_types=["ball", "hinge"])
    agent = AMP.__new__(AMP)
    agent.reference_motion_dataset = SimpleNamespace(reference_joint_names=["rod", "hip"],
        reference_joint_types=["ball", "hinge"], reference_pos_offsets=[0, 4, 5], reference_vel_offsets=[0, 3, 4])
    agent._reference_frame_sampler = lambda n: raw
    state = agent.sample_rsi(torch.tensor([3, 7]))
    assert set(state) == {"root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel"}
    torch.testing.assert_close(state["root_quat"], torch.tensor([[0.8, 0.6, 0., 0.]]).repeat(2, 1))
    torch.testing.assert_close(state["joint_pos"], raw["joint_pos"])
    assert agent.get_rsi_joint_layout()["joint_pos_offsets"] == (0, 4, 5)


def test_legacy_scalar_loader_uses_the_same_fixed_dictionary():
    agent = AMP.__new__(AMP)
    agent.reference_motion_dataset = SimpleNamespace(joint_names=["hip"])
    agent._reference_frame_sampler = lambda n: dict(root_pos=torch.zeros(n, 3),
        root_quat_xyzw=torch.tensor([[0., 0., 0., 1.]]).repeat(n, 1), root_lin_vel=torch.zeros(n, 3),
        root_ang_vel=torch.zeros(n, 3), joint_pos=torch.full((n, 1), 0.2), joint_vel=torch.full((n, 1), 0.3))
    state = agent.sample_rsi(torch.tensor([0]))
    torch.testing.assert_close(state["joint_pos"], torch.tensor([[0.2]]))
    torch.testing.assert_close(state["joint_vel"], torch.tensor([[0.3]]))
    assert set(state) == {"root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel"}
    assert agent.get_rsi_joint_layout()["joint_names"] == ("hip",)
