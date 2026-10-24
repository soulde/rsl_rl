from types import SimpleNamespace
import torch
from tensordict import TensorDict
from rsl_rl.models.supermimic.kinematics import UrdfKinematics
from rsl_rl.models.supermimic.tracking_mdp import AnalyticTrackingMDP
from rsl_rl.models.supermimic.state import flatten_state
from rsl_rl.datasets import UniformReferenceStateSampler


def make_mdp(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="test"><link name="root"/><link name="arm"/><link name="tip"/>
    <joint name="hinge" type="revolute"><parent link="root"/><child link="arm"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint>
    <joint name="hinge2" type="revolute"><parent link="arm"/><child link="tip"/><origin xyz="1 0 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1"/></joint></robot>""")
    kin = UrdfKinematics(path, ["hinge2", "hinge"], ["root", "arm", "tip"])
    q = torch.zeros(80, 2)
    quat = torch.zeros(80, 1, 4)
    quat[..., 3] = 1
    motion = {
        "body_pos_w": torch.tensor([0.0, 0.0, 1.0]).expand(80, 1, 3).clone(),
        "body_quat_xyzw": quat,
        "body_lin_vel_w": torch.zeros(80, 1, 3),
        "body_ang_vel_w": torch.zeros(80, 1, 3),
        "joint_pos": q,
        "joint_vel": torch.zeros_like(q),
    }

    class Dataset:
        device = "cpu"
        joint_names = ("hinge2", "hinge")
        motions = [motion]
        reference_bins = [[(0, i) for i in range(40)], [(0, i) for i in range(40, 79)]]
        reference_state_sampler = UniformReferenceStateSampler()

        def set_reference_state_sampler(self, sampler):
            self.reference_state_sampler = sampler

        def sample_reference_state_batch(self, batch_size):
            return self.reference_state_sampler.sample(self, batch_size)

        def update_reference_state_sampler(self, bin_ids, returns):
            self.reference_state_sampler.update_returns(bin_ids, returns)

        def reference_state_sampler_state_dict(self):
            return self.reference_state_sampler.state_dict()

        def load_reference_state_sampler_state_dict(self, state):
            self.reference_state_sampler.load_state_dict(state)

        def _decode_reference_states(self, frames):
            ids = torch.tensor([f for _, f in frames])
            return {
                "root_pos": motion["body_pos_w"][ids, 0],
                "root_quat_xyzw": motion["body_quat_xyzw"][ids, 0],
                "root_lin_vel": motion["body_lin_vel_w"][ids, 0],
                "root_ang_vel": motion["body_ang_vel_w"][ids, 0],
                "joint_pos": q[ids],
                "joint_vel": motion["joint_vel"][ids],
            }

    return AnalyticTrackingMDP(Dataset(), kin, torch.zeros(2), goal_frames=2)


def make_env(mdp, count=2):
    return SimpleNamespace(unwrapped=SimpleNamespace(_supermimic_mdp=mdp), num_actions=2, num_envs=count)


def observation(mdp, frame=0, count=2):
    ids = torch.zeros(count, dtype=torch.long)
    frames = torch.full((count,), frame, dtype=torch.long)
    state = mdp.reference(ids, frames)
    context = torch.zeros(count, 6)
    context[:, 0] = 0.8
    clean = mdp.observation(state, torch.zeros(count, 2), context, ids, frames)
    return TensorDict(
        {
            "policy": clean,
            "state": flatten_state(state),
            "context": context,
            "goal": mdp.goal(ids, frames)[0],
            "reference_motion_id": ids[:, None],
            "reference_frame_id": frames[:, None],
            "reference_bin_id": torch.zeros(count, 1, dtype=torch.long),
        },
        batch_size=[count],
    )


def config():
    return {
        "algorithm": {
            "class_name": "rsl_rl.algorithms:SuperMimic",
            "horizon": 3,
            "wm_only_iters": 0,
            "wm_batches": 1,
            "policy_batch_size": 4,
            "critic_epochs": 1,
            "critic_minibatches": 2,
            "warmup_steps": 4,
        },
        "actor": {"hidden_dims": [16], "encoder_hidden_dims": [16]},
        "critic": {"hidden_dims": [16]},
        "model": {"hidden_dims": [16]},
        "memory": {"capacity": 32, "sequence_length": 3, "batch_size": 4},
        "obs_groups": {"actor": ["policy"], "state": ["state"], "context": ["context"], "goal": ["goal"]},
    }
