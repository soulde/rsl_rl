from types import SimpleNamespace
import numpy as np
import pytest
import torch
import onnx
from rsl_rl.models.supermimic import SuperMimicActor, SuperMimicWorldModel
from rsl_rl.models.supermimic.tracking_mdp import AnalyticTrackingMDP
from rsl_rl.models.supermimic.rotations import so3_exp
from rsl_rl.models.supermimic.state import flatten_state, unflatten_state
from rsl_rl.runners import OffPolicyRunner
from rsl_rl.algorithms.supermimic import _SuperMimicPolicy
from rsl_rl.storage import SuperMimicSequenceBatch
from tests.supermimic_helpers import make_mdp


def test_goal_uses_future_local_velocity_and_tilt_ignores_yaw(tmp_path):
    mdp = make_mdp(tmp_path)
    motion = mdp.dataset.motions[0]
    motion["body_quat_xyzw"][1:, 0] = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])
    motion["body_lin_vel_w"][1:, 0] = torch.tensor([0.0, 1.0, 0.0])
    mdp = AnalyticTrackingMDP(mdp.dataset, mdp.kin, mdp.default_q, goal_frames=2)
    ids = torch.zeros(1, dtype=torch.long)
    frames = ids.clone()
    goal = mdp.goal(ids, frames)[0].reshape(1, 2, -1)
    torch.testing.assert_close(goal[0, 0, 9:12], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6)
    state = list(mdp.reference(ids, frames))
    state[1] = so3_exp(torch.tensor([[0.0, 0.0, 1.0]]))
    assert not mdp.termination(tuple(state), ids, frames).any()


def test_urdf_order_fk_and_analytic_gradients(tmp_path):
    mdp = make_mdp(tmp_path)
    ids = torch.zeros(2, dtype=torch.long)
    state = list(mdp.reference(ids, ids))
    state[4] = torch.tensor([[0.1, 0.2], [0.2, 0.3]], requires_grad=True)
    bp, *_ = mdp.kin.forward(*state)
    torch.testing.assert_close(
        bp[0, 2, :2], torch.tensor([1 + np.cos(0.2), np.sin(0.2)], dtype=torch.float32), atol=1e-6, rtol=1e-6
    )
    loss = mdp.reward(tuple(state), ids, ids, torch.zeros(2, 2), torch.zeros(2, 2)).sum()
    loss.backward()
    assert torch.isfinite(state[4].grad).all() and state[4].grad.abs().sum() > 0
    torch.testing.assert_close(mdp.soft_hi, torch.tensor([0.9, 1.8]))


def test_world_model_integrates_rotation_and_trains_autoregressively(tmp_path):
    mdp = make_mdp(tmp_path)
    ids = torch.zeros(2, dtype=torch.long)
    state = mdp.reference(ids, ids)
    model = SuperMimicWorldModel(2, 6, hidden_dims=(16,))
    context = torch.zeros(2, 6)
    context[:, 0] = 0.8
    model.update_normalization(state, state, torch.zeros(2, 2), context)
    with torch.no_grad():
        model.net[-1].bias[3] = 0.1
    result = model.step(state, torch.zeros(2, 2), context)
    torch.testing.assert_close(result[1].transpose(-1, -2) @ result[1], torch.eye(3).expand(2, 3, 3))
    batch = SuperMimicSequenceBatch(
        flatten_state(state)[:, None].expand(-1, 4, -1),
        torch.zeros(2, 3, 2),
        context[:, None].expand(-1, 3, -1),
        torch.zeros(2, 3, dtype=torch.bool),
        torch.ones(2, 3),
    )
    losses = model.loss(batch)
    losses["total_loss"].backward()
    assert torch.isfinite(losses["total_loss"])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("obs_dim,action_dim,goal_dim", [(18, 2, 38), (81, 23, 610), (99, 29, 730)])
def test_runner_exports_normalized_unbounded_actor_with_dynamic_batch(tmp_path, obs_dim, action_dim, goal_dim):
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(7)
    actor = SuperMimicActor(obs_dim, action_dim, goal_dim, hidden_dims=(16,), encoder_hidden_dims=(16,)).eval()
    actor.obs_norm.update(torch.randn(12, obs_dim) + 2)
    actor.goal_norm.update(torch.randn(12, goal_dim) - 1)
    with torch.no_grad():
        actor.network[-1].bias.fill_(2.0)
    alg = SimpleNamespace(actor=actor)
    alg.get_policy = lambda: _SuperMimicPolicy(alg)
    runner = object.__new__(OffPolicyRunner)
    runner.alg = alg
    runner.export_policy_to_jit(str(tmp_path))
    runner.export_policy_to_onnx(str(tmp_path))
    jit = torch.jit.load(str(tmp_path / "policy.pt"))
    onnx.checker.check_model(str(tmp_path / "policy.onnx"))
    session = ort.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    for batch in (1, 3):
        h, g = torch.randn(batch, 10, obs_dim), torch.randn(batch, goal_dim)
        with torch.no_grad():
            expected = actor(h, g)
            torch.testing.assert_close(jit(h, g), expected)
        assert expected.abs().max() > 1
        np.testing.assert_allclose(
            session.run(None, {"observation_history": h.numpy(), "goal": g.numpy()})[0],
            expected.numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
