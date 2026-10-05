import numpy as np
import pytest
import torch

from rsl_rl.datasets import MotionDataset, SomaMotionDataset


@pytest.mark.parametrize("source_order", ["xyzw", "wxyz"])
def test_world_features_preserve_order_and_ignore_root_height_rotation(tmp_path, source_order):
    root = np.array([[10, 20, 30], [110, 220, 330]], dtype=np.float32)
    positions = np.stack([root, root + [14, 15, 16], root + [11, 12, 13]], axis=1)
    # Change root orientation between frames; world-axis offsets must stay intact.
    quat = np.array([[0, 0, np.sqrt(0.5), np.sqrt(0.5)], [0, 0, 0, 1]], dtype=np.float32)
    source_quat = quat if source_order == "xyzw" else quat[:, [3, 0, 1, 2]]
    np.savez(
        tmp_path / "motion.npz",
        fps=np.float32(50),
        joint_pos=np.tile([7, 8], (2, 1)),
        joint_vel=np.tile([9, 10], (2, 1)),
        body_names=np.array(["root", "hand", "foot"]),
        body_pos_w=positions,
        body_quat_w=np.repeat(source_quat[:, None], 3, axis=1),
        body_lin_vel_w=np.tile([1, 2, 3], (2, 3, 1)),
        body_ang_vel_w=np.tile([4, 5, 6], (2, 3, 1)),
    )
    if source_order == "xyzw":
        payload = {}
        with np.load(tmp_path / "motion.npz") as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["joint_names"] = np.array(["j0", "j1"])
        payload["body_names"] = np.array(["root", "hand", "foot"])
        np.savez(tmp_path / "motion.npz", **payload)
        dataset_type = SomaMotionDataset
    else:
        dataset_type = MotionDataset
    dataset = dataset_type(str(tmp_path), key_body_names=["foot", "hand"])
    expected = torch.arange(1, 17, dtype=torch.float32)
    assert dataset.amp_observation_dim == 16
    torch.testing.assert_close(dataset.transitions[0], torch.cat([expected, expected]))
    torch.testing.assert_close(dataset.motions[0]["body_quat_w"][:, 0], torch.tensor(quat))
    torch.testing.assert_close(dataset.motions[0]["body_quat_xyzw"][:, 0], torch.tensor(quat))
    with pytest.raises(ValueError, match="dimension"):
        dataset_type(str(tmp_path), key_body_names=["foot", "hand"], amp_observation_dim=23)
