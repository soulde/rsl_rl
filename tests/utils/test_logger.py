import torch

from rsl_rl.utils.logger import Logger


class _Writer:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, key, value, step):
        self.scalars[key] = (float(value), step)


def test_logger_records_mean_step_reward():
    cfg = {"algorithm": {"rnd_cfg": None}, "num_steps_per_env": 2, "logger": "tensorboard"}
    logger = Logger(
        log_dir=None,
        cfg=cfg,
        env_cfg={},
        num_envs=2,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device="cpu",
    )
    writer = _Writer()
    logger.writer = writer
    logger.logger_type = "tensorboard"

    logger.process_env_step(torch.tensor([1.0, 3.0]), torch.zeros(2), {})
    logger.log(
        it=0,
        start_it=0,
        total_it=1,
        collect_time=1.0,
        learn_time=1.0,
        loss_dict={},
        learning_rate=1.0e-3,
        action_std=torch.ones(1),
        rnd_weight=None,
        print_minimal=True,
    )

    assert writer.scalars["Train/mean_step_reward"] == (2.0, 0)
    assert logger.step_reward_count == 0
