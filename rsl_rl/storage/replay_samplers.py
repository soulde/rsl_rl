"""Reusable sampling strategies for replay storage backends."""

from __future__ import annotations

import torch

from .replay import ReplayBatch, ReplaySampler


class UniformTransitionSampler(ReplaySampler):
    """Sample individual tensor-ring records uniformly with replacement."""

    def sample(self, buffer, batch_size: int, **kwargs) -> ReplayBatch:
        if buffer.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        if batch_size <= 0:
            raise ValueError("Sample batch size must be positive")
        ages = torch.randint(buffer.size, (batch_size,), device=buffer.device)
        return buffer.gather(ages)


class EpisodeSequenceSampler(ReplaySampler):
    """Sample fixed-length sequences without crossing episode boundaries."""

    def sample(self, buffer, batch_size: int | None = None, **kwargs) -> ReplayBatch:
        batch_size = buffer.batch_size if batch_size is None else batch_size
        eligible = [episode for episode in buffer._episodes if len(episode) >= buffer.sequence_length]
        if not eligible:
            raise RuntimeError(f"No completed episode is long enough for sequence length {buffer.sequence_length}.")
        observations, actions, rewards, terminated = [], [], [], []
        for _ in range(batch_size):
            episode = eligible[int(torch.randint(len(eligible), ()).item())]
            max_start = len(episode) - buffer.sequence_length
            start = int(torch.randint(max_start + 1, ()).item())
            sequence = episode[start : start + buffer.sequence_length]
            observations.append(
                torch.stack([sequence[0]["observation"]] + [row["next_observation"] for row in sequence])
            )
            actions.append(torch.stack([row["action"] for row in sequence]))
            rewards.append(torch.stack([row["reward"] for row in sequence]))
            terminated.append(torch.stack([row["terminated"] for row in sequence]))
        return ReplayBatch(
            observations=torch.stack(observations),
            actions=torch.stack(actions),
            rewards=torch.stack(rewards),
            terminated=torch.stack(terminated),
        )


class SuperMimicReplaySampler(ReplaySampler):
    """Sample time-major windows and policy starts from SuperMimic replay."""

    def sample(
        self,
        buffer,
        batch_size: int | None = None,
        *,
        kind: str = "windows",
        history_length: int | None = None,
        **kwargs,
    ):
        if kind == "windows":
            return self._sample_windows(buffer, batch_size)
        if kind == "starts":
            return self._sample_starts(buffer, batch_size, history_length)
        raise ValueError(f"Unknown SuperMimic replay sample kind: {kind!r}")

    @staticmethod
    def _sample_windows(buffer, batch_size):
        batch_size = buffer.batch_size if batch_size is None else batch_size
        if buffer.size < buffer.sequence_length:
            raise RuntimeError(f"Replay needs {buffer.sequence_length} vector steps")
        start = torch.randint(buffer.size - buffer.sequence_length + 1, (batch_size,), device=buffer.device)
        ages = start[:, None] + torch.arange(buffer.sequence_length, device=buffer.device)
        env = torch.randint(buffer.num_envs, (batch_size, 1), device=buffer.device)
        done = buffer.storage.get("done", ages, env)
        mask = torch.cumprod(
            torch.cat((torch.ones(batch_size, 1, device=buffer.device), 1 - done[:, :-1].float()), dim=1), dim=1
        )
        states = torch.cat(
            (
                buffer.storage.get("state", ages[:, :1], env),
                buffer.storage.get("next_state", ages, env),
            ),
            dim=1,
        )
        return ReplayBatch(
            states=states,
            actions=buffer.storage.get("action", ages, env),
            contexts=buffer.storage.get("context", ages, env),
            done=done,
            mask=mask,
        )

    @staticmethod
    def _sample_starts(buffer, batch_size, history_length):
        batch_size = buffer.batch_size if batch_size is None else batch_size
        history_length = buffer.history_length if history_length is None else history_length
        if not buffer.size:
            raise RuntimeError("Replay needs samples")
        ages = torch.randint(buffer.size, (batch_size,), device=buffer.device)
        env = torch.randint(buffer.num_envs, (batch_size,), device=buffer.device)
        keys = (
            "state",
            "a_prev",
            "context",
            "reference_bin_id",
            "reference_motion_id",
            "reference_frame_id",
            "ep_step",
        )
        start = {key: buffer.storage.get(key, ages, env) for key in keys}
        back = torch.arange(history_length - 1, -1, -1, device=buffer.device)[None]
        back = torch.minimum(back, start["ep_step"].long()[:, None])
        back = torch.minimum(back, ages[:, None])
        history_ages = ages[:, None] - back
        history_env = env[:, None].expand_as(history_ages)
        history = {key: buffer.storage.get(key, history_ages, history_env) for key in keys}
        return ReplayBatch(start=ReplayBatch(**start), history=ReplayBatch(**history))
