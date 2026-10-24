"""Time-major SuperMimic replay on shared tensor-ring storage."""

import torch

from .replay import ReplayBatch, ReplaySampler, TensorRingStorage
from .replay_samplers import SuperMimicReplaySampler


class SuperMimicSequenceBatch(ReplayBatch):
    """Sequence batch consumed by the SuperMimic world model."""

    def __init__(self, states, actions, contexts, done, mask):
        super().__init__(states=states, actions=actions, contexts=contexts, done=done, mask=mask)


class SuperMimicReplayBuffer:
    _FIELDS = (
        "state",
        "next_state",
        "action",
        "a_prev",
        "reward",
        "terminated",
        "done",
        "context",
        "reference_bin_id",
        "reference_motion_id",
        "reference_frame_id",
        "ep_step",
    )

    def __init__(
        self,
        capacity,
        num_envs,
        sequence_length,
        batch_size,
        history_length=10,
        device="cpu",
        sampler: ReplaySampler | None = None,
    ):
        if min(capacity, num_envs, sequence_length, batch_size, history_length) <= 0 or sequence_length > capacity:
            raise ValueError("Invalid SuperMimic replay dimensions")
        self.capacity, self.num_envs, self.sequence_length = capacity, num_envs, sequence_length
        self.batch_size, self.history_length, self.device = batch_size, history_length, torch.device(device)
        self.storage = TensorRingStorage(self.capacity, device=self.device)
        self.sampler = sampler or SuperMimicReplaySampler()
        self.num_episodes = 0
        self.episode_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)

    @property
    def data(self):
        return self.storage.data

    @property
    def ptr(self):
        return self.storage.position

    @property
    def size(self):
        return self.storage.size

    def add(self, transition):
        missing = set(self._FIELDS) - {"ep_step"} - transition.keys()
        if missing:
            raise ValueError(f"Missing replay fields: {sorted(missing)}")
        values = {
            key: torch.as_tensor(transition[key], device=self.device).detach()
            for key in self._FIELDS
            if key != "ep_step"
        }
        values["ep_step"] = self.episode_steps.clone()
        for key, value in values.items():
            if value.shape[0] != self.num_envs:
                raise ValueError(f"Invalid replay batch for {key}")
            if key in ("done", "terminated"):
                values[key] = value.bool().reshape(self.num_envs)
            elif key.startswith("reference_"):
                values[key] = value.long().reshape(self.num_envs)
            elif key != "ep_step":
                values[key] = value.float()
        self.storage.add({key: value.unsqueeze(0) for key, value in values.items()})
        dones = values["done"].bool().reshape(-1)
        self.episode_steps += 1
        self.episode_steps[dones] = 0
        self.num_episodes += int(dones.sum())

    def _row(self, age):
        return self.storage.row_indices(age)

    def sample_windows(self, batch_size=None):
        batch = self.sampler.sample(self, batch_size, kind="windows")
        return SuperMimicSequenceBatch(**batch)

    def sample_starts(self, batch_size=None, history_length=None):
        return self.sampler.sample(self, batch_size, kind="starts", history_length=history_length)

    def state_dict(self):
        return {
            k: getattr(self, k)
            for k in ("capacity", "num_envs", "sequence_length", "history_length", "ptr", "size", "num_episodes")
        } | {"episode_steps": self.episode_steps.clone(), "data": {k: v.clone() for k, v in self.data.items()}}

    def load_state_dict(self, state):
        for key in ("capacity", "num_envs", "sequence_length", "history_length"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Replay checkpoint {key} mismatch")
        self.storage.load_state_dict(
            {"capacity": self.capacity, "size": state["size"], "position": state["ptr"], "data": state["data"]}
        )
        self.num_episodes = state["num_episodes"]
        self.episode_steps = state["episode_steps"].to(self.device)
