"""Reference SuperMimic: learned dynamics with an analytic tracking MDP."""

import math
from collections.abc import Mapping
import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.datasets import ReturnAwareReferenceSampler
from rsl_rl.models.supermimic import SuperMimicActor, SuperMimicCritic, SuperMimicWorldModel
from rsl_rl.models.supermimic.state import unflatten_state, normalize_context
from rsl_rl.storage import SuperMimicReplayBuffer


class SuperMimic:
    checkpoint_version = 2

    def __init__(
        self,
        model,
        actor,
        critic,
        replay_buffer,
        mdp,
        obs_groups,
        device="cpu",
        horizon=32,
        gamma=0.99,
        gae_lambda=0.95,
        lr=2e-4,
        action_noise=0.1,
        warmup_steps=64,
        wm_only_iters=500,
        utd=0.4,
        wm_batches=10,
        policy_batch_size=4096,
        critic_epochs=5,
        critic_minibatches=4,
        lr_decay_iters=250000,
        grad_clip_norm=1.0,
        save_replay=False,
    ):
        if min(horizon, utd, wm_batches, policy_batch_size, critic_epochs, critic_minibatches, lr_decay_iters) <= 0:
            raise ValueError("SuperMimic schedule values must be positive")
        self.model, self.actor, self.critic = model.to(device), actor.to(device), critic.to(device)
        self.replay_buffer, self.mdp = replay_buffer, mdp
        self.dataset = mdp.dataset
        self.obs_groups, self.device = obs_groups, torch.device(device)
        self.horizon, self.gamma, self.gae_lambda = horizon, gamma, gae_lambda
        self.initial_lr, self.learning_rate = lr, lr
        self.action_noise, self.warmup_steps, self.wm_only_iters = action_noise, warmup_steps, wm_only_iters
        self.utd, self.wm_batches, self.policy_batch_size = utd, wm_batches, policy_batch_size
        self.critic_epochs, self.critic_minibatches = critic_epochs, critic_minibatches
        self.lr_decay_iters, self.grad_clip_norm, self.save_replay = lr_decay_iters, grad_clip_norm, save_replay
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr, betas=(0.9, 0.95))
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
        self.num_steps = 0
        self.num_updates = 0
        self.training = True
        self._transition = None
        self._obs_history = None
        self._previous_actions = None
        self._episode_returns = None
        self._undiscounted_returns = None
        self._episode_steps = None

    @property
    def warmup_steps_collected(self):
        """Return replayed vector steps available toward the configured warmup."""
        return min(self.replay_buffer.size, self.warmup_steps)

    def _extract(self, obs, name):
        return torch.cat([obs[key].to(self.device, dtype=torch.float32) for key in self.obs_groups[name]], -1)

    def act(self, obs):
        actor_obs = self._extract(obs, "actor")
        goal = self._extract(obs, "goal")
        if "reference_bin_id" not in obs:
            raise ValueError("Missing reference_bin_id")
        bins = obs["reference_bin_id"].to(self.device).long().reshape(-1)
        if (bins < 0).any() or (bins >= len(self.dataset.reference_bins)).any():
            raise ValueError("reference_bin_id out of range")
        if self._obs_history is None:
            self._obs_history = actor_obs[:, None].expand(-1, self.actor.history_length, -1).clone()
            self._previous_actions = torch.zeros(len(actor_obs), self.actor.action_dim, device=self.device)
            self._episode_returns = torch.zeros(len(actor_obs), device=self.device)
            self._undiscounted_returns = torch.zeros_like(self._episode_returns)
            self._episode_steps = torch.zeros(len(actor_obs), device=self.device, dtype=torch.long)
        else:
            self._obs_history = torch.cat((self._obs_history[:, 1:], actor_obs[:, None]), 1)
        with torch.no_grad():
            if self.training:
                self.actor.obs_norm.update(actor_obs)
                self.actor.goal_norm.update(goal)
            actions = self.actor(self._obs_history, goal)
            if self.training:
                actions = (actions + self.action_noise * torch.randn_like(actions)).clamp(-20, 20)
        self._transition = {
            "state": self._extract(obs, "state").detach(),
            "context": self._extract(obs, "context").detach(),
            "reference_bin_id": bins,
            "reference_motion_id": obs["reference_motion_id"].to(self.device).long().reshape(-1),
            "reference_frame_id": obs["reference_frame_id"].to(self.device).long().reshape(-1),
            "action": actions.detach(),
            "a_prev": self._previous_actions.clone(),
        }
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        if self._transition is None:
            raise RuntimeError("Call act before process_env_step")
        dones = dones.to(self.device).bool().reshape(-1)
        next_obs = obs
        if dones.any():
            terminal = extras.get("final_observation", extras.get("final_obs", extras.get("terminal_observation")))
            if terminal is None:
                raise RuntimeError("SuperMimic requires pre-reset final observations on done steps")
            if isinstance(terminal, Mapping):
                terminal = TensorDict(terminal, batch_size=obs.batch_size)
            next_obs = obs.clone()
            next_obs[dones] = terminal.to(self.device)[dones]
        timeouts = extras.get("time_outs", torch.zeros_like(dones)).to(self.device).bool().reshape(-1)
        terminated = extras.get("terminated", dones & ~timeouts).to(self.device).bool().reshape(-1)
        rewards = rewards.to(self.device).reshape(-1)
        next_state = self._extract(next_obs, "state")
        transition = self._transition
        self.replay_buffer.add({
            **transition,
            "next_state": next_state,
            "reward": rewards,
            "terminated": terminated,
            "done": dones,
        })
        self.model.update_normalization(
            unflatten_state(transition["state"]),
            unflatten_state(next_state),
            transition["action"],
            transition["context"],
        )
        self._episode_returns += self.gamma ** self._episode_steps.float() * rewards
        self._undiscounted_returns += rewards
        self._episode_steps += 1
        if dones.any():
            length = self._episode_steps[dones].float()
            bootstrap = torch.where(
                terminated[dones],
                0.0,
                self.gamma**length * (self._undiscounted_returns[dones] / length) / (1 - self.gamma),
            )
            self.dataset.update_reference_state_sampler(
                transition["reference_bin_id"][dones], self._episode_returns[dones] + bootstrap
            )
            self._episode_returns[dones] = 0
            self._undiscounted_returns[dones] = 0
            self._episode_steps[dones] = 0
            reset_obs = self._extract(obs, "actor")
            self._obs_history[dones] = reset_obs[dones, None].expand(-1, self.actor.history_length, -1)
        self._previous_actions = transition["action"].clone()
        self._previous_actions[dones] = 0
        self.num_steps += len(dones)
        self._transition = None

    def _sample_valid_starts(self):
        starts, histories = [], []
        count = 0
        for _ in range(128):
            replay_batch = self.replay_buffer.sample_starts(2 * self.policy_batch_size)
            start, history = replay_batch.start, replay_batch.history
            valid = self.mdp.valid_starts(start["reference_motion_id"], start["reference_frame_id"], self.horizon)
            ids = valid.nonzero(as_tuple=True)[0][: self.policy_batch_size - count]
            if len(ids):
                starts.append({k: v[ids] for k, v in start.items()})
                histories.append({k: v[ids] for k, v in history.items()})
                count += len(ids)
            if count == self.policy_batch_size:
                return (
                    {k: torch.cat([x[k] for x in starts]) for k in starts[0]},
                    {k: torch.cat([x[k] for x in histories]) for k in histories[0]},
                )
        raise RuntimeError("Replay needs valid reference starts with a full imagination horizon")

    def update(self, iteration=None):
        iteration = self.num_updates if iteration is None else iteration
        fraction = min(iteration / self.lr_decay_iters, 1.0)
        self.learning_rate = self.initial_lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * fraction)))
        for optimizer in (self.model_optimizer, self.actor_optimizer, self.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = self.learning_rate
        if self.replay_buffer.size < self.replay_buffer.sequence_length:
            return {}
        for _ in range(self.wm_batches):
            loss = self.model.loss(self.replay_buffer.sample_windows())
            self.model_optimizer.zero_grad(set_to_none=True)
            loss["total_loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.model_optimizer.step()
        metrics = {
            "model_loss": float(loss["state_loss"].detach()),
            "last_step_error": float(loss["last_step_error"].detach()),
        }
        if iteration >= self.wm_only_iters:
            try:
                metrics.update(self._update_imagined_policy())
            except RuntimeError as error:
                if "needs valid reference" not in str(error):
                    raise
        self.num_updates += 1
        return metrics

    def _update_imagined_policy(self):
        start, history = self._sample_valid_starts()
        state = unflatten_state(start["state"])
        initial_state = state
        ids, frame = start["reference_motion_id"], start["reference_frame_id"]
        context = start["context"]
        normalized = normalize_context(context)
        with torch.no_grad():
            obs_history = self.mdp.observation(
                unflatten_state(history["state"]),
                history["a_prev"],
                history["context"],
                history["reference_motion_id"],
                history["reference_frame_id"],
                noisy=True,
            )
        previous = start["a_prev"]
        goal = self.mdp.goal(ids, frame)[0]
        initial_frame, initial_goal = frame, goal
        model_flags = [p.requires_grad for p in self.model.parameters()]
        critic_flags = [p.requires_grad for p in self.critic.parameters()]
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.critic.parameters():
            p.requires_grad_(False)
        states, frames, goals, rewards, terminals = [], [], [], [], []
        survival = torch.ones(len(frame), device=self.device)
        objective = torch.zeros_like(survival)
        try:
            for step in range(self.horizon):
                action = self.actor(obs_history, goal)
                state = self.model.step(state, action, normalized)
                frame = frame + 1
                reward = self.mdp.reward(state, ids, frame, action, previous)
                terminal = self.mdp.termination(state, ids, frame, soft=True)
                objective = objective + self.gamma**step * survival * reward
                survival = survival * (1 - terminal)
                observation = self.mdp.observation(state, action, context, ids, frame, noisy=True)
                obs_history = torch.cat((obs_history[:, 1:], observation[:, None]), 1)
                goal = self.mdp.goal(ids, frame)[0]
                previous = action
                states.append(state)
                frames.append(frame)
                goals.append(goal)
                rewards.append(reward)
                terminals.append(terminal)
            objective = objective + self.gamma**self.horizon * survival * self.critic(
                self.mdp.critic_features(state, ids, frame, normalized, goal)
            )
            actor_loss = -objective.mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
            if torch.isfinite(norm):
                self.actor_optimizer.step()
        finally:
            for p, flag in zip(self.model.parameters(), model_flags, strict=True):
                p.requires_grad_(flag)
            for p, flag in zip(self.critic.parameters(), critic_flags, strict=True):
                p.requires_grad_(flag)
        with torch.no_grad():
            all_states = tuple(torch.stack([initial_state[i]] + [s[i].detach() for s in states]) for i in range(6))
            all_frames = torch.stack([initial_frame] + frames)
            all_goals = torch.stack([initial_goal] + goals)
            all_ids = ids[None].expand_as(all_frames)
            features = self.mdp.critic_features(
                all_states, all_ids, all_frames, normalized[None].expand(self.horizon + 1, -1, -1), all_goals
            )
            self.critic.norm.update(features)
            values = self.critic(features)
            _, returns = self.compute_gae(
                torch.stack(rewards).detach(),
                values[:-1],
                values[-1],
                torch.stack(terminals).detach(),
                self.gamma,
                self.gae_lambda,
            )
            x = features[:-1].flatten(0, 1)
            y = returns.flatten()
        for _ in range(self.critic_epochs):
            for idx in torch.tensor_split(torch.randperm(len(x), device=self.device), self.critic_minibatches):
                if not len(idx):
                    continue
                critic_loss = (self.critic(x[idx]) - y[idx]).square().mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
                self.critic_optimizer.step()
        return {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "imagined_survival": float(survival.detach().mean()),
        }

    @staticmethod
    def compute_gae(rewards, values, next_value, terminated, gamma, gae_lambda):
        returns = []
        target = next_value
        for step in range(len(rewards) - 1, -1, -1):
            next_v = next_value if step == len(rewards) - 1 else values[step + 1]
            target = rewards[step] + gamma * (1 - terminated[step].float()) * (
                (1 - gae_lambda) * next_v + gae_lambda * target
            )
            returns.append(target)
        returns = torch.stack(returns[::-1])
        return returns - values, returns

    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        task = getattr(env, "unwrapped", env)
        mdp = getattr(task, "_supermimic_mdp", None)
        if mdp is None:
            raise ValueError("SuperMimic task must expose _supermimic_mdp (analytic tracking heads)")
        algorithm = dict(cfg["algorithm"])
        algorithm.pop("class_name", None)
        algorithm.pop("rnd_cfg", None)
        memory = cfg.get("memory", {})
        n = env.num_actions
        model = SuperMimicWorldModel(
            n, mdp.context_dim, tuple(cfg.get("model", {}).get("hidden_dims", (2048, 2048, 2048)))
        )
        actor_cfg = dict(cfg.get("actor", {}))
        critic_cfg = dict(cfg.get("critic", {}))
        actor = SuperMimicActor(mdp.observation_dim, n, mdp.goal_dim, memory.get("history_length", 10), **actor_cfg)
        critic = SuperMimicCritic(mdp.critic_dim, **critic_cfg)
        dataset = mdp.dataset
        sampler = ReturnAwareReferenceSampler(
            len(dataset.reference_bins), max_return=mdp.max_step_reward / (1 - algorithm.get("gamma", 0.99)),
            device=device,
        )
        dataset.set_reference_state_sampler(sampler)
        replay = SuperMimicReplayBuffer(
            memory.get("capacity", 1000),
            env.num_envs,
            memory.get("sequence_length", 8),
            memory.get("batch_size", 2048),
            memory.get("history_length", 10),
            device,
        )
        if mdp.clip_lengths.max() <= algorithm.get("horizon", 32):
            raise ValueError("Reference clips must exceed imagination horizon")
        return cls(model, actor, critic, replay, mdp, cfg["obs_groups"], device, **algorithm)

    def sample_rsi(self, env_ids):
        sampled = self.dataset.sample_reference_state_batch(len(env_ids))
        if sampled.bin_ids is None:
            raise RuntimeError("SuperMimic reference-state sampler must return reference bin IDs")
        return sampled.states, sampled.bin_ids

    def initialize_env(self, env):
        """Bind this algorithm's reference sampler through the task's public RSI API."""
        task = getattr(env, "unwrapped", None)
        reset = getattr(env, "reset", None)
        set_sampler = getattr(task, "set_reference_state_sampler", None)
        if not callable(set_sampler) or not callable(reset):
            raise RuntimeError("SuperMimic requires an environment with set_reference_state_sampler() and reset()")
        set_sampler(self.sample_rsi, self.get_rsi_joint_layout())
        reset()

    def get_rsi_joint_layout(self):
        from rsl_rl.datasets.rsi_contract import reference_joint_layout

        return reference_joint_layout(self.mdp.dataset)

    def train_mode(self):
        self.training = True
        self.model.train()
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        self.training = False
        self.model.eval()
        self.actor.eval()
        self.critic.eval()

    def get_policy(self):
        return _SuperMimicPolicy(self)

    def save(self):
        state = {
            "supermimic_version": self.checkpoint_version,
            "num_steps": self.num_steps,
            "num_updates": self.num_updates,
            "sampler_state_dict": self.dataset.reference_state_sampler_state_dict(),
        }
        for name in ("model", "actor", "critic"):
            state[name + "_state_dict"] = getattr(self, name).state_dict()
            state[name + "_optimizer_state_dict"] = getattr(self, name + "_optimizer").state_dict()
        if self.save_replay:
            state["replay_buffer_state_dict"] = self.replay_buffer.state_dict()
        return state

    def load(self, state, load_cfg=None, strict=True):
        if state.get("supermimic_version") != self.checkpoint_version:
            raise ValueError(
                "Incompatible pre-reference-parity SuperMimic checkpoint; retain it and start a new experiment"
            )
        load_cfg = load_cfg or {"model": True, "optimizer": True, "iteration": True}
        for name in ("model", "actor", "critic"):
            if load_cfg.get("model", True):
                getattr(self, name).load_state_dict(state[name + "_state_dict"], strict=strict)
            if load_cfg.get("optimizer", True):
                getattr(self, name + "_optimizer").load_state_dict(state[name + "_optimizer_state_dict"])
        if load_cfg.get("replay_buffer", True) and "replay_buffer_state_dict" in state:
            self.replay_buffer.load_state_dict(state["replay_buffer_state_dict"])
            self.replay_buffer.episode_steps.zero_()
            if self.replay_buffer.size:
                # The simulator starts new episodes; close the last saved row
                # so replay windows cannot span the checkpoint boundary.
                self.replay_buffer.data["done"][(self.replay_buffer.ptr - 1) % self.replay_buffer.capacity] = True
        self.dataset.load_reference_state_sampler_state_dict(state["sampler_state_dict"])
        self.num_updates = state["num_updates"]
        self.num_steps = state["num_steps"]
        self._obs_history = None
        self._transition = None
        return load_cfg.get("iteration", True)


class _SuperMimicPolicy(nn.Module):
    def __init__(self, algorithm):
        super().__init__()
        self.algorithm = algorithm
        self.actor = algorithm.actor
        self._history = None
        self._pending_reset = None

    def forward(self, obs):
        device = next(self.actor.parameters()).device
        current = torch.cat([obs[name].to(device) for name in self.algorithm.obs_groups["actor"]], -1)
        goal = torch.cat([obs[name].to(device) for name in self.algorithm.obs_groups["goal"]], -1)
        if self._history is None or self._history.device != device or self._history.shape[0] != len(current):
            self._history = current[:, None].expand(-1, self.algorithm.actor.history_length, -1).clone()
            self._pending_reset = None
        else:
            self._history = torch.cat((self._history[:, 1:], current[:, None]), 1)
        if self._pending_reset is not None:
            mask = self._pending_reset.to(device)
            self._history[mask] = current[mask, None].expand(-1, self._history.shape[1], -1)
            self._pending_reset = None
        return self.actor(self._history, goal)

    def reset(self, dones=None):
        if dones is None:
            self._history = None
            self._pending_reset = None
        elif self._history is not None:
            mask = dones.to(self._history.device).bool().reshape(-1)
            self._pending_reset = mask.clone() if self._pending_reset is None else (self._pending_reset | mask)

    def as_jit(self):
        return self.actor.as_jit()

    def as_onnx(self, verbose=False):
        return self.actor.as_onnx(verbose)
