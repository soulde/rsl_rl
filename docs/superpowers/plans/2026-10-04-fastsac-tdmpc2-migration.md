# FastSAC and TD-MPC2 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FastSAC and TD-MPC2 as native `rsl_rl` algorithms with compatible configuration, runner, inference, and checkpoint behavior.

**Architecture:** Keep `construct_algorithm(obs, env, cfg, device)` and algorithm lifecycle methods consistent with PPO/AMP. Add focused replay storage and model components for each algorithm, plus a runner for off-policy collection and update scheduling that follows existing `rsl_rl` logging and checkpoint conventions.

**Tech Stack:** Python 3.9+, PyTorch, TensorDict, existing `VecEnv`, model, runner, and logging APIs.

**Spec:** `docs/superpowers/specs/2026-10-04-fastsac-tdmpc2-migration-design.md`

## Global Constraints

- Preserve the `rsl_rl` config dictionary and callable resolution model.
- Preserve `TensorDict` observations and the `VecEnv` action/step contract.
- Keep algorithm-specific replay memory in dedicated storage modules.
- Do not change PPO, AMP, or Distillation behavior.
- Do not launch GPU training or download data/checkpoints for validation.
- Compare FastSAC with Holosoma and TD-MPC2 with the official implementation and ManiSkill3 baseline; transfer algorithm behavior, not their package APIs.

## Review Focus

- Replay buffers wrap across capacity: test oldest/newest indexing and restored cursor position in Task 1 and Task 5.
- Timeout and true termination differ for bootstrapping: test FastSAC masks in Task 2 and TD-MPC2 episode segmentation in Task 5.
- TensorDict observations may contain multiple groups: test configured flattening and invalid shapes in Task 2 and Task 6.
- TD-MPC2 sequences near episode boundaries may be too short: test sampling excludes incomplete sequences in Task 5.
- Checkpoint restore must recover targets, optimizers, entropy state, replay cursor, and runner step: test algorithm round trips in Task 2/6 and runner resume in Task 7.

---

### Task 1: Flat Off-Policy Replay Buffer

**Files:**
- Create: `rsl_rl/storage/off_policy_replay_buffer.py`
- Modify: `rsl_rl/storage/__init__.py`
- Test: `tests/storage/test_off_policy_replay_buffer.py`

**Interfaces:**
- Produces `OffPolicyBatch` with `observations`, `actions`, `rewards`, `next_observations`, `dones` tensors and `OffPolicyReplayBuffer(capacity, obs_dim, action_dim, device)` with `add(...)`, `sample(batch_size)`, `state_dict()`, and `load_state_dict(state_dict)`.

- [ ] **Step 1: Write tests** for add/sample shapes, ring wraparound, rejection of sampling an empty buffer, and state round trip preserving samples/cursor.
- [ ] **Step 2: Run** `pytest tests/storage/test_off_policy_replay_buffer.py -q`; confirm the new API is missing.
- [ ] **Step 3: Implement** the typed batch and fixed-capacity tensor ring buffer; keep data on the configured device and validate positive capacity and consistent leading dimensions.
- [ ] **Step 4: Run** `pytest tests/storage/test_off_policy_replay_buffer.py -q`; all cases pass.

### Task 2: FastSAC Models and Algorithm

**Files:**
- Create: `rsl_rl/models/fast_sac.py`
- Modify: `rsl_rl/models/__init__.py`
- Create: `rsl_rl/algorithms/fast_sac.py`
- Modify: `rsl_rl/algorithms/__init__.py`
- Test: `tests/algorithms/test_fast_sac.py`

**Interfaces:**
- Produces `FastSACActor`, `DistributionalCritic`, and `FastSAC` with `construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str)`, `act(obs)`, `process_env_step(obs, rewards, dones, extras)`, `update() -> dict[str, float]`, `train_mode()`, `eval_mode()`, `save()`, `load(loaded_dict, load_cfg, strict)`, and `get_policy()`.
- Config keys live under existing `algorithm`, `actor`, `critic`, `memory`, and runner keys; `class_name` is resolved by `resolve_callable` as in PPO.

- [ ] **Step 1: Write tests** for configured TensorDict observation selection, bounded stochastic/deterministic action shapes, distributional critic shape and C51 projection mass, one CPU update from replay, timeout-aware bootstrap masks, and checkpoint round trip.
- [ ] **Step 2: Run** `pytest tests/algorithms/test_fast_sac.py -q`; confirm missing imports/API fail.
- [ ] **Step 3: Implement** tanh-squashed Gaussian actor log-probability correction, categorical distributional Q critics with configurable `num_atoms`, `v_min`, `v_max`, and `num_q_networks`, target distribution projection, target critics, and automatic entropy tuning with target entropy ratio based on source and Holosoma references.
- [ ] **Step 4: Implement** the `rsl_rl` construction and lifecycle methods; flatten only configured 1D observation groups and validate continuous action dimensions.
- [ ] **Step 5: Run** `pytest tests/algorithms/test_fast_sac.py -q`; all cases pass.

### Task 3: Off-Policy Runner

**Files:**
- Create: `rsl_rl/runners/off_policy_runner.py`
- Modify: `rsl_rl/runners/__init__.py`
- Test: `tests/runners/test_off_policy_runner.py`

**Interfaces:**
- Produces `OffPolicyRunner(env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu")` with `learn(num_learning_iterations)`, `save(path, infos=None)`, `load(path, load_cfg=None, strict=True, map_location=None)`, and `get_inference_policy(device=None)`.
- Runner config uses `num_steps_per_env`, `save_interval`, `algorithm.learning_starts`, `algorithm.train_every`, and `algorithm.gradient_steps`; policy access goes through `alg.get_policy()`.

- [ ] **Step 1: Write tests** using a minimal vector environment and fake algorithm for warmup scheduling, update metrics, checkpoint save/restore, and inference policy return.
- [ ] **Step 2: Run** `pytest tests/runners/test_off_policy_runner.py -q`; confirm the runner is missing.
- [ ] **Step 3: Implement** rollout collection with the existing `VecEnv` contract, call `process_env_step` for each transition, schedule updates only after replay warmup, and use the existing `Logger` conventions for logs and checkpoints.
- [ ] **Step 4: Run** `pytest tests/runners/test_off_policy_runner.py -q`; all cases pass.

### Task 4: TD-MPC2 Neural Components

**Files:**
- Create: `rsl_rl/models/tdmpc2/init.py`
- Create: `rsl_rl/models/tdmpc2/layers.py`
- Create: `rsl_rl/models/tdmpc2/math.py`
- Create: `rsl_rl/models/tdmpc2/world_model.py`
- Modify: `rsl_rl/models/__init__.py`
- Test: `tests/models/test_tdmpc2_world_model.py`

**Interfaces:**
- Produces `TDMPC2WorldModel(obs_dim, action_dim, cfg)` exposing `encode`, `next`, `reward`, `termination`, `pi`, `Q`, and target-Q update methods, in the single-task state-observation configuration.

- [ ] **Step 1: Write tests** for world-model output shapes, categorical scalar transforms, policy action bounds, termination head on/off, and target Q parameter updates.
- [ ] **Step 2: Run** `pytest tests/models/test_tdmpc2_world_model.py -q`; confirm components are missing.
- [ ] **Step 3: Implement** the single-task state encoder, latent dynamics, reward and optional termination heads, Gaussian policy prior, Q ensemble, initialization, and scalar transform utilities against the official implementation.
- [ ] **Step 4: Run** `pytest tests/models/test_tdmpc2_world_model.py -q`; all cases pass.

### Task 5: Episode Replay Storage

**Files:**
- Create: `rsl_rl/storage/episode_replay_buffer.py`
- Modify: `rsl_rl/storage/__init__.py`
- Test: `tests/storage/test_episode_replay_buffer.py`

**Interfaces:**
- Produces `EpisodeReplayBuffer(capacity, sequence_length, batch_size, device)` with `add_step(env_id, observation, action, reward, next_observation, terminated)`, `end_episode(env_id)`, `sample()`, `state_dict()`, and `load_state_dict(state_dict)`. `end_episode` closes an episode independently of the stored `terminated` flag, so timeouts can close replay sequences while retaining bootstrap semantics.

- [ ] **Step 1: Write tests** for multi-environment episode separation, termination vs timeout behavior, eviction at capacity, sequence shape, exclusion of too-short episodes, and checkpoint restore.
- [ ] **Step 2: Run** `pytest tests/storage/test_episode_replay_buffer.py -q`; confirm the replay API is missing.
- [ ] **Step 3: Implement** episode accumulation and contiguous sequence sampling, preserving terminal flags and never sampling across episode boundaries.
- [ ] **Step 4: Run** `pytest tests/storage/test_episode_replay_buffer.py -q`; all cases pass.

### Task 6: TD-MPC2 Algorithm and Planner

**Files:**
- Create: `rsl_rl/algorithms/tdmpc2.py`
- Modify: `rsl_rl/algorithms/__init__.py`
- Test: `tests/algorithms/test_tdmpc2.py`

**Interfaces:**
- Produces `TDMPC2` with the common `construct_algorithm`, lifecycle, checkpoint, and inference methods from Task 2; `get_policy()` returns a callable policy wrapper retaining planner state and exposing `reset()`.
- Config keys use existing training dictionaries: `algorithm`, `model`, `memory`, `obs_groups`, `num_steps_per_env`, and standard runner save/log settings.

- [ ] **Step 1: Write tests** for TensorDict state extraction, random-action warmup, planner action shape/bounds, one sequence update, episodic termination handling, and full checkpoint round trip.
- [ ] **Step 2: Run** `pytest tests/algorithms/test_tdmpc2.py -q`; confirm the public algorithm API is missing.
- [ ] **Step 3: Implement** TD-MPC2 value normalization, latent rollout/critic/policy losses, target updates, and MPPI planning from the source algorithm, cross-checking termination behavior with ManiSkill3.
- [ ] **Step 4: Implement** algorithm construction, per-env episode collection, update lifecycle, train/eval modes, policy reset, and model/optimizer/replay persistence.
- [ ] **Step 5: Run** `pytest tests/algorithms/test_tdmpc2.py -q`; all cases pass.

### Task 7: Public Runner Integration, Documentation, Regression

**Files:**
- Modify: `rsl_rl/runners/__init__.py`
- Modify: `docs/api/algorithms.rst`
- Modify: `docs/api/runners.rst`
- Test: `tests/runners/test_off_policy_runner.py`
- Test: existing `tests/algorithms`, `tests/runners`, and `tests/storage` suites

**Interfaces:**
- Public imports expose `FastSAC`, `TDMPC2`, `OffPolicyReplayBuffer`, `EpisodeReplayBuffer`, and `OffPolicyRunner` through their package modules.

- [ ] **Step 1: Extend runner tests** to construct both algorithms from dictionary configs and verify final checkpoint/inference compatibility.
- [ ] **Step 2: Document** configuration keys, supported state-observation/action assumptions, runner selection, checkpoint resume, and inference policy reset behavior.
- [ ] **Step 3: Run focused verification:** `pytest tests/storage/test_off_policy_replay_buffer.py tests/storage/test_episode_replay_buffer.py tests/models/test_tdmpc2_world_model.py tests/algorithms/test_fast_sac.py tests/algorithms/test_tdmpc2.py tests/runners/test_off_policy_runner.py -q`.
- [ ] **Step 4: Run regression verification:** `pytest tests/algorithms tests/runners tests/storage -q`.
- [ ] **Step 5: Inspect** `git diff --check`, the final public exports, and that no PPO/AMP/Distillation config or behavior changed.
