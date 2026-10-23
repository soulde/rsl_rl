# SuperMimic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SuperMimic sequence replay, differentiable world model, imagined actor/critic trainer, and an Isaac Lab return-aware RSI sampler interface.

**Architecture:** Implement the model, replay, and sampler as separate modules and compose them in a native `SuperMimic` algorithm. Reuse `OffPolicyRunner` for collection and checkpointing, adding only the SuperMimic-specific required reset protocol wiring.

**Tech Stack:** Python 3.9+, PyTorch, TensorDict, pytest, existing `rsl_rl` algorithm/model/storage conventions.

**Spec:** `docs/superpowers/specs/2026-10-23-supermimic-design.md`

## Global Constraints

- Keep `VecEnv` required methods unchanged.
- Do not change TD-MPC2 model or replay behavior.
- SuperMimic requires and validates a reference-bin ID from the task reset protocol; no ordinary-reset fallback.
- Simulator-specific domain randomization remains in the Isaac Lab task.
- All algorithm tests run on CPU; do not start simulator or GPU training.
- Save/load model, actor, critic, optimizers, replay, sampler statistics, update counts, and runner iteration.

## Review Focus

- Autoreset environments can supply final observations and immediate reset observations; test replay uses the final state for transition learning and reset state for the next episode.
- Episode windows must never cross termination or timeout boundaries; test both terminal and truncated episode closure.
- Missing or malformed reference-bin IDs must fail clearly because the algorithm requires the task protocol; test constructor and transition validation.
- Imagined actor gradients must flow through frozen world-model operations without changing world-model weights; test actor gradient and parameter immutability.
- Return-aware sampling must handle equal returns, unobserved bins, and zero mixture components without NaN probabilities; test normalized finite distributions and returned bin IDs.

---

### Task 1: Sequence replay and return-aware reference sampler

**Files:**
- Create: `rsl_rl/storage/supermimic_replay_buffer.py`
- Modify: `rsl_rl/storage/__init__.py`
- Create: `rsl_rl/datasets/supermimic_rsi.py`
- Modify: `rsl_rl/datasets/base_motion_dataset.py`
- Modify: `rsl_rl/datasets/__init__.py`
- Test: `tests/storage/test_supermimic_replay_buffer.py`
- Test: `tests/datasets/test_supermimic_rsi.py`

**Interfaces:**
- Produce `SuperMimicReplayBuffer(capacity, sequence_length, batch_size, device)` with `add_step(env_id, transition)`, `end_episode(env_id)`, `sample() -> SuperMimicSequenceBatch`, `state_dict()`, and `load_state_dict(state_dict)`.
- `SuperMimicSequenceBatch` contains `[B,T+1,D]` states/observations/clean observations, `[B,T,...]` actions/rewards/terminated/context/goals/bin IDs.
- Add `BaseMotionDataset.sample_reference_states_by_bin(bin_ids) -> dict[str, Tensor]`, where bins are one-second motion intervals and sampled frame offsets are uniform within each selected interval.
- Produce `ReturnAwareReferenceSampler(dataset, num_bins, uniform_fraction=0.5, low_return_fraction=0.4, historical_fraction=0.1, temperature=25.0)` with `sample_reference_states(env_ids) -> (state_dict, bin_ids)`, `update_returns(bin_ids, episode_returns)`, and serializable state.

- [ ] **Step 1: Write failing replay tests** for per-env append, complete-window sampling, no boundary crossing, FIFO capacity, timeout closure, and state-dict round trip.
- [ ] **Step 2: Run** `pytest tests/storage/test_supermimic_replay_buffer.py -q`; expect failures because the module does not exist.
- [ ] **Step 3: Implement** transition validation, per-env episode staging, bounded complete-episode replay, fixed-length sampling, and serialization.
- [ ] **Step 4: Run** `pytest tests/storage/test_supermimic_replay_buffer.py -q`; expect all tests to pass.
- [ ] **Step 5: Write failing sampler tests** for aligned returned states/bin IDs, default 50/40/10 mixture, finite normalized probabilities with equal/unobserved returns, updates, and checkpoint round trip.
- [ ] **Step 6: Run** `pytest tests/datasets/test_supermimic_rsi.py -q`; expect failures because sampler is absent.
- [ ] **Step 7: Implement** one-second interval indexing and `sample_reference_states_by_bin` in `BaseMotionDataset`, preserving aligned root/joint/reference fields and existing RSI frame filtering.
- [ ] **Step 8: Implement** weighted bin selection and delegate selected-bin frame sampling to that dataset method.
- [ ] **Step 9: Run** `source ~/env_isaaclab/bin/activate && python -m pytest tests/datasets/test_supermimic_rsi.py tests/datasets/test_variable_dof_motion_dataset.py -q`; expect all tests to pass.
- [ ] **Step 10: Export** both public storage and dataset classes and run import tests.

### Task 2: SuperMimic world model and actor/critic modules

**Files:**
- Create: `rsl_rl/models/supermimic/world_model.py`
- Create: `rsl_rl/models/supermimic/policy.py`
- Create: `rsl_rl/models/supermimic/critic.py`
- Create: `rsl_rl/models/supermimic/__init__.py`
- Modify: `rsl_rl/models/__init__.py`
- Test: `tests/models/test_supermimic.py`

**Interfaces:**
- `SuperMimicWorldModel(state_dim, action_dim, observation_dim, context_dim=0, goal_dim=0, hidden_dims=(256,256,256))`; `predict(state, action, context, goal) -> dict(next_state, reward, clean_observation, termination_logit)`; `loss(sequence_batch) -> dict[str, Tensor]`.
- `SuperMimicActor(observation_dim, action_dim, goal_dim=0, hidden_dims=(256,256))`; `forward(observation, goal=None) -> action` bounded to `[-1,1]`.
- `SuperMimicCritic(input_dim, hidden_dims=(256,256))`; `forward(observation) -> value`.

- [ ] **Step 1: Write failing tests** for output shapes, bounded deterministic actions, each world-model head loss, finite gradients, and serialization.
- [ ] **Step 2: Run** `pytest tests/models/test_supermimic.py -q`; expect import failures.
- [ ] **Step 3: Implement** MLP modules and supervised next-state/reward/clean-observation/termination losses.
- [ ] **Step 4: Run** `pytest tests/models/test_supermimic.py -q`; expect all tests to pass.

### Task 3: SuperMimic algorithm and imagined policy optimization

**Files:**
- Create: `rsl_rl/algorithms/supermimic.py`
- Modify: `rsl_rl/algorithms/__init__.py`
- Test: `tests/algorithms/test_supermimic.py`

**Interfaces:**
- Implement standard lifecycle `construct_algorithm`, `act`, `process_env_step`, `update`, `train_mode`, `eval_mode`, `get_policy`, `save`, and `load`.
- Constructor consumes model/actor/critic, replay, sampler, observation-group config, optimizer settings, imagination horizon, gamma, and GAE lambda.
- `update()` returns scalar model/actor/critic loss metrics. During actor updates, world-model parameters have `requires_grad=False`, while autograd through model computations remains enabled.

- [ ] **Step 1: Write failing algorithm tests** for config construction, env collection, reference-bin validation, world-model updates, imagined actor gradients with unchanged model parameters, GAE critic targets, and checkpoint restoration.
- [ ] **Step 2: Run** `pytest tests/algorithms/test_supermimic.py -q`; expect failures due to missing algorithm.
- [ ] **Step 3: Implement** collection semantics, model update, survival-weighted imagination, GAE critic update, and deterministic policy adapter.
- [ ] **Step 4: Run** `pytest tests/algorithms/test_supermimic.py tests/algorithms/test_tdmpc2.py tests/algorithms/test_fast_sac.py -q`; expect all tests to pass.

### Task 4: Runner protocol, config example, and docs

**Files:**
- Modify: `rsl_rl/runners/off_policy_runner.py`
- Modify: `rsl_rl/runners/__init__.py` only if a dedicated runner is required by discovered conventions.
- Create: `docs/guide/supermimic.rst`
- Modify: `docs/guide/configuration.rst`
- Test: `tests/runners/test_off_policy_runner.py`
- Test: `tests/algorithms/test_supermimic.py`

**Interfaces:**
- Before first rollout, runner detects SuperMimic's required RSI contract, attaches `_supermimic_rsi_sampler` and reference-bin metadata to `env.unwrapped`, and resets the task after injection.
- Fail with a clear error if an algorithm requiring RSI is paired with an environment wrapper that cannot expose `unwrapped` or if the task fails to return bin IDs.

- [ ] **Step 1: Write failing runner integration test** proving sampler injection precedes initial reset and missing task protocol fails clearly.
- [ ] **Step 2: Run** `pytest tests/runners/test_off_policy_runner.py -q`; expect the new test to fail.
- [ ] **Step 3: Implement** the minimal optional algorithm runner protocol and add a sample Isaac Lab configuration documenting context, clean observations, and bin IDs.
- [ ] **Step 4: Run** `pytest tests/runners/test_off_policy_runner.py tests/algorithms/test_supermimic.py -q`; expect all tests to pass.

### Task 5: Full focused verification

**Files:**
- No additional source files; fix issues found in Tasks 1–4.

- [ ] **Step 1: Run** `pytest tests/storage/test_supermimic_replay_buffer.py tests/datasets/test_supermimic_rsi.py tests/models/test_supermimic.py tests/algorithms/test_supermimic.py tests/runners/test_off_policy_runner.py -q`.
- [ ] **Step 2: Run** project formatting and lint commands discovered from `pyproject.toml`.
- [ ] **Step 3: Review** the complete branch diff for API consistency, CPU-only validation, and no changes to TD-MPC2 behavior.
- [ ] **Step 4: Report** test results, branch name, and the required Isaac Lab task-side contract; do not push unless separately requested.
