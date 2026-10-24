# SuperMimic Design

> Architecture update (2026-10-24): the initial runner-to-task sampler
> injection proposal below was superseded. `OffPolicyRunner` now only invokes
> the optional generic algorithm hook `initialize_env(env)`; SuperMimic binds
> its sampler through the task's public `set_reference_state_sampler()` API.

## Goal

Add a native SuperMimic-style model-based imitation learning algorithm to `rsl_rl`, with sequence replay, a differentiable state-space world model, an imagined-rollout actor/critic trainer, and an Isaac Lab reference-state reset protocol.

## Integration

- Add dedicated `SuperMimic`, world model, sequence replay, and return-aware reference sampler implementations. Do not change TD-MPC2 replay or model behavior.
- Construct the algorithm through the existing `construct_algorithm(obs, env, cfg, device)` path and use `OffPolicyRunner` for vector collection, logging, and checkpoint lifecycle.
- Keep `VecEnv` required methods unchanged. SuperMimic requires an Isaac Lab task-side reset protocol, but keeps sampler binding inside the SuperMimic algorithm and the task's explicit public API rather than the generic runner.

## Replay and model data

- Store simulator transitions in a fixed-capacity time-major ring buffer, with a fixed env axis and state, policy observation, previous action, reward, clean observation, termination, done, context, goal/reference features, reference-bin ID, and episode step.
- Configure dimensions and observation-group mapping explicitly. State and clean-observation targets may be the same tensor when a task has no separate clean target; context and goal may be zero-width when not used by the model.
- Train the world model to predict next state, reward, clean observation, and termination probability, conditioned on action, context, and goal/reference input.
- Sample contiguous windows with a validity mask that stops model loss after episode boundaries; sample imagination starts with observation history clamped to the current episode start.

## Trainer

- Update the supervised world model from replay sequences.
- Roll out the deterministic policy through the learned model for a configurable horizon. Freeze world-model parameters during policy optimization while preserving gradients through model inputs and predicted states to the actor.
- Train a deterministic goal-conditioned actor from a configurable observation-history window. Train the actor with survival-weighted imagined returns and train a value critic with bootstrapped GAE targets.
- Make horizons, batch sizes, loss weights, learning rates, exploration noise, discount, and GAE configurable. Defaults and tests must be CPU-friendly; implementation must not launch a training job.

## Isaac Lab reset and domain randomization

- Add a reference-state sampler alongside AMP's uniform RSI sampler. It accepts configurable mixture weights and per-reference-bin return estimates, then samples aligned reset state plus reference-bin ID. Default mixture: `0.5 / num_bins + 0.4 * softmax(-25 * bin_return / max_return) + 0.1 * normalized_historical_softmax_weights`; update historical weights by EMA on sampling and bin returns by EMA on episode completion.
- The SuperMimic algorithm binds its sampler through the task's public `set_reference_state_sampler(sampler, joint_layout)` API during `initialize_env(env)`. The task calls it during reset, applies the returned state, and publishes the selected reference-bin ID in observations or step extras. The generic runner only dispatches the optional algorithm lifecycle hook.
- SuperMimic requires the reference-bin ID contract and does not fall back to ordinary reset. The trainer accumulates episode returns by reset bin and updates sampler statistics.
- Simulator-specific domain randomization remains in the Isaac Lab task. Context values affecting dynamics are supplied by the task and stored in replay for model conditioning.

## Checkpointing and inference

- Save/load model, actor, critic, optimizers, replay, sampler probabilities/statistics, update counts, and runner iteration using existing checkpoint conventions.
- Expose deterministic inference policy through the standard runner `get_inference_policy` path.

## Validation

- CPU unit tests cover replay windows and episode boundaries, model heads and finite losses, differentiable actor updates with frozen model weights, critic targets, weighted sampling, task protocol wiring, and checkpoint round trips.
- Run focused tests and relevant formatting/type checks. No simulator or GPU training is needed for implementation validation.

## Scope boundary

This adds algorithm-side components and a runner/environment contract in `rsl_rl`; it does not modify a specific Isaac Lab task repository. A consuming task must implement the sampler reset contract and return reference-bin IDs.
