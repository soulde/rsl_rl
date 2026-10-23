# FastSAC and TD-MPC2 Migration Design

## Goal

Add FastSAC and TD-MPC2 to `rsl_rl` on `feat/fastsac-tdmpc2-migration`, preserving the library's established configuration, algorithm, runner, and checkpoint conventions.

## Context and references

The starting implementations are in `/home/jvwei/robot_lab/source/rll_rl`. FastSAC will be checked against Amazon FAR's Holosoma implementation. TD-MPC2 will be checked against the official `nicklashansen/tdmpc2` implementation and ManiSkill3's `examples/baselines/tdmpc2`, with special attention to episodic termination and vectorized environments.

References:

- FastSAC: <https://github.com/amazon-far/holosoma>
- Official TD-MPC2: <https://github.com/nicklashansen/tdmpc2>
- ManiSkill3 TD-MPC2 baseline: <https://github.com/mani-skill/ManiSkill/tree/main/examples/baselines/tdmpc2>

## Design

Both algorithms are implemented as native `rsl_rl` algorithms. Their public construction entry point is `construct_algorithm(obs, env, cfg, device)`, configuration comes from the existing training configuration dictionary, and they expose the lifecycle and persistence methods expected by the library: `act`, `process_env_step`, `update`, `train_mode`, `eval_mode`, `save`, and `load`. Inference uses the repository's policy access/export conventions where the algorithm's policy representation permits it.

FastSAC uses a dedicated off-policy replay buffer and the FastSAC actor/critic update, while retaining `rsl_rl`'s environment observations, action tensors, optimizer/config resolution, and checkpoint conventions. TD-MPC2 keeps the world model, planner, and episode-based replay needed by the algorithm, but its environment interaction, configuration, logging, and persistence remain integrated with `rsl_rl`.

Off-policy learning needs scheduling and replay behavior that the current on-policy runner does not provide. Add the smallest runner support needed for the two algorithms, following the existing runner's configuration, logging, save/load, and policy-export behavior. Existing PPO, AMP, and Distillation configurations and execution remain compatible.

## Scope

- Add FastSAC and TD-MPC2 algorithm/model/storage code required for supported vectorized continuous-control environments.
- Export the algorithms using the existing `rsl_rl.algorithms` package conventions.
- Integrate training and evaluation with `rsl_rl` runner conventions, including checkpoint save and restore.
- Add focused unit tests for public construction, replay sampling/episode boundaries, algorithm updates, and runner checkpoint/inference paths.
- Document configuration and runner usage in the existing API documentation.

## Out of scope

- Porting `rll_rl` package names, config classes, or runner interfaces as a second public API.
- Multi-task/offline TD-MPC2, pixel observation support, or task-specific ManiSkill launch scripts unless required by the existing source implementation's supported target scope.
- Changing existing PPO, AMP, or Distillation behavior.
- Starting simulator training runs or downloading datasets/checkpoints.

## Compatibility and behavior

- Preserve the current `rsl_rl` configuration dictionary and callable resolution model.
- Preserve `TensorDict` observations and the `VecEnv` action/step contract.
- Keep algorithm-specific replay memory private to the relevant algorithm/storage modules; do not force replay semantics into `RolloutStorage`.
- Handle terminal and timeout signals consistently with the vectorized environment contract, and test episode boundaries explicitly for TD-MPC2.
- Save enough algorithm, optimizer, and replay state for configured training resumption; keep inference checkpoints usable through the existing runner policy interface.
- Validate incompatible observation/action shapes and algorithm settings early with clear errors.

## Validation

Run focused CPU unit tests for configuration construction, tensor shapes, one update step, terminal handling, and save/load round trips. Run the relevant existing test suite for regression coverage. Do not launch a full GPU training job as a software validation step.

## Risks

- The TD-MPC2 world model and planner have substantial implementation surface; shape conventions and termination masks require careful tests.
- FastSAC's high-throughput assumptions may not map one-to-one to `VecEnv`; the integration must preserve correctness even if throughput-oriented optimizations differ.
- The source's replay storage and checkpoint format may not match `rsl_rl`; only algorithm behavior should be transferred, not source package APIs.
