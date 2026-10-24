# SuperMimic reference-compatible integration

The dedicated SuperMimic modules implement normalized invariant dynamics,
SO(3) state integration, autoregressive world-model fitting,
motion-encoder/history-decoder actor, privileged critic, and analytic imagined
rewards, observations and soft termination. The world model predicts state
dynamics only. CUDA training MLPs use BF16 autocast; geometry/integration and
exported inference stay FP32.

The task exposes `_supermimic_mdp: AnalyticTrackingMDP`. Construct it with
`BeyondMimicMotionDataset`, `UrdfKinematics(urdf_path, joint_names, tracking_body_names)`
and the task's default joint pose. Those ordered names come from the robot's
profile. `UrdfKinematics` is differentiable link-origin FK for predicted states;
it is not a dataset reordering layer. Neither module imports Isaac Lab or a
training project.

Task observation groups are `policy`, quaternion `state` (`13+2*num_joints`,
WXYZ), `context` (`friction, joint biases, torso COM offset`), `goal`, and scalar
`reference_bin_id`, `reference_motion_id`, `reference_frame_id`. Root velocities
are world-frame link-origin velocities. Same-step autoreset must expose
pre-reset `final_obs` (mapping or TensorDict) and distinguish terminations from
timeouts. Missing terminal observations fail explicitly instead of storing
reset jumps as dynamics.

`BaseMotionDataset` defines the shared motion and RSI interface. Format loaders
are concrete subclasses such as `BeyondMimicMotionDataset`, `SomaMotionDataset`
and `VariableDofMotionDataset`; `MotionDataset` remains an alias for the
BeyondMimic loader for existing configurations. Each dataset owns a
`ReferenceStateSampler`, defaulting to uniform frame sampling. SuperMimic
replaces that strategy with `ReturnAwareReferenceSampler`; the algorithm and
its checkpoint state access the sampler through the dataset.

`OffPolicyRunner` only invokes an algorithm's optional `initialize_env(env)`
hook. SuperMimic uses it to call the task's public
`set_reference_state_sampler(sampler, joint_layout)` API and reset once before
collection. RSI sampling and profile joint-order validation stay inside the
algorithm/task boundary; the generic runner has no SuperMimic-specific fields.

The off-policy runner honors SuperMimic's collection schedule: 64 vector steps
of warmup; UTD .4; ten model updates per iteration; 500 model-only iterations;
policy horizon 32 and history 10. The default task configurations restore the
reference network widths, batches, critic epochs and cosine LR schedule.
Other off-policy algorithms retain the generic runner path.

Replay stores physical transitions, not duplicated goals/observations.
At G1 defaults (1000 time slots × 4096 environments), replay alone is about
3.7 GiB. Full-size physics, motion caches and rollout activation memory must
also fit the device; two-environment smoke is not a full-size benchmark.

Checkpoint version 2 identifies the analytic-MDP architecture. Pre-parity
checkpoints are rejected with an explicit error. By default replay is not
saved, matching reference restart practice; resume primes a fresh buffer.
Optional replay restore marks the previous simulator session boundary done.
Actor/model/critic normalizers, optimizers, sampler and iteration counters are
saved. `max_iterations` is the target total for the SuperMimic schedule.

`OffPolicyRunner.export_policy_to_jit()` and `export_policy_to_onnx()` export
only the normalized actor. Inputs are `observation_history[B,10,O]` (oldest
first) and `goal[B,G]`; output is raw, unbounded `actions[B,A]`. The caller fills
history with the first observation at reset, then appends one observation per
control step and applies the task's action scale/default pose/bias externally.
ONNX has a dynamic batch dimension. FK and the world model are not part of the
inference artifact.

Reference source: https://github.com/zita-ch/supermimic_vibed/tree/main/supermimic
(commit a1508d42c9d1699a52e8233ab16d08c488441ab9).
