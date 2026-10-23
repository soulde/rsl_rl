SuperMimic
==========

SuperMimic trains a deterministic policy through a learned differentiable
world model. It uses sequence replay for supervised next-state, reward, clean
observation, and termination prediction, then optimizes the actor and value
critic on imagined trajectories. Configure it with ``OffPolicyRunner``.

The algorithm requires an Isaac Lab task with reference-state initialization.
Before constructing the runner, the unwrapped task must expose a motion dataset
as ``_supermimic_reference_dataset``. It must provide
``sample_reference_states_by_bin(bin_ids)`` and ``reference_bins``. The runner
injects ``_supermimic_rsi_sampler``, ``_supermimic_rsi_joint_layout``, and
``_supermimic_num_reference_bins`` into that task, then calls ``env.reset()``
after injection.

At reset, the task calls:

.. code-block:: python

   states, bin_ids = self._supermimic_rsi_sampler(env_ids)

It applies the returned root and joint
state using ``_supermimic_rsi_joint_layout``, records the bin IDs, and exposes
them in a top-level scalar-per-environment observation named
``reference_bin_id``. Keep this field in every observation until the next
reset; the trainer uses it to associate episode returns with reset bins.
SuperMimic requires this contract and raises an error if the field is missing.
The sampled state dictionary also includes ``reference_motion_ids`` and
``reference_frame_ids``. Tasks using reference goals can expose them in each
observation as ``reference_motion_id`` and ``reference_frame_id``.

For imagined rollouts with time-varying reference goals, define
``task._supermimic_goal_provider(motion_ids, frame_ids, count)`` before creating
the runner. It must return goals with shape ``[count, batch, goal_dim]``;
``count`` is ``horizon + 1`` so the critic also receives the final goal.
Without this optional hook, the trainer holds the sampled goal constant over
the imagined rollout.

.. code-block:: python

   train_cfg = {
       "algorithm": {
           "class_name": "rsl_rl.algorithms:SuperMimic",
           "horizon": 32,
           "gamma": 0.99,
           "gae_lambda": 0.95,
           "learning_starts": 2048,
           "gradient_steps": 1,
       },
       "model": {"hidden_dims": [256, 256, 256]},
       "actor": {"hidden_dims": [256, 256]},
       "critic": {"hidden_dims": [256, 256]},
       "memory": {
           "capacity": 200_000,
           "sequence_length": 8,
           "history_length": 10,
           "batch_size": 256,
       },
       "obs_groups": {
           "actor": ["policy"],
           "state": ["privileged_state"],
           "clean_observation": ["clean_policy"],
           "context": ["dynamics_context"],
           "goal": ["reference_goal"],
           "critic": ["clean_policy"],
       },
       "num_steps_per_env": 32,
   }

The ``actor`` and ``clean_observation`` groups must have the same dimension.
``state`` is the physical state predicted by the model. ``context`` contains
task-provided dynamics parameters such as friction or actuator calibration;
``goal`` contains reference-motion or goal features. Omit either group or map
it to an empty list when unused. Simulator-specific domain randomization stays
in the Isaac Lab task.

The return-aware sampler mixes 50% uniform bin sampling, 40% softmax sampling
favoring lower mean returns, and 10% historical softmax weights by default.
These fractions and the temperature are available when constructing
``ReturnAwareReferenceSampler`` directly. The task is responsible for applying
the sampled state; ``rsl_rl`` supplies the sampler and tracks return statistics.
