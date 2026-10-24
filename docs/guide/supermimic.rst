SuperMimic
==========

SuperMimic trains a deterministic policy through a learned differentiable
world model. It uses sequence replay for supervised next-state, reward, clean
observation, and termination prediction, then optimizes the actor and value
critic on imagined trajectories. Configure it with ``OffPolicyRunner``.

The algorithm requires an environment whose unwrapped task exposes its analytic
MDP as ``_supermimic_mdp`` and implements the public
``set_reference_state_sampler(sampler, joint_layout)`` method. The runner only
calls the algorithm's optional ``initialize_env(env)`` lifecycle hook; the
SuperMimic algorithm uses that hook to pass its sampler through the task API and
reset the environment before collection. The generic runner does not inspect
SuperMimic or RSI fields.

The task's motion command stores the sampler and joint layout locally. At reset,
it calls:

.. code-block:: python

   states, bin_ids = self._reference_state_sampler(env_ids)

It applies the returned root and joint
state using the supplied ``joint_layout``, records the bin IDs, and exposes
them in a top-level scalar-per-environment observation named
``reference_bin_id``. Keep this field in every observation until the next
reset; the trainer uses it to associate episode returns with reset bins.
SuperMimic requires this contract and raises an error if the field is missing.
The sampled state dictionary also includes ``reference_motion_ids`` and
``reference_frame_ids``. Tasks using reference goals can expose them in each
observation as ``reference_motion_id`` and ``reference_frame_id``.

.. code-block:: python

   train_cfg = {
       "algorithm": {
           "class_name": "rsl_rl.algorithms:SuperMimic",
           "horizon": 32,
           "gamma": 0.99,
           "gae_lambda": 0.95,
           "warmup_steps": 64,
           "utd": 0.4,
           "wm_only_iters": 500,
           "wm_batches": 10,
           "policy_batch_size": 4096,
           "critic_epochs": 5,
           "critic_minibatches": 4,
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
