Algorithms
==========

PPO
---

.. automodule:: rsl_rl.algorithms.ppo
   :members:
   :undoc-members:


Distillation
------------

.. automodule:: rsl_rl.algorithms.distillation
   :members:
   :undoc-members:

FastSAC
------

.. automodule:: rsl_rl.algorithms.fast_sac
   :members:
   :undoc-members:

TD-MPC2
-------

.. automodule:: rsl_rl.algorithms.tdmpc2
   :members:
   :undoc-members:

Off-policy algorithm configuration
----------------------------------

FastSAC and TD-MPC2 use the normal ``rsl_rl`` algorithm construction contract and
``TensorDict`` observation groups. Select them with ``OffPolicyRunner`` and set
``algorithm.class_name`` to ``FastSAC`` or ``TDMPC2``. Both implementations currently
expect continuous, bounded actions and vector state observations; image encoders and
asymmetric TD-MPC2 observations are not supported. FastSAC can use separate ``actor``
and ``critic`` observation groups. TD-MPC2 uses the ``actor`` group as its state input.
Actions are normalized to ``[-1, 1]``. For auto-reset environments, include a
``final_observation`` TensorDict in ``extras`` when a done transition's returned
observation has already been reset; the algorithms use it for replay targets.

Replay settings are provided in ``memory`` (``capacity`` and ``batch_size``), with
collection cadence configured by ``num_steps_per_env`` and ``algorithm.train_every``.
Use ``algorithm.learning_starts`` to collect warm-up experience before updates.
FastSAC's ``actor`` and ``critic`` sections configure their networks, while TD-MPC2's
``model`` section configures the latent world model and its ``algorithm`` section
configures the planner and model updates.

The TD-MPC2 planner updates its sampling distribution from a soft weighted mean
of elite trajectories, then selects an elite trajectory using those weights.
Its value estimate uses the configured full Q ensemble.

``OffPolicyRunner.save`` checkpoints model, optimizer, replay, and iteration state;
``load`` restores that state. ``get_inference_policy`` switches to evaluation mode.
TD-MPC2 inference policies expose ``reset`` so callers can clear planner state at
episode boundaries.
