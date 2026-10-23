Algorithms

PPO
---

.. automodule:: rsl_rl.algorithms.ppo
   :members:
   :undoc-members:

AMP
---

.. automodule:: rsl_rl.algorithms.amp
   :members:
   :undoc-members:

AMP uses the AMP-Go2 discriminator update through the native PPO/TensorDict
interfaces. Each PPO minibatch samples an equal-sized policy batch from AMP
replay and an expert batch. Rollout transitions enter replay immediately.
The shared Adam optimizer applies weight decay of 0.001 to the discriminator
trunk and 0.1 to its output head. The default discriminator uses LeakyReLU,
least-squares classification loss and an expert gradient penalty with coefficient
10. Actor mean-network and critic gradients are clipped separately. Gaussian
distribution and discriminator gradients remain unclipped.

One normalizer is applied to both transition states, clipping normalized features
to [-10, 10]. Its running moments use a 0.0001 initial count and variance epsilon.
After each optimizer step, only policy and expert current states update the moments.

Configure reward mixing in the existing ``algorithm`` dictionary::

    algorithm = {
        "class_name": "AMP",
        "motion_dir": "/path/to/motions",
        "amp_reward_coef": 0.2,
        "task_reward_lerp": 0.8,
        "discriminator_hidden_dims": [512, 256],
        "amp_replay_buffer_size": 1_000_000,
    }

These reward and discriminator settings match Chocolate AMP Pose in AMP-Go2.
The mixed reward is ``task_reward_lerp * task_reward + (1 - task_reward_lerp) *
amp_reward_coef * bounded_style_reward``. The generic task fraction defaults to
zero. Legacy ``task_reward_scale`` and ``style_reward_scale`` remain usable by
converting their additive weights into this formula; they cannot be combined with
an explicit ``task_reward_lerp``.

``discriminator_learning_rate``, ``discriminator_batch_size`` and
``discriminator_updates`` are deprecated settings from the previous implementation.
The shared PPO learning rate, minibatch size and epoch count determine their
effective values. Mismatched legacy values issue a deprecation warning. Leave
additional logit regularization and loss weight decay at zero for reference
behavior. ``min_learning_rate`` controls the adaptive learning-rate floor;
optional ``min_std`` clamps actor standard-deviation parameters after each step.
Convert reference ``min_normalized_std`` to action units before passing ``min_std``.

Auto-reset environments must supply pre-reset terminal AMP states in
``extras["terminal_amp_observations"]`` or observation groups in ``extras["final_obs"]``.
Terminal AMP states can include all environments or only done environments in
done-mask order. These transitions receive style rewards and enter replay.
Missing terminal states raise an error; Isaac Lab wrappers should enable
``compute_final_obs`` and forward its observation TensorDict.


Distillation
------------

.. automodule:: rsl_rl.algorithms.distillation
   :members:
   :undoc-members:

AMP motion observation contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All motion loaders use one observation layout, concatenated in this order:

1. Root linear velocity in configured AMP axes (3).
2. Root angular velocity in configured AMP axes (3).
3. Configured joint positions.
4. Configured joint velocities.
5. Configured key-point positions minus the root position, expressed in configured AMP
   axes (3 per point, in ``key_body_names`` order).

Root height and rotation features are excluded. ``amp_coordinate_frame="world"``
is the backwards-compatible default. With ``"heading"``, the feature builder
rotates root velocities and root-relative keypoints by inverse root yaw. Z remains
world-up; roll and pitch are not removed. Velocities are absolute physical
velocities expressed in these axes, not derivatives relative to the moving frame.
The first body in the configured motion body order is the root. The environment
must produce the same feature order, joint/key-point order and velocity reference
point. Legacy ``observation_profile`` is ignored.
``amp_include_root_height`` defaults to excluded; setting it to ``True`` is an error.
Dimensions must match ``6 + joint_position_width + joint_velocity_width + 3 * key_point_count``.

The setting applies to all motion loaders and to cached expert transitions. Each
endpoint uses its own root heading. Environment-side AMP observations must use
the same axes, ordering, quaternion convention and root velocity reference point.
The exported helpers ``rsl_rl.datasets.amp_features_to_heading_frame`` and
``world_to_heading_frame`` implement the same convention for new environments.
The first accepts already assembled world-axis features, root XYZW quaternions,
and the total joint-position-plus-joint-velocity feature width. It leaves joint
coordinates unchanged. NPZ schemas, raw loaded motion tensors and RSI state
outputs are unaffected; do not pre-rotate dataset files or convert twice.
Invalid frame names raise errors. Old world-frame discriminator normalizers,
replay buffers and weights must not be resumed as heading-frame AMP state.

Input quaternion order follows the selected dataset: BeyondMimic uses WXYZ,
SOMA uses XYZW, and variable-DOF records declare ``quaternion_order`` in each
file. Dataset loaders convert body and joint quaternions to XYZW on input; RSI
states and metadata also use XYZW. Quaternions remain available for
reference-state initialization although they are excluded from discriminator
features.
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
