Algorithms
==========

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

1. Root linear velocity in world axes (3).
2. Root angular velocity in world axes (3).
3. Configured joint positions.
4. Configured joint velocities.
5. Configured key-point positions minus the root position, expressed in world
   axes (3 per point, in ``key_body_names`` order).

Root height and rotation features are excluded. Positions are not rotated into
root-local axes; velocities are absolute velocities, not relative velocities.
The first body in the configured motion body order is the root. The environment
must produce the same feature order, joint/key-point order and velocity reference
point. Legacy ``observation_profile`` is ignored.
``amp_include_root_height`` defaults to excluded; setting it to ``True`` is an error.
Dimensions must match ``6 + joint_position_width + joint_velocity_width + 3 * key_point_count``.

``motion_quaternion_format`` specifies the input file order and defaults to
``"wxyz"`` for the BeyondMimic/SOMA NPZ contract. Set it to ``"xyzw"`` for
files using that order; loading converts body quaternions to XYZW. Variable-DOF joint
quaternion blocks are converted as well, and returned RSI metadata declares
``quaternion_order="xyzw"``. Quaternions remain available for reference-state
initialization although they are excluded from discriminator features.
