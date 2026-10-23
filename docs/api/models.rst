Models
======

MLP Model
---------

.. automodule:: rsl_rl.models.mlp_model
   :members:
   :undoc-members:


RNN Model
---------

.. automodule:: rsl_rl.models.rnn_model
   :members:
   :undoc-members:


CNN Model
---------

.. automodule:: rsl_rl.models.cnn_model
   :members:
   :undoc-members:


Transformer Model
-----------------

``TransformerModel`` supports ordered rank-2 vector groups and rank-3 sequence groups. List the sequence groups with
``sequence_groups``; all remaining selected groups are treated as vectors. Sequence tokens are projected to a shared
embedding width, receive sinusoidal positions, pass through independent Transformer encoders, and are pooled with
``mean``, ``max``, or ``last`` before fusion with vectors. ``causal_groups`` enables future-token masking for selected
sequences. Per-group right or arbitrary padding can be supplied to ``forward`` through ``sequence_padding_masks``;
mask entries set to ``True`` are ignored as keys and during pooling. The usual ``masks`` argument remains reserved for
RSL-RL trajectory padding. Deterministic JIT export accepts an ordered list of tensors, and ONNX export exposes one
named input per selected observation group.

.. automodule:: rsl_rl.models.transformer_model
   :members:
   :undoc-members:


Conditional VAE
---------------

.. automodule:: rsl_rl.models.vae
   :members:
   :undoc-members:


Conditional Diffusion
---------------------

.. automodule:: rsl_rl.models.diffusion
   :members:
   :undoc-members:
