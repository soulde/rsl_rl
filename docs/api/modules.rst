Modules
=======

MLP
---

.. automodule:: rsl_rl.modules.mlp
   :members:
   :undoc-members:


RNN
---

.. automodule:: rsl_rl.modules.rnn
   :members:
   :undoc-members:


CNN
---

.. automodule:: rsl_rl.modules.cnn
   :members:
   :undoc-members:


Transformer
-----------

The Transformer building blocks operate on batch-first tensors. ``SinusoidalPositionalEncoding`` adds fixed position
features to rank-3 ``[batch, time, width]`` inputs. ``CrossAttention`` reads key/value tokens from query tokens and
accepts optional attention and key-padding masks.

.. automodule:: rsl_rl.modules.transformer
   :members:
   :undoc-members:


Normalization
-------------

.. automodule:: rsl_rl.modules.normalization
   :members:
   :undoc-members:


Distribution
------------

.. automodule:: rsl_rl.modules.distribution
   :members:
   :undoc-members:
