# General Transformer Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable Transformer sequence and cross-attention support to RSL-RL without changing PPO or rollout storage.

**Architecture:** Add reusable attention/position-encoding modules and a model following the existing model interface. The model will consume named TensorDict observation groups, support vector and sequence inputs, optional causal/padding masks, configurable pooling, output distributions, normalization, and deterministic export.

**Tech Stack:** Python, PyTorch, TensorDict, pytest, Ruff.

**Spec:** `../../../../chocolate_training/docs/superpowers/specs/2026-10-02-rgmt-reproduction-design.md` (shared design; local copy is in the Chocolate repository).

## Global Constraints

- Keep `rsl_rl` independent of Isaac Lab, robot terminology, RGMT, and feature-specific dimensions.
- Keep PPO, rollout storage, and optimizer behavior unchanged.
- Preserve the existing MLP, CNN, and RNN model construction/configuration paths.
- Support deterministic and stochastic forward behavior, output distribution accessors, normalization updates, and export.

## Review Focus

- Mixed rank-2 vector and rank-3 sequence observation groups retain batch alignment and configured ordering.
- Causal and padding masks block future/padded tokens and reject incompatible shapes clearly.
- Empty or malformed sequence sets and unsupported pooling modes fail with actionable errors.
- Stochastic distribution behavior and normalization match the established model contract.
- JIT/ONNX deterministic export preserves group concatenation, positional encoding, and pooling semantics.

---

### Task 1: Reusable Transformer primitives

**Files:**
- Create: `rsl_rl/modules/transformer.py`
- Modify: `rsl_rl/modules/__init__.py`
- Test: `tests/modules/test_transformer.py`

**Interfaces:**
- Produce `SinusoidalPositionalEncoding.forward(x: Tensor) -> Tensor` for `[batch, time, width]` tensors.
- Produce `CrossAttention.forward(query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Tensor | None = None, attn_mask: Tensor | None = None) -> Tensor`, preserving query length and embedding width.

- [x] Add unit cases for positional encoding shape/determinism, cross-attention output shape, and mask validation.
- [x] Implement the primitives with batch-first PyTorch attention and explicit dimension checks.
- [x] Run `pytest tests/modules/test_transformer.py -q`; expect all primitive cases to pass.

### Task 2: Generic TransformerModel

**Files:**
- Create: `rsl_rl/models/transformer_model.py`
- Modify: `rsl_rl/models/__init__.py`
- Test: `tests/models/test_transformer_model.py`
- Reference: `rsl_rl/models/mlp_model.py`, `rsl_rl/models/cnn_model.py`, `rsl_rl/models/rnn_model.py`

**Interfaces:**
- `TransformerModel(obs, obs_groups, obs_set, output_dim, sequence_groups, embedding_dim=128, num_layers=2, num_heads=4, feedforward_dim=256, pooling="mean", causal_groups=(), obs_normalization=False, distribution_cfg=None)` follows existing forward, distribution, normalization, and export APIs.
- `sequence_groups` names the selected observation groups expected to be rank 3; other selected groups must be rank 2. Encode each sequence group separately with shared configured encoder shape, pool it, then concatenate pooled and vector inputs in `obs_groups[obs_set]` order before the output head.

- [x] Add tests for vector-only, sequence-only, mixed inputs, causal/padding masks, pooling modes, output distribution accessors, and normalization updates.
- [x] Implement rank/width validation, sequence projection, positional encoding, Transformer encoder, pooling, vector fusion, and distribution head.
- [x] Add deterministic JIT/ONNX wrappers and verify exported deterministic outputs match eager outputs within tolerance.
- [x] Run `pytest tests/models/test_transformer_model.py -q` and `ruff check rsl_rl/models/transformer_model.py rsl_rl/modules/transformer.py`.

### Task 3: Public API and compatibility verification

**Files:**
- Modify: `docs/api/models.rst`
- Modify: `docs/api/modules.rst`
- Test: existing model test suite

- [x] Document the generic model and primitives, including accepted ranks and mask semantics.
- [x] Run `pytest tests/models tests/modules -q` and confirm the existing MLP, CNN, and RNN cases remain unchanged.
- [x] Inspect the diff to confirm PPO, storage, and training configuration code were not modified.
