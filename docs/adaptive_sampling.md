# Adaptive reference sampling

`CategoricalSampler(num_bins, device)` draws uniform bins when no weights are supplied, or categorical bins from a supplied nonnegative finite distribution. It owns no adaptive statistics. Explicit distributions use multinomial without an extra normalization, preserving the existing numerical and RNG path.

`SamplingTracker` defines read-only `probabilities`, `state_dict`, and `load_state_dict`. Feedback formats and update timing belong to each method.

- `PHPFailureTracker.observe_failures` accumulates failed-bin counts; `advance` folds them into the EMA once per control step. Probabilities retain the right-padded temporal convolution and uniform floor. The environment still maps failed frames to global bins and bins back to reference frames, preserving terrain pairing and reset ordering.
- `SuperMimicReturnTracker.update_returns` retains the batched return EMA (including duplicate-bin semantics). `prepare_sample` updates historical weights immediately before sampling. Episode return accumulation and truncation bootstrap remain in the SuperMimic algorithm. `ReturnAwareReferenceSampler` remains the dataset-facing adapter, decoding sampled bins and preserving the existing flat checkpoint format.

Ordinary tracking remains uniform. This refactor does not enable adaptive sampling for that task. Tracker state APIs permit persistence; PHP training checkpoint integration is unchanged and does not automatically save the new tracker.

Validation: existing SuperMimic RSI tests (5 passed), PHP formula and seeded categorical draw comparison, tracker state roundtrip. No simulator or training process was started or modified.
