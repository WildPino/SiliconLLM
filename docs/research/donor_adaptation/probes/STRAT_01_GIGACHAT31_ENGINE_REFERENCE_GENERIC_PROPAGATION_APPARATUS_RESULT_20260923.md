# STRAT-01 reference-generic Q4 propagation apparatus result

**Verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The apparatus required by the frozen
[propagation protocol](STRAT_01_GIGACHAT31_ENGINE_REFERENCE_GENERIC_PROPAGATION_PROTOCOL_20260923.md)
is qualified. It reused and validated the immutable Rung-2C reference payloads
but executed zero donor and zero reference graphs. This result authorizes one
scientific C-producer invocation after implementation and documentation are
committed; it is not a downstream parity result.

## What the apparatus proves

- The exact reference-generic Q4 predecessor and the original canonical
  Rung-2C FAIL adjudications match their frozen SHA-256 values and states.
- The immutable Rung-2C reference root, `prefill8`, and `cached7p1` manifests
  match their three frozen hashes; every referenced tensor and cache payload
  passes the existing schema, size, digest, type, shape, operation, and
  finiteness checks.
- Production dispatch reaches the isolated reference-generic helper. The
  historical AVX-translation-unit generic and active-AVX2 implementations
  remain named controls.
- The ordinary Clang C11 `-O3 -mavx2 -mfma` engine compiles without error.
- All **131** registered STRAT-01 Python tests pass.
- All **16** registered C self-tests pass, including the legacy suite's
  **73,024 checks** with worst scalar-reference error `0`.
- `donor_producer_invocations = 0`, `donor_graph_executions = 0`,
  `reference_producer_invocations = 0`, and
  `reference_graph_executions = 0`.

The five source controls are all true:
`rung2c_source_controls`, `reference_generic_target_isolated`,
`production_dispatches_reference_generic`,
`historical_generic_control_retained`, and
`active_avx2_control_retained`.

## Reproducibility

Observed protocol commit:
`dd50c74c3255bd99497d607f1a48501894744083`

Runner:
`benchmarks/donor_adaptation/engine/run_strat01_reference_generic_propagation.py`

Raw apparatus directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_reference_generic_propagation_apparatus_20260923/`

`adjudication.json` SHA-256:
`9ab4e55936c96b021397080bf1772e180edf6a49fa18257e02b57fce26e1f410`

Compiled candidate SHA-256:
`64c9cdcfc9ba4cb525a434ce2d669f02d70d1ebbe3b72cc6dcd95bb8df0e2043`

## Scope and next action

The apparatus establishes that the changed coordinate is live, its frozen
reference evidence is reusable, and the runner is fail-closed. It does not
establish block-0 or layer-1 parity, full-model execution, quality, RAM, or
rate. The next and only authorized action is the protocol's single scientific
C-producer invocation for `prefill8` and `cached7p1`, with zero reference
graphs. The raw apparatus directory remains immutable and outside source
control.
