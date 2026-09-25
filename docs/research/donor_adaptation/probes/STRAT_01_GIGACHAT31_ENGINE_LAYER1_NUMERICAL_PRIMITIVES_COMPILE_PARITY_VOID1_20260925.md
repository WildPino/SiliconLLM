# STRAT-01 layer-1 numerical-primitives compile parity — VOID 1

**Date:** 2026-09-25

**Status:** `VOID_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY`

## Disposition

The first scientific invocation is invalid and must not be adjudicated as a
primitive result. Its raw directory is preserved at
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_numerical_primitives_compile_parity_20260925/`;
the adjudication SHA-256 is
`8b6200eecffd445cb96442b26b79fa7b9b66adf6c48d4e854f68578d3b5d0162`.

At commit `2e255eba96288e0a6562f559fc08688333562ea9`, one local diagnostic
invocation and one accepted-model read completed. Producer invocations, donor
graph executions and reference graph executions were all zero. The run then
failed the exact production-router replay gate: the nominal scalar replay
changed 373/512 values relative to captured C output, with maximum absolute
difference `1.43051147e-6`.

Two router one-ULP mutation controls were also dead. This is an apparatus
failure, not evidence against either primitive. Descriptive inspection found
candidate/oracle, router candidate/reference, routed SSE2/reference and shared
SSE2/reference all byte-exact; those observations remain non-evidentiary
because the frozen replay/control gates did not pass.

The root cause and sole permitted repair are frozen in
[addendum A](STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY_ADDENDUM_A_20260925.md).
Do not overwrite, recover or reinterpret this directory.
