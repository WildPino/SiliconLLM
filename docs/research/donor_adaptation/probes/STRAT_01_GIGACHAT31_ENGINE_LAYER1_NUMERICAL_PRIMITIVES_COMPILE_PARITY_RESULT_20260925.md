# STRAT-01 layer-1 numerical-primitives compile-parity result

**Date:** 2026-09-25

**Cell:** `STRAT-01-ENGINE-LAYER1-NUMERICAL-PRIMITIVES-COMPILE-PARITY`

**Status:** `LAYER1_ROUTER_AND_SWIGLU_EXACT_PRIMITIVES`

## Result

The committed scientific `repair1` closes all three frozen local coordinates:

- the layer-1 router candidate using F32 products, source-order F64
  accumulation and one final F32 conversion is byte-exact to both its
  independently compiled `GGML_CPU_GENERIC` oracle and immutable reference
  logits;
- the existing four-lane SSE2/no-FMA SwiGLU helper is byte-exact to the
  immutable routed-expert SwiGLU population;
- the same helper is byte-exact to the immutable shared-expert SwiGLU
  population.

The raw directory is
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_numerical_primitives_compile_parity_repair1_20260925/`.
Its adjudication SHA-256 is
`ac04b6daeadb0ae6e71d19746585cf04fffeb74620217ed211e56484578b045a`.
Execution commit was `2923b78371ffff0c166bf6e0f6ae27eefa39ee8b`;
the binary SHA-256 was
`a1ee335730f1d2b482814cda140661873be47d91aec394f0c91a4a73a3218e1d`.

## Exact gates and controls

All seven exact gates and all eight negative controls passed. In particular:

- production router replay exactly recovered captured C output;
- router candidate and independent oracle shared SHA-256
  `8188ff0d608387dc8ce6295018c0f54a751f8a57c348d3a45bf0446770a0bbf5`,
  identical to immutable reference;
- routed SSE2 output/reference shared SHA-256
  `5097dc8599d0635477ad83f63f7e2de21b659775abc345dc933ce1e173be29e8`;
- shared SSE2 output/reference shared SHA-256
  `fa0d1b4fe5da3f561ac30b8b8cb95520f359232528ea66c1d60e9244642c6003`;
- router float-accumulator replay remained distinct from reference at NRMSE
  `5.601014943612015e-7` and normalized maximum
  `1.1549762536882241e-6`;
- the routed scalar replay remained distinct at NRMSE
  `4.51507707404167e-8`; the shared scalar replay remained distinct at
  `6.086971788375971e-8`;
- promoted-product, one-value input, one-value weight, one-value routed-gate
  and swapped gate/up controls all rejected.

Both captured schedules were exact twins before execution. The run made one
local diagnostic invocation and one accepted-model read. Producer invocations,
donor graph executions and reference graph executions were all zero. There
were no errors; total duration was 7.073 s.

## Interpretation and next boundary

The remaining layer-1 residual has now been assigned to exact, reusable
numerical primitives rather than tensor identity, routing choice, selected
experts, Q6, layouts or residual order. This is local compile parity, not yet
proof that normal Rung-2C production delegates to those helpers.

Do not repeat this cell, its apparatus, VOID 1, router alternatives or either
SwiGLU population. The only next fidelity action is the separately frozen
[production-integration protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_PRODUCTION_INTEGRATION_PROTOCOL_20260925.md).
No quality, generation, memory or rate claim follows, and `SPEED_LEDGER.md`
does not change.
