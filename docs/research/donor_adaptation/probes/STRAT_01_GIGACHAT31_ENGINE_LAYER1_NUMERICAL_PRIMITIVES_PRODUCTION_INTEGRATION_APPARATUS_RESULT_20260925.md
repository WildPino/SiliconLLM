# STRAT-01 layer-1 numerical-primitives production-integration apparatus result

**Date:** 2026-09-25

**Cell:** `STRAT-01-ENGINE-LAYER1-NUMERICAL-PRIMITIVES-PRODUCTION-INTEGRATION`

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

## Result

The repaired model-free apparatus qualifies the production-only changes frozen
in the [protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_PRODUCTION_INTEGRATION_PROTOCOL_20260925.md).
Normal Rung-2C now delegates the F32 router reduction to the proven
F32-product/F64-accumulator reference-generic helper and both routed/shared
layer-1 SwiGLU populations to the proven SSE2/no-FMA helper. Q6 remains on its
closed reference-generic path; all other operators and layouts are unchanged.

The qualifying record is:

- raw directory:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_numerical_primitives_production_integration_apparatus_repair1_20260925/`;
- adjudication SHA-256:
  `f9c828fca2eb4bd6fac820c7e3d9b293ebd473f5c80c31b227654686a700289a`;
- observed pre-commit HEAD:
  `b2036a4e5b68b825b1a8a36263703025d28061c0`;
- all 311 registered STRAT-01 Python tests passed in 155.235 s;
- compilation plus all 22 C self-test commands passed;
- all 13 production-ownership, predecessor, unchanged-coordinate,
  configuration and protocol controls passed;
- production invocations: 0; donor graph executions: 0; reference graph
  executions: 0;
- total duration: 163.031 s;
- compiled binary SHA-256:
  `2fee798d6e651f88bb9703c3eb844b8dfaf106fcee93c2649c1dad9e963633d7`.

Key recorded source identities include engine
`8588915a5f4cb2fcc85f6edff4c8478960646a15a447b8606af60510f9cbb736`,
Rung-2C header
`6a4fe93469f3fa66a640a412c0e430727250af024ebeee27593a3151aecb61c5`,
runner `17a5e48b506bb2aa378f93f80224a4372a914d0997a50b4ba67649e43472d91c`,
tests `642ec4b99ed99adf2fcd561646b814e646c04e35fff6859a288c84e17f72238a`,
F32 helper `362abe0f3fddb6e49b55ce289cbb4212f3b5fa001b991b4a84db894e573d0ccd`,
and SwiGLU helper
`c87a7f9456807d5cca82a644cfe87a1d6b0e8c0dc08dcb7c9fc1c33e06ec42d1`.

## Preserved apparatus VOID 1

The initial apparatus directory without `repair1` is retained as VOID;
adjudication SHA-256
`25915f203a7588cc5a08c5ad745507a462c6429e7be565b1196aa55073f3afa6`.
It compiled successfully and ran 311 tests, but four historical assertions
required the newly preregistered layer-1 coordinate to remain scalar. No model,
payload, diagnostic, producer or graph ran. The protocol's apparatus-repair
addendum authorized changing only those stale assertions; production
arithmetic was unchanged between attempts.

## Authorization and non-claims

Commit the exact qualified sources before execution. Then run exactly one
normal two-schedule Rung-2C producer against the accepted artifact and reuse
the immutable reference tree. Do not run a reference graph or any local
diagnostic, and do not repeat either apparatus.

This result establishes no production fidelity, quality, generation, memory
or rate claim by itself.
