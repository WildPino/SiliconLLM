# STRAT-01 layer-1 numerical-primitives compile-parity apparatus result

**Date:** 2026-09-25

**Cell:** `STRAT-01-ENGINE-LAYER1-NUMERICAL-PRIMITIVES-COMPILE-PARITY`

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

## Result

The repaired model-free apparatus qualifies the standalone diagnostic frozen
in the [protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY_PROTOCOL_20260925.md).
It builds the router candidate in the production AVX2 translation unit, an
independent `GGML_CPU_GENERIC` C++ oracle without AVX/FMA flags, and reuses the
already qualified SSE2/no-FMA SwiGLU helper. It does not modify or invoke
`engine.c`.

The qualifying record is:

- raw directory:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_numerical_primitives_compile_parity_apparatus_repair1_20260925/`;
- adjudication SHA-256:
  `c541185125115a9af73e026d51bf5fa059451fcd6c84d36eb3b5e2fe0a9bad15`;
- observed pre-commit HEAD:
  `b1f7b0f89b276ccc4b907334fce83b1a01885024`;
- four registered Python tests passed;
- the C/C++ synthetic self-test passed, including exact candidate/oracle
  router parity and live float-accumulator, promoted-product and scalar-SwiGLU
  negative controls;
- all nine source, compiler-path, descriptor and ownership controls passed;
- diagnostic invocations: 0; model reads: 0; producer invocations: 0; donor
  graph executions: 0; reference graph executions: 0;
- total duration: 0.908 s;
- compiled binary SHA-256:
  `ff9c299d86098ffdf579fadffbc5884f22dbb6ae651892d6266988e1e601d53b`.

Recorded source identities include runner
`5132cf89cbfc66715fbaec530caf6b4432c1bf8850451317e8c7fdd5ae923aee`,
tests `ef527fb51728ee29126e1bf695e22f295fa385fc6ac824a94d58d63463b165b8`,
probe `f06aee12fef91df7d515565731d6255c6a7ed9d56e9003839ca416105b2ef44e`,
independent oracle
`15d9c3c2d397e09079a0a55d7c78f0b466214c0c5612dab84ef63bfec2653f61`,
and candidate header
`362abe0f3fddb6e49b55ce289cbb4212f3b5fa001b991b4a84db894e573d0ccd`.

## Preserved VOID attempt

The initial apparatus directory without `repair1` is intentionally retained as
`VOID`. Compilation stopped before any executable existed because both local
includes climbed one directory too far. It performed no diagnostic, model
read, producer invocation or graph execution. Repair 1 changed only those two
include paths and the output directory; it did not alter either numerical
primitive or a scientific gate.

## Authorization and non-claims

Commit the exact qualified sources before execution. Then run exactly one
zero-graph local diagnostic against the accepted GGUF and immutable captured
layer-1 payloads. The run must compare the F32-product/F64-accumulator router
candidate with its independent oracle and reference logits, and compare the
SSE2/no-FMA SwiGLU helper with both routed and shared reference outputs.

This apparatus result establishes no donor parity, production integration,
Q6 propagation, later-layer, quality, generation, memory or rate claim. Do
not rerun the apparatus.
