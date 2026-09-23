# STRAT-01 Q4_K×Q8_K reference-generic compile-parity protocol

**Frozen:** 2026-09-23, before implementation or adjudication

**Cell:** `STRAT-01-ENGINE-Q4K-Q8K-REFERENCE-GENERIC-COMPILE-PARITY`

**Purpose:** reproduce the exact arithmetic lowering of the baseline
`GGML_CPU_GENERIC` Q4_K×Q8_K kernel used by the frozen reference producer,
without changing Q8_K bytes, packed Q4_K semantics, inputs, weights, matrix
layout, or any non-Q4 operator.

## Why this is a new coordinate

The closed
[active-AVX2 parity cell](STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY_RESULT_20260923.md)
falsified the hypothesis that the frozen Q/KV references came from the pinned
active x86 branch. Its standalone active implementation is bit-exact to the
true x86 oracle at four lengths, but full Q/KV hashes fail and are slightly
farther from reference than the historical project-generic arm. That arm
replays its old hashes exactly.

The preserved reference build supplies the missing distinction. Its
`compile_commands.json` shows that `ggml-cpu/quants.c` was compiled with
`-O3 -DGGML_CPU_GENERIC`, without `-mavx2` or `-mfma`. The project helper uses
the same algorithmic formula inside the AVX2/FMA `engine.c` translation unit.
Compiler lowering, vectorization, contraction, inlining, and target attributes
are therefore a distinct, unmeasured coordinate.

## Immutable identities

- Accepted GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Exact reference `attn_norm-0`: SHA-256
  `c8c7bd47772b1f153f28183892795b9bc91be978f5473bb322f869b2c10c1efd`.
- Reference Q/KV targets:
  `4dc1424d3f93651acaa152bd57222754c132ab3e43330d286c61e9739caff64b`
  and `6a364dd45c12142fb45ab90287caa14874ed40160089e716e23bd58d3785653c`.
- Historical project-generic Q/KV:
  `6255f40d5a717af2a742c484b0ab75ff549b37acea70f73556698c794c8b3366`
  and `5c3fdab029c1660bae7c4f4d5d256991def1ccce39468ccdb5d9b1346d5be9ce`.
- Closed active-x86 Q/KV:
  `b1d355be461cf30f42cc0acc88c4cf08963678a252be74761031a41e44978cd3`
  and `a2c177598c5ea5e7a9b5cb5707a6a738b711edd548ce6d9430eb3f66be2222f2`.
- Pinned llama.cpp revision:
  `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.
- Reference generic source `ggml-cpu/quants.c` SHA-256:
  `459ecbb123f56bd9230b2681430e7591764a5587b6b83f058f002bf6511fdbad`.
- Preserved reference `compile_commands.json` SHA-256:
  `de27906dc35179b5efd47ffab98adb33476425ef215f9b0513ae9229454ecb80`.
- Preserved reference binary SHA-256:
  `abdadf3fdac1bed75e24b1353d1fbff85159c1f6e4c3e8d81e9efe5e1e246627`.
- Reference builder/source SHA-256:
  `3e137bf69fbd3640d545ec972ea26663a9ea2056d8d0a3b51795fc4d6fd14af1`
  and `031abeb46622cc636d52fb1c2a891c5009402edb4a1aa9b5291edbc59015c9c7`.

## Changed coordinate and implementation contract

Only this coordinate may change:

> project generic Q4_K×Q8_K helper lowered in an AVX2/FMA translation unit →
> exact baseline-target generic arithmetic used by the frozen reference build.

The implementation must:

1. preserve the exact pinned generic source operation order, including eight
   lane accumulators, per-block minimum subtraction, and terminal lane sum;
2. isolate that helper from AVX/FMA contraction or inlining differences using
   explicit compiler target/function controls or an equivalently auditable
   separate object;
3. fail closed on an unsupported compiler instead of silently choosing another
   arithmetic path;
4. keep the historical AVX-translation-unit generic and active-x86 helpers as
   named diagnostic controls, never implicit fallbacks;
5. route production Q4_K matrix calls only through the candidate after its
   model-free exact gate passes;
6. compile the surrounding engine with Clang C11 `-O3 -mavx2 -mfma`, no
   fast-math, while binding and recording the candidate's effective target
   controls.

## Gates and controls

All are mandatory:

- candidate versus an independently built pinned baseline-generic oracle is
  bit-exact at `n = 256, 512, 1536, 6144` on deterministic registered inputs;
- the oracle build proves from `compile_commands.json` that exactly
  `ggml-cpu/quants.c` was selected with `GGML_CPU_GENERIC`, without AVX/FMA,
  and verifies its source hash and pinned revision;
- Q8_K payloads are byte-exact;
- the historical AVX-translation-unit generic and active-x86 controls differ
  on at least one registered multi-block fixture and reproduce their four
  frozen full-projection hashes;
- candidate full Q and KV SHA-256 values equal the immutable reference hashes;
- packed-scale, minimum, high-nibble, Q8-scale, transpose, descriptor,
  short-read, mutation, non-finite, source-inventory, and compile-mode controls
  reject;
- all existing STRAT-01 Python tests, all registered C self-tests, and the
  73,024-check legacy kernel suite remain green;
- donor and reference graph executions are zero.

## Decision and stop rule

- `PASS_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY` only if the
  model-free and full Q/KV outputs are all byte-exact and every control passes.
- `FAIL_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY` for a valid non-exact
  result.
- `VOID_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY` for any apparatus,
  identity, compile-mode, source, or control failure.

Exactly one non-VOID full-projection invocation is allowed after a fresh
apparatus-only qualification and implementation commit. PASS would authorize
a separately frozen downstream propagation confirmation; it would not prove
quality, generation, RAM, rate, or the final ≥50 tok/s goal. FAIL closes
compile-lowering parity and requires a new causal hypothesis, not gate
relaxation.
