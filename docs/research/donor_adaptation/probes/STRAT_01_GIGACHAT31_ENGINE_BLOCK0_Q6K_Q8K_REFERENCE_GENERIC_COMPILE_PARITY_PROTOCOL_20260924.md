# STRAT-01 block-0 Q6_K×Q8_K reference-generic compile parity protocol

**Status:** frozen before implementation or execution.

## Question and non-duplication boundary

The closed
[active-AVX2 result](STRAT_01_GIGACHAT31_ENGINE_BLOCK0_Q6K_Q8K_AVX2_PARITY_RESULT_20260924.md)
proves exact Q8_K bytes and exact agreement between the project AVX2
transcription and pinned x86 oracle, but both differ from immutable
`ffn_out-0`. The historical scalar helper still exactly replays its own known
output. Thus another AVX2, Q8, matrix-layout, SwiGLU, residual-add, or
downstream run would duplicate closed evidence.

The changed coordinate is compiler lowering of pinned
`ggml_vec_dot_q6_K_q8_K_generic`: current project-generic arithmetic is
compiled inside the `-mavx2 -mfma` engine translation unit. The preserved
reference producer selected baseline `ggml-cpu/quants.c` with
`GGML_CPU_GENERIC`, no AVX/FMA flags. The prior Q4_K compile-parity cell proved
this distinction causal for the same reference build, but Q6_K has never been
tested under isolated baseline-generic target controls.

Question: with identical stored Q6_K and Q8_K bytes, does an exact-operation-
order, non-inlined Clang helper restricted to `no-avx,no-avx2,no-fma`
reproduce the pinned baseline-generic oracle and immutable block-0 output
byte-for-byte?

## Frozen identities

- accepted GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- matrix `blk.0.ffn_down.weight`, Q6_K `[8960,1536]`, file offset
  `292858752`, span `11289600` bytes;
- input `ffn_swiglu-0`: 286,720 bytes, SHA-256
  `de9245a2f6f60e8d7511d03fe981e6b55ef21a8634d5071dd36727a1a5950cef`;
- reference/current `ffn_inp-0`: SHA-256
  `baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1`;
- reference `ffn_out-0`: SHA-256
  `f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4`;
- historical AVX-TU generic: SHA-256
  `f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce`;
- closed active AVX2: SHA-256
  `3d0da87b33d531458bd0ca6529e1be8a715b67ff88b03ab82541f4d47fa445ec`;
- pinned llama.cpp revision
  `5b335f413e4f73b0809c4fe39af894efbcc6a0d2` and generic `quants.c`
  SHA-256 `459ecbb123f56bd9230b2681430e7591764a5587b6b83f058f002bf6511fdbad`;
- preserved reference `compile_commands.json` SHA-256
  `de27906dc35179b5efd47ffab98adb33476425ef215f9b0513ae9229454ecb80`;
- reference executable SHA-256
  `abdadf3fdac1bed75e24b1353d1fbff85159c1f6e4c3e8d81e9efe5e1e246627`.

## Arms and gates

Provide three disjoint C arms on the same Q8 population and matrix rows:

1. `reference_generic_candidate`: literal pinned generic source operation
   order, `noinline`, target `no-avx,no-avx2,no-fma`;
2. `historical_avx_tu_generic`: exact replay of
   `f52e0e5...97d6ce`;
3. `closed_active_avx2`: exact replay of
   `3d0da87...445ec`.

Build an independent pinned `GGML_CPU_GENERIC` oracle and prove through its
compile database that generic `quants.c` is selected without AVX/FMA. Require:

- bit-exact candidate/oracle stored-block parity at 256, 512, 1536, and 8960;
- byte-exact Q8 populations;
- pairwise-distinct live control outputs on at least one multi-block fixture;
- candidate full output equal to both oracle and immutable reference;
- candidate plus exact `ffn_inp-0` equal to reference `l_out-0`;
- complete downstream amplifier output equal to the immutable layer-2 target;
- generic/active historical hashes, one-byte mutation, source, compiler,
  descriptor, size, twin, and zero-graph controls all pass.

Qualify model-free first. Apparatus-only must open no GGUF or scientific
payload and must report zero model, Q8, matrix, diagnostic, and graph counts.
Commit exact sources before one zero-graph scientific invocation.

## Decisions and stop rule

- all exact gates pass:
  `BLOCK0_Q6_REFERENCE_GENERIC_EXACT_REPAIR`;
- valid candidate/oracle equality but immutable output remains non-exact:
  `BLOCK0_Q6_REFERENCE_GENERIC_COMPILE_INSUFFICIENT`;
- candidate differs from the independently compiled baseline-generic oracle:
  `BLOCK0_Q6_REFERENCE_GENERIC_IMPLEMENTATION_MISMATCH`;
- any apparatus, identity, accounting, or control failure: VOID.

PASS authorizes a separately frozen production integration of only this
helper. Any valid FAIL closes compiler lowering and requires a new causal
hypothesis. No quality, generation, RAM, rate, or full-model claim is allowed.
