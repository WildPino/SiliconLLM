# STRAT-01 block-0 Q6_K×Q8_K AVX2 parity result

**Verdict: `BLOCK0_Q6_AVX2_REDUCTION_INSUFFICIENT`.**

The Q8_K quantizer is byte-exact, and the standalone C transcription of
pinned llama.cpp's active x86 AVX2/FMA `ggml_vec_dot_q6_K_q8_K` is byte-exact
to a clean pinned oracle. Nevertheless, both differ from the immutable
reference graph's block-0 `ffn_out-0`. Installing active AVX2 semantics is
therefore not an exact fidelity repair.

This result is the canonical offline re-adjudication of the complete evidence
preserved by [VOID 1](STRAT_01_GIGACHAT31_ENGINE_BLOCK0_Q6K_Q8K_AVX2_PARITY_VOID1_20260924.md).
It performs zero new model, Q8, matrix, diagnostic, donor-graph, or reference-
graph executions.

## Immutable evidence

- accepted GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- scientific implementation HEAD:
  `e46205390b2e4117ae8a63b1946f5b2c2773d6e1`;
- original scientific adjudication SHA-256:
  `c38fad6017cb1524bc92025410cd16c70398c5387a2cb9357a9fd801aa42cad2`;
- canonical offline adjudication SHA-256:
  `cd5e10de35e7c278cf22086392fc421812e1b637b252f7258d3e8e8bb22ef672`;
- C/oracle Q8 population SHA-256:
  `bac09497e73a1eee2a7bed0f151a741a0d98e584e0d423b8665ef2048724241f`;
- C/oracle active-AVX2 output SHA-256:
  `3d0da87b33d531458bd0ca6529e1be8a715b67ff88b03ab82541f4d47fa445ec`;
- immutable reference `ffn_out-0` SHA-256:
  `f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4`;
- historical generic replay SHA-256:
  `f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce`.

## Measured propagation

| Boundary | changed floats | NRMSE | normalized max | maximum absolute delta |
|---|---:|---:|---:|---:|
| AVX2 `ffn_out-0` vs reference | 11,009 / 12,288 | `2.6890881292569316e-7` | `1.9443275255492414e-7` | `3.725290298461914e-8` |
| AVX2 `l_out-0` vs reference | 10,039 / 12,288 | `1.5608138818863224e-7` | `8.52435968760745e-8` | `3.725290298461914e-8` |
| complete downstream vs reference | 188 / 49,152 | `1.467939211440746e-7` | `4.900371526567658e-7` | `5.587935447692871e-8` |

Active AVX2 reduces the earlier generic residual dramatically, but exactness
is the preregistered gate. The downstream output SHA-256 is
`b6d6d8b2e43ffa14637db5c767b08f9db250977bb6c1e6d4456e88a0f3483e96`,
not the immutable target.

## Controls and interpretation

All non-exact-repair controls pass: Q8 equality, C/oracle equality, historical
generic replay, generic-negative separation, one-byte Q6 mutation, schedule
twins, identities, and zero graphs. The result therefore excludes a Q8
packing error and excludes an incorrect transcription of the active x86
`vec_dot` path.

The preserved reference build already proves `GGML_CPU_GENERIC` compilation
without AVX/FMA. The analogous Q4_K investigation showed that compiling the
generic operation inside the AVX2 engine translation unit changes rounding,
while an isolated non-inlined `no-avx,no-avx2,no-fma` helper restores exact
reference parity. Q6 has not yet tested that compile/lowering coordinate.

## Decision and next gate

Close active Q6 AVX2 parity permanently; do not rerun it and do not install it
as an exact repair. The sole next fidelity coordinate is the separately frozen
[Q6 reference-generic compile-parity protocol](STRAT_01_GIGACHAT31_ENGINE_BLOCK0_Q6K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_PROTOCOL_20260924.md).
It preserves the same Q8 bytes, Q6 rows, inputs, matrix, and downstream
amplifier, changing only the compiler target/lowering of the exact pinned
generic operation structure.

No quality, generation, RAM, throughput, or full-model repair claim follows.
