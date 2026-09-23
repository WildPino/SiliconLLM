# STRAT-01 Q4_K×Q8_K AVX2 reduction-parity apparatus

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

This qualifies the apparatus for the frozen
[AVX2 reduction-parity protocol](STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY_PROTOCOL_20260923.md).
It does not measure the full immutable Q/KV projections and does not establish
downstream parity.

The canonical apparatus-only directory is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q4k_q8k_avx2_parity_apparatus_repair1_20260923/`

Its adjudication SHA-256 is
`64340e4d52b407cabcee6c73f602b11fd17d93f8504227ee7cb2a51c6b75b1f0`.

## Qualified implementation

- The standalone production dot now transcribes the pinned x86 AVX2/FMA
  `ggml_vec_dot_q4_K_q8_K` accumulation and terminal reduction order.
- The old scalar/generic reduction remains separately named and can be
  selected only by the diagnostic compile-time control; the production build
  fails closed without AVX2 and FMA.
- Q8_K quantization, packed Q4_K decoding, scales/minima, matrix layout, and
  every non-Q4 operator are unchanged.
- The independent oracle build now forces an x86_64 toolchain and verifies
  from `compile_commands.json` that the pinned
  `ggml-cpu/arch/x86/quants.c` is compiled with `-mavx2 -mfma`. This closes a
  Windows CMake trap in the historical oracle builder, where flags on the
  executable did not prevent the linked GGML library from selecting
  `GGML_CPU_GENERIC`.

## Qualification evidence

- Active standalone dot versus the true pinned active-x86 oracle: bit-exact at
  `n = 256, 512, 1536, 6144`.
- Generic diagnostic control: differs at all registered multi-block lengths
  (`512`, `1536`, and `6144`), so the test population detects the changed
  coordinate.
- Short-read and invalid-length controls reject.
- All 123 registered STRAT-01 Python tests pass, including packed-scale,
  high-nibble, minimum, Q8-scale, transpose, mutation, source-binding, and
  Rung-1/2A/2B/2C/downstream diagnostics.
- All 16 registered C self-test commands pass; the legacy kernel self-test
  contributes 73,024 checks.
- Both active and generic-control engines compile under Clang C11 with
  `-O3 -mavx2 -mfma` and no fast-math.
- Donor graph executions: 0. Reference graph executions: 0. Model reads: 0.

The earlier apparatus-only directory without the `repair1` suffix also passed
its computations with zero model reads, but its manifest omitted the runner
self-test and full-projection header from the critical-source inventory. It is
preserved as pre-qualification history and is not canonical.

Exactly one non-VOID scientific invocation is authorized after this apparatus
and implementation are committed. It must require byte-exact active Q/KV
reference hashes and exact reproduction of both historical generic hashes.
No gate may be relaxed.
