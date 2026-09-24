# STRAT-01 block-0 Q6_K×Q8_K AVX2 parity — full apparatus repair 1

**Status: `APPARATUS_READY_NO_SCIENTIFIC_EXECUTION`.** This record supersedes
the primitive-only apparatus qualification for authorization of the
scientific cell. It remains a mechanical readiness result and makes no model,
quality, downstream, or throughput claim.

## Canonical record

The qualifying raw directory is
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q6k_q8k_avx2_parity_apparatus_repair1_20260924`
(unversioned). Its adjudication SHA-256 is
`c0771eb574a281c5a98ec858694f1b88fe3b0ef13f2d2d193a47592c9261a584`.
It observed HEAD `2a72b3a03614674c35d8917e2282d5cd8b1ced2b`; exact
source hashes are recorded in the adjudication.

All ten controls pass:

- the literal AVX2/FMA primitive is bit-exact to clean pinned llama.cpp on
  1, 2, 6, and 35 stored Q6_K blocks;
- a complete synthetic matrix schedule emits bit-exact Q8_K populations and
  token-major matrix outputs in the C probe and pinned oracle;
- the scalar/generic path remains a live negative control;
- one copied Q6 byte mutation changes the output;
- short reads and misaligned lengths reject;
- pinned x86 `quants.c` has SHA-256
  `99a98747c1ac84ec40e2d1a31227b947aeb0a05bf1d8e79ec630a6d783b89f86`;
- the full `engine.c` diagnostic compiles with Clang C11
  `-O3 -mavx2 -mfma` and its five-check self-test passes.

The three full-matrix/stored-block tests pass in 24.387 seconds. Compilation
and test duration are apparatus costs, not performance measurements. Every
scientific counter is zero: model reads, scientific payload reads, Q8
population emissions, Q6 matrix reads, diagnostic executions, donor graphs,
and reference graphs.

## Authorized next step

Bind this adjudication hash in the scientific runner and commit every critical
source. Then execute the frozen scientific cell exactly once. It may read the
accepted GGUF and immutable payloads, but it must execute zero donor and
reference graphs. The decision is restricted to the three frozen outcomes:
Q8 quantizer mismatch, AVX2 reduction insufficient, or exact AVX2 repair.
