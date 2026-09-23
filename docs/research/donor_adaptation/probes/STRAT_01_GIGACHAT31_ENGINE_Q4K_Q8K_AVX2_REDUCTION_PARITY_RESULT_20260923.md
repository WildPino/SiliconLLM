# STRAT-01 Q4_K×Q8_K AVX2 reduction-parity result

**Verdict:** `FAIL_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY`

The frozen hypothesis is false: replacing the standalone generic reduction
with a bit-exact transcription of the pinned active x86 AVX2/FMA kernel does
not reproduce either immutable block-0 reference projection byte-for-byte.
This is a valid FAIL, not an apparatus VOID, and closes this exact repair
route.

Protocol:
[Q4_K×Q8_K AVX2 reduction parity](STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY_PROTOCOL_20260923.md).
Qualified apparatus:
[apparatus result](STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY_APPARATUS_RESULT_20260923.md).

The canonical raw directory is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q4k_q8k_avx2_parity_20260923/`

The execution was bound to commit
`08315705c84f788033ed6f3366df23a97b58f847`. Its adjudication SHA-256 is
`ac192a94227b12c2adf537d93017297371d77df5238945b2c32bb947e62633e2`.

## Primary result

| Arm | Q SHA-256 | KV SHA-256 | Required identity | Result |
|---|---|---|---|---|
| active x86 AVX2/FMA transcription | `b1d355be461cf30f42cc0acc88c4cf08963678a252be74761031a41e44978cd3` | `a2c177598c5ea5e7a9b5cb5707a6a738b711edd548ce6d9430eb3f66be2222f2` | immutable reference Q/KV | FAIL / FAIL |
| historical generic diagnostic control | `6255f40d5a717af2a742c484b0ab75ff549b37acea70f73556698c794c8b3366` | `5c3fdab029c1660bae7c4f4d5d256991def1ccce39468ccdb5d9b1346d5be9ce` | accepted historical generic hashes | exact / exact |
| immutable reference | `4dc1424d3f93651acaa152bd57222754c132ab3e43330d286c61e9739caff64b` | `6a364dd45c12142fb45ab90287caa14874ed40160089e716e23bd58d3785653c` | — | — |

Against the immutable reference, active AVX2 has Q/KV NRMSE
`8.9018423797e-8` / `1.0471660684e-7`; the historical generic control has
`6.4478969411e-8` / `7.3224546015e-8`. Thus active AVX2 is not only non-exact,
but is slightly farther from the frozen reference on both matrices.

## Controls and interpretation

- The active standalone dot is bit-exact against the true pinned active-x86
  oracle at `n = 256, 512, 1536, 6144`.
- The generic negative control differs on every multi-block fixture and
  exactly reproduces both historical full-projection hashes.
- All 123 STRAT-01 Python tests, all registered C self-tests, and the 73,024
  legacy kernel checks pass.
- Pinned source, model, input, reference, protocol, and implementation
  identities pass. Donor/reference graph executions are zero; the two model
  reads are the active and generic standalone projections.

The failure therefore is not explained by a bad AVX2 transcription, changed
input, changed Q8 bytes, or historical-output drift. The frozen reference
producer's preserved `compile_commands.json` establishes that its llama.cpp
backend compiled `ggml-cpu/quants.c` with `-O3 -DGGML_CPU_GENERIC`, without
`-mavx2` or `-mfma`; that source has SHA-256
`459ecbb123f56bd9230b2681430e7591764a5587b6b83f058f002bf6511fdbad`.
The project generic control uses the same algorithmic formula inside an
AVX2/FMA translation unit. Compiler lowering and contraction are therefore a
remaining changed coordinate. This is a lineage finding, not yet proof that
baseline-generic lowering alone is sufficient.

## Stop rule

Do not rerun this cell, relax exact gates, or install the active-AVX2 path as a
parity repair. Any successor must be separately frozen and target the exact
resolved backend/build arithmetic used by the reference producer. It must
first establish bit identity model-free and retain the historical generic arm
as a causal control. This result makes no downstream, quality, RAM, or rate
claim.
