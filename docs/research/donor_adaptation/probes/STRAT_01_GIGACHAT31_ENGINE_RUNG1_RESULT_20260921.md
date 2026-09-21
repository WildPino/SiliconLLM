# STRAT-01 GigaChat 3.1 base — C-engine quantized numerical parity rung 1

**Date:** 2026-09-21
**Outcome:** `PASS_ENGINE_RUNG1`
**Scope:** complete dequantized float32 streams and scalar stored-row matvecs for one frozen accepted-artifact tensor in each of Q4_K, Q5_0, and Q6_K. This is a numerical kernel result, not a model-operator or speed result.

## Frozen question and lineage

The [frozen rung-1 brief](../briefs/BRIEF_STRAT_01_GIGACHAT31_ENGINE_RUNG1.md) changed the engine-fidelity coordinate from rung 0's exact GGUF identity/descriptor census to decoding real quantized payloads and multiplying every stored row by one deterministic input. The [rung-0 result](STRAT_01_GIGACHAT31_ENGINE_RUNG0_RESULT_20260921.md) established identity and layout only. The [compatibility audit](../audits/STRAT_01_GIGACHAT31_ENGINE_COMPATIBILITY_AUDIT_20260921.md) leaves the GigaChat operators, tokenizer, and complete C execution path open.

- Frozen brief commit: `c802954728f61ad0a7692cfe662bcc7dc6f9f3ed`.
- Implementation/run commit: `d2f635168ff76df87def5eec1fe15b6be6a2d631`.
- Accepted artifact: `benchmarks/donor_adaptation/density/results/strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf`.
- Artifact size: **6,474,702,976 bytes**.
- Artifact SHA-256: `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Independent reference: `gguf-py/gguf/quants.py` from pinned llama.cpp commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.

The C path is an early-dispatched rung-1 command in `benchmarks/phase60/engine.c`, built with Clang C11 `-O3 -mavx2 -mfma` and without `-ffast-math`. The reference independently dequantizes the same raw GGUF spans through pinned `gguf-py`, regenerates the brief's input, and compares every row. The C program's interim `ENGINE_OUTPUT_READY_PENDING_REFERENCE` state is deliberately not a pass; only the separate reference adjudicator emits the combined `PASS_ENGINE_RUNG1`.

## Registered cells and complete decode parity

For row length `K`, the frozen input was `x[i] = (((i * 73 + 19) mod 257) - 128) / 128.0f`, `0 <= i < K`. The stored-row interpretation uses `dims[0]` as row length and the product of the remaining dimensions as row count. Counts below are independently recorded by C and the reference.

| Cell | Tensor / type | GGUF dimensions | Rows × values/row | Decoded values | C and reference canonical f32le SHA-256 | Whole-stream equality |
|---|---|---:|---:|---:|---|---|
| G-R1-Q4 | `blk.0.attn_q.weight` / Q4_K | `[1536, 6144]` | 6,144 × 1,536 | 9,437,184 | `7d70c299e806d539c452c51e7c3e366da9e917a5233085a888038bac3aa71304` | Exact |
| G-R1-Q5 | `blk.0.attn_k_b.weight` / Q5_0 | `[128, 512, 32]` | 16,384 × 128 | 2,097,152 | `a608e134a3bb3801457dbd721da9c51c31f55fd69e4afbb208e5dd4bc0781022` | Exact |
| G-R1-Q6 | `blk.0.ffn_down.weight` / Q6_K | `[8960, 1536]` | 1,536 × 8,960 | 13,762,560 | `83d5bf44945674c3ddc7e711fb90222bce91d78c2d1b0302477acb7634af50d1` | Exact |

Every decoded value was finite. For each complete stream, the canonical little-endian float32 SHA-256 from C equals the independently produced reference digest. This tests the full payload for these three tensors; it does not generalize automatically to other tensors or call shapes.

## Scalar stored-row matvec gate

For every stored row `r`, the adjudicator measured `abs(c_r - p_r) / (1 + sum_i abs(w_ri * x_i))`. The frozen acceptance limit was `2e-6` for the maximum normalized residual in each cell.

| Cell | Rows compared | Max absolute error | RMSE | Max normalized residual | Gate |
|---|---:|---:|---:|---:|---|
| G-R1-Q4 | 6,144 | `3.6954879760742188e-06` | `3.077649365094794e-07` | `1.5472069223217056e-07` | PASS |
| G-R1-Q5 | 16,384 | `0` | `0` | `0` | PASS |
| G-R1-Q6 | 1,536 | `2.002716064453125e-05` | `2.600125239945843e-06` | `1.530169167198841e-07` | PASS |

All outputs were finite. C/reference output SHA-256 values were equal for Q5_0 and differed for Q4_K and Q6_K, as permitted by the brief because the row reductions can round differently. Each C output hash matches the C raw JSON report. The maximum normalized residual, rather than output-hash identity, is the numerical gate.

## Apparatus and controls

**G-R1A — PASS.** The registered runner compiled successfully with Clang 21.1.8 for `x86_64-w64-windows-gnu`. The rung-0 suite passed **5/5** tests; rung-1 passed **4/4** tests; the codec self-test passed; and the historical `--kselftest` passed **73,024 checks**, with zero worst LUT/scalar difference and `1.191e-07` maximum relative error for `exp256_ps` versus libm. The rung-1 fixtures contain planted Q4_K/Q5_0/Q6_K layout traps. Negative controls reject a wrong early CLI shape and an unrecognized artifact before the legacy loader. The rung-0 tests also retain exact-copy acceptance, one-byte hash-tamper rejection, malformed GGUF refusals, and wrong compile-time identity refusal.

The run used Python 3.12.10 and NumPy 2.4.6 for the independent adjudicator. The reference reader revision was checked against the pinned llama.cpp commit. Both run JSON records report no errors; the reference comparator reports zero cell failures and `PASS_REFERENCE_COMPARE`, combined as `PASS_ENGINE_RUNG1`.

## Operational timings — not speed measurements

The run took place on Windows 11, beginning 11:00:26 and ending 11:01:17 local time. The manifest records **2.494971 s** for compilation, **25.822128 s** for the accepted-artifact C command, **7.674240 s** for reference adjudication, and **51.155371 s** total runner wall time. The C command performs identity/payload work for this parity estimand; these wall times are operational diagnostics, not inference latency, tok/s, a clean-box rate, or a `SPEED_LEDGER` entry.

## Provenance hashes

### Source files and executable

| Input | SHA-256 |
|---|---|
| `benchmarks/phase60/engine.c` | `a2543a9f39b527b7372ff898d4573945cd5af750f14fb4a19c60334595e3d5fb` |
| `benchmarks/phase60/strat01_gguf_inspect.h` | `3d577150ea442f5fd5ae993621024b421e000a9f66727e67b109184aa74fd711` |
| `benchmarks/phase60/strat01_gguf_rung1.h` | `59b6b93834d56a657a8246d7ff60381de7a2458f53d6cb83f0adcfd3c197c3bf` |
| `benchmarks/donor_adaptation/engine/run_strat01_engine_rung1.py` | `931e71db74503d2311f204d6241552d3ef14b267bf9bd0c9bd625f4b00087af6` |
| `benchmarks/donor_adaptation/engine/run_strat01_engine_rung0.py` | `d7a1ca7cfba53377c0b83ad53d65b3db8c355d21cad32e043f1b069682e92f90` |
| `benchmarks/donor_adaptation/engine/verify_strat01_engine_rung0_reference.py` | `8f49a10c1257dced6e33a9b989990f1c7400c2607528b7eba1fb2fc1f56acc40` |
| `benchmarks/donor_adaptation/engine/test_strat01_engine_rung0.py` | `9f4644badd47fdf4c1a1c1b1f29419e2a04de1a86ad69dc3f066a57f2a81a51c` |
| `benchmarks/donor_adaptation/engine/test_strat01_engine_rung1.py` | `9c527a075abf0380fa617c5147d9c68838918ce31dfc02f3d505af8557bbee2e` |
| `engine_rung1.exe` (416,768 bytes) | `efd40a84691f27c485b25363303124c05e94dafe8530bc35101a288ea0a2a8c7` |

### Raw run files

Directory: `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung1_20260921/`. The hashes below were read from every file in that directory; `.f32` entries are binary C row-output streams.

| Raw file | Bytes | SHA-256 |
|---|---:|---|
| `clang_version.stdout.log` | 548 | `8ccaf2d3944bce65e1c62d13712338e9b299f6ed1b1eac947c44547d63b53e02` |
| `kselftest.stdout.log` | 206 | `2a982a44b73042adb7ca00189da0e3e56d21522fe46c902b9f9dff9ea5793382` |
| `reference_compare.json` | 20,219 | `a82bb41fa3a00dbd03cfed3cc3e04fe1eafdad345ecf9fd7d07e346068e0d0f6` |
| `run_manifest.json` | 14,507 | `15510754cf3ccf7204f696065eb46b15f6616b988c5dea52447221173a45340e` |
| `rung0_unittest.stderr.log` | 958 | `9c19a3413fa26ca1433774f84502692dede4dc4d5b9fe6b25c2cdaa477281d87` |
| `rung0_unittest.stdout.log` | 102 | `8b22041a4abf77c1ed852708c42be83b9c12b8273eafcd68d23f91fdd734563b` |
| `rung1_c.stderr.log` | 56 | `48053727f86e5aec22611832381fe7e043f03d14c3a10e42c3aad9b9b48fc8e9` |
| `rung1_codec_selftest.stderr.log` | 38 | `5c38b9b4a82e87603bfca0eae3b335a75bc59f4faf44eacce3d079cc588a590a` |
| `rung1_unittest.stderr.log` | 763 | `23ccb896e9e0cf3469f700de175bc1591a3b70f66222815eb51fa0edcfe46d1c` |
| `strat01_rung1.json` | 2,914 | `4246bfe22552bd976f6ba360f6e373927aebb6276fb4828ecbeebcbec10926f5` |
| `strat01_rung1_Q4.f32` | 24,576 | `8818ea874627cb505fd8b59833309fb99a4fe829a9b3de94e8d5d7a9d4e688c9` |
| `strat01_rung1_Q5.f32` | 65,536 | `ab85d4e93c5770358d024935a76230f6b2166f71a5a37ede3df0ef6ffb7d0e5c` |
| `strat01_rung1_Q6.f32` | 6,144 | `467324a3b928c1720b0ab1b8815f6c5a733e4311c256acfdda52cd56d582cbf3` |
| Empty logs: `clang_version.stderr.log`, `compile.stderr.log`, `compile.stdout.log`, `kselftest.stderr.log`, `rung1_c.stdout.log`, `rung1_codec_selftest.stdout.log`, `rung1_unittest.stdout.log` | 0 each | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` each |

## Claim classes, adjudication, and boundary

- **MEASURED:** for these three named tensors in this hash-pinned artifact, the full canonical dequantized float32 streams match the pinned reference exactly, and every scalar stored-row output passes the frozen `2e-6` maximum normalized-residual gate.
- **SOURCE-DERIVED:** GGUF descriptor meanings and quantized block interpretations follow the pinned llama.cpp/gguf-py revision. This result tests the recorded row interpretation for the three fixed tensors only.
- **PROPOSED:** rung 2 organ/layer semantic parity is the next coordinate to freeze; no rung-2 implementation or result is included here.
- **VOID:** none for the accepted-artifact estimand. The negative fixtures are expected refusals, not void experiments.

This pass does not establish optimized-kernel performance, any attention/MLA/YaRN/MoE/shared-expert or dense-FFN operator semantics, tokenizer parity, full logits or generation, language-model quality through C, HumanEval, RAM fit, or accepted-token rate. HumanEval remains `PENDING_SANDBOX`. It makes no claim toward the ≥50 tok/s lower-CI target or 100 tok/s stretch target and does not update `SPEED_LEDGER.md`.

## No-duplication rule and next coordinate

Never repeat these three codec/scalar stored-row cells: `G-R1-Q4` (`blk.0.attn_q.weight`), `G-R1-Q5` (`blk.0.attn_k_b.weight`), or `G-R1-Q6` (`blk.0.ffn_down.weight`) on the same artifact, formats, inputs, and estimand. Their whole-stream decode and stored-row matvec gates are closed. A future changed kernel, operator, artifact, or estimand needs a new brief naming that change; it must not be presented as another rung-1 confirmation.

The next coordinate is a **separately frozen rung 2: organ/layer semantic parity**, using fixed source token IDs, positions, reference intermediates, and explicit cache/layout semantics. A likely staged plan is (a) a narrow block-0 MLA attention cell including its required projection/YaRN/cache pieces, (b) block-0 dense SwiGLU, and (c) one routed MoE layer including sigmoid/correction routing, top-4, and the always-active shared expert. The exact sequence and subcell boundaries still require a brief. This is a proposed plan only; no rung-2 implementation exists. Full-base logits, greedy generation, tokenizer, C-path quality, HumanEval, RAM, and rate remain open.

All timings here are operational only. This result is excluded from `SPEED_LEDGER.md`.
