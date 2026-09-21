# STRAT-01 GigaChat 3.1 upstream RMSNorm propagation diagnostic result

**Date:** 2026-09-21

**Outcome:** `UPSTREAM_DOUBLE_RMSNORM_INSUFFICIENT_FOR_PROJECTION_GATES`

**Cell:** `STRAT-01-ENGINE-UPSTREAM-RMSNORM-PROPAGATION-DIAGNOSTIC`

**Implementation revision:** `69cc7dc`

**Raw evidence:** [`strat01_gigachat_engine_upstream_rmsnorm_diag_20260921`](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_upstream_rmsnorm_diag_20260921/)

**Adjudication SHA-256:** `aebf238eb627f7eaa39f29f737d5295c22fa12a4a1fcb8bb1e88d1286d98bf80`

**Claim labels.** Numerical metrics, Q8 censuses, controls, identities, and execution counts are **MEASURED**. Pinned llama.cpp's Q5_0×Q8_0 runtime pairing is **SOURCE-DERIVED**. Selecting the K-B operator boundary as the next diagnostic is an **INFERENCE**, not a repair claim.

## Result

The frozen [protocol](STRAT_01_GIGACHAT31_ENGINE_UPSTREAM_RMSNORM_PROPAGATION_DIAGNOSTIC_PROTOCOL_20260921.md) executes the accepted block-0 C attention graph once for both schedules. Only attention-input and compressed-KV RMSNorm accumulation change from float to pinned double semantics; the final FFN normalization also uses the already measured double semantic. All reference payloads are reused, with no reference execution.

The upstream correction is real and propagates beneficially:

| boundary | accepted float NRMSE | upstream-double NRMSE | result |
|---|---:|---:|---|
| `attn_norm-0` | `7.96505745836979e-7` | `0` | exact |
| `q-0` | `7.90350041401589e-7` | `6.44789694114499e-8` | improved |
| `kv_cmpr_pe-0` | `7.90132157222658e-7` | `7.32245460150407e-8` | improved |
| `kv_cmpr-0` | `3.1624057871625e-7` | `1.81751447369191e-7` | improved |
| `kqv_out-0` | `5.63656317001493e-4` | `4.74051600440346e-4` | improved |
| `ffn_inp-0` | `5.95483462881182e-4` | `4.96891896193701e-4` | improved |

Every inherited Rung-2A tensor, cache, and continuity gate passes in both schedules. Prefill and cached outputs are identical. The attention-input Q8 representation changes in 45/48 blocks and 45/14,016 bytes; the propagated FFN-normalized representation changes in 46/48 blocks and 156/14,016 bytes.

The primary FFN gates nevertheless remain outside the frozen NRMSE limit:

| projection | candidate NRMSE | normalized maximum | frozen gate | result |
|---|---:|---:|---|---|
| up | `0.00262649240855995` | `0.00170202668779476` | `0.002 / 0.01` | FAIL |
| gate | `0.00231478705864183` | `0.0010324237770339` | `0.002 / 0.01` | FAIL |

Both schedules produce the same values. The candidate FFN-normalized input passes at NRMSE `0.000692286203953438`, but remains far enough from reference to cross Q8 boundaries and fail both projections.

## New first residual boundary

After exact `attn_norm-0` and near-exact Q4_K Q/KV projections, the first material residual is `q_nope_absorbed_perm-0`:

- candidate `q-0`: NRMSE `6.44789694114499e-8`;
- candidate `q_nope_absorbed_perm-0`: NRMSE `0.00031511401613621`;
- accepted float baseline at that output: `0.000315150388555855`.

The current C `strat01_r2a_kb_batch` dequantizes Q5_0 weights to F32 and accumulates F32 products. Pinned llama.cpp registers Q5_0 matrix multiplication with Q8_0 activation quantization and `ggml_vec_dot_q5_0_q8_0`. Rung 1 proved Q5_0 stored-row decoding and a scalar dequantized-row matvec, not this runtime operator semantic. Therefore the next cell is not a duplicate of Rung 1.

## Controls and provenance

- apparatus-only preflight closed `APPARATUS_READY_NO_DONOR_EXECUTION` with empty errors;
- accepted artifact C executions: one; pinned-reference executions: zero;
- all four prior-record hashes and both baseline payload hashes validate;
- one-byte baseline mutation is refused;
- swapped up/gate targets reject at NRMSE `1.3039360803366793` and `1.5610788415318508`;
- all three double-accumulator call sites and the omitted-coordinate source control are present;
- `errors=[]`; runtime is operational provenance only.

## Consequence and no-duplication boundary

Do not repeat this cell, patch only RMSNorm, relax the projection gates, or advance to Rung 2C. Double RMSNorm semantics are correct and improve the path, but are insufficient alone; retain them only as a candidate component pending a combined changed-coordinate confirmation.

Freeze a no-full-donor K-B operator diagnostic before any production edit. It should cross-feed immutable exact-reference and upstream-double `q-0` payloads through both the current dequant-F32 Q5_0 path and a pinned Q5_0×Q8_0 path, compare against immutable `q_nope_absorbed_perm-0`, verify activation bytes and matrix orientation, and reproduce the accepted current output exactly. If pinned semantics close this boundary, a single combined RMSNorm+K-B propagation confirmation may follow.

No production repair, complete Rung 2B, Rung 2C, quality, generation, RAM, or rate claim follows. No `SPEED_LEDGER.md` update is due.
