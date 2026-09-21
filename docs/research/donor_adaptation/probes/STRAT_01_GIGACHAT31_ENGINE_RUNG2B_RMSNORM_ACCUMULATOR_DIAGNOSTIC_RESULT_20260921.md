# STRAT-01 GigaChat 3.1 Rung-2B RMSNorm accumulator diagnostic result

**Date:** 2026-09-21

**Outcome:** `DOUBLE_RMSNORM_INSUFFICIENT_FOR_PROJECTION_GATES`

**Cell:** `STRAT-01-ENGINE-RUNG2B-RMSNORM-ACCUMULATOR-DIAGNOSTIC`

**Implementation revision:** `62572cb311ff6258008f35889c0c557ac1279ae0`

**Raw evidence:** [`strat01_gigachat_engine_rung2b_rmsnorm_diag_20260921`](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2b_rmsnorm_diag_20260921/)

**Claim labels.** Payload identities, numerical comparisons, Q8_K census, controls, and execution count are **MEASURED**. The correspondence between the double accumulator and pinned llama.cpp is **SOURCE-DERIVED** and confirmed on the exact reference payload. Selecting upstream attention RMSNorm sites as the next diagnostic boundary is an **INFERENCE**, not a repair claim.

## Result

The frozen [protocol](STRAT_01_GIGACHAT31_ENGINE_RUNG2B_RMSNORM_ACCUMULATOR_DIAGNOSTIC_PROTOCOL_20260921.md) compares the current float accumulator with pinned llama.cpp's double `ggml_float` accumulator on immutable reference and C `ffn_inp-0` payloads. It then sends the double-normalized C payload through the unchanged Q4_K×Q8_K up and gate projections. No donor graph executes.

The source-semantic check is decisive: double accumulation on the reference input reproduces the reference normalized payload exactly, while float accumulation differs at NRMSE `2.3204559139313124e-7` and normalized maximum `3.80220643306713e-7`. The mismatch is real, but it is not the dominant error on the accepted C input:

| normalized C input | NRMSE vs reference | normalized maximum |
|---|---:|---:|
| float accumulator | `0.0007914143404266998` | `0.0006736163184616577` |
| double accumulator | `0.0007914140992836706` | `0.0006736163184616577` |
| double vs float C | `1.897674285721251e-7` | `3.8019084059434475e-7` |

Both primary double-C projections remain outside the frozen NRMSE gate:

| projection | double-C NRMSE | normalized maximum | frozen gate | result |
|---|---:|---:|---|---|
| up | `0.0030612606456965185` | `0.0017181935481242877` | `0.002 / 0.01` | FAIL |
| gate | `0.002691405701711681` | `0.001061259056391299` | `0.002 / 0.01` | FAIL |

The float baselines are respectively `0.0030612619849613985` and `0.0026914051074363363`; changing only the FFN RMSNorm accumulator therefore has negligible effect. It changes 39 of 48 serialized Q8_K blocks, but only 39 of 14,016 bytes.

## Controls and provenance

- float C normalization, up, and gate replay byte-for-byte;
- all frozen baseline metrics reproduce within `1e-12`;
- a one-byte input mutation is refused;
- swapped projection targets reject strongly at NRMSE `1.3039452633952808` and `1.5610843979958677`;
- artifact, input, source, output, and Q8 payload identities validate;
- `donor_graph_executions=0` and `errors=[]`;
- the accepted diagnostic runtime was 42.513 s, an operational record only.

## Consequence and no-duplication boundary

Do not repeat this cell, relax its gates, alter the accepted Q4_K matrix path, or install a production repair limited to the FFN RMSNorm accumulator. The double accumulator is the pinned source semantic and exactly repairs reference-input normalization, but the accepted C `ffn_inp-0` already contains nearly all of the discrepancy before FFN normalization.

The next admissible boundary is upstream of `ffn_inp-0`, inside the already accepted attention path. A separately frozen diagnostic should test pinned double accumulation at the layer-0 attention-input RMSNorm and compressed-KV RMSNorm, retain all existing attention tensor/cache gates, reuse immutable reference traces, and avoid a Rung-2B or reference rerun. Only if that changed coordinate materially reduces terminal `ffn_inp-0` error and closes the inherited up/gate projections may a production RMSNorm repair be proposed.

No production repair, complete Rung 2B, Rung 2C, quality, generation, RAM, or rate claim follows. No `SPEED_LEDGER.md` update is due.
