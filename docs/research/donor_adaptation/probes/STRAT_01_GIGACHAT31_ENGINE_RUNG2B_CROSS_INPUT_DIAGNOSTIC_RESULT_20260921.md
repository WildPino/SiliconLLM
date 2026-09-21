# STRAT-01 GigaChat 3.1 Rung-2B cross-input Q4_K diagnostic result

**Date:** 2026-09-21

**Outcome:** `REFERENCE_INPUT_PASSES_Q4K_PATH`

**Cell:** `STRAT-01-ENGINE-RUNG2B-CROSS-INPUT-DIAGNOSTIC`

**Implementation revision:** `c3e491bc502f813c116416b447e55f2da3cd193c`

**Raw evidence:** [`strat01_gigachat_engine_rung2b_cross_input_20260921`](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2b_cross_input_20260921/)

**Claim labels.** Identities, replay equality, numerical metrics, Q8_K census, controls, and execution count are **MEASURED**. The pinned llama.cpp RMSNorm accumulator type and the project helper's accumulator type are **SOURCE-DERIVED**. Selecting RMSNorm accumulation as the next diagnostic boundary is an **INFERENCE**, not yet a repair claim.

## Result

The frozen [cross-input protocol](STRAT_01_GIGACHAT31_ENGINE_RUNG2B_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260921.md) executes no donor graph. It feeds either the captured pinned-reference or C-engine `ffn_norm-0` into the existing C Q4_K×Q8_K path for block-0 up and gate.

With the exact reference normalized input, both primary projections pass by four orders of magnitude relative to the NRMSE gate:

| projection | NRMSE | normalized maximum | frozen gate | result |
|---|---:|---:|---:|---|
| up | `1.6233990094531767e-7` | `8.486540855395793e-8` | `0.002 / 0.01` | PASS |
| gate | `1.483582107322407e-7` | `7.548502449579627e-8` | `0.002 / 0.01` | PASS |

The captured C normalized input differs from the reference at NRMSE `0.0007914143404266998` and normalized maximum `0.0006736163184616577`. Replaying it produces byte-identical accepted C up/gate payloads and reproduces the frozen Rung-2B failures within `1e-12`: up NRMSE `0.0030612619849613985`, gate `0.0026914051074363363`.

The C-path output delta caused only by changing normalized input is up NRMSE `0.003061262264819756` and gate `0.002691404004961108`, essentially the entire measured disagreement. The two serialized Q8_K input representations differ in **46 of 48 blocks** and **369 of 14,016 bytes**. This census localizes a discontinuous representation change but does not prove that Q8_K is the exclusive amplification mechanism.

## Controls and provenance

- accepted C up and gate outputs replay byte-for-byte;
- frozen Rung-2B metrics reproduce within `1e-12`;
- a one-byte input mutation is refused;
- swapped targets reject strongly: NRMSE `1.3039597477599665` and `1.5611144063021254`;
- source, matrix descriptor, artifact, input, output, and Q8 payload hashes validate;
- `donor_graph_executions=0`;
- apparatus preflight was separately `APPARATUS_READY_NO_DONOR_EXECUTION`;
- final runtime was 38.512 s, an operational record only, not throughput.

## Consequence and no-duplication boundary

The registered up/gate Q4_K matrix path, layout, scale decode, and accumulation are sufficient when started from the exact reference input. Do not rewrite that kernel, rerun the cross-input cell, relax Rung-2B gates, or advance to Rung 2C.

The accepted upstream perturbation is sufficient to break composed FFN parity. The next admissible boundary is therefore the normalized input, not the projection matrix. Pinned llama.cpp commit `5b335f4` accumulates RMSNorm squared values in `ggml_float` (double), then converts the mean to float; `strat01_r2a_rmsnorm` currently accumulates in float. Freeze a payload-only diagnostic that compares those two accumulation semantics on captured `ffn_inp-0` and the accepted F32 FFN norm weight. It must execute no donor graph and must precede any production edit.

No quality, Rung-2B repair, Rung 2C, RAM, or rate claim follows. No `SPEED_LEDGER.md` update is due.

