# STRAT-01 GigaChat 3.1 K-B Q5_0×Q8_0 diagnostic result

**Date:** 2026-09-21

**Outcome:** `Q5_0_Q8_0_SUFFICIENT_FOR_KB_BOUNDARY`

**Cell:** `STRAT-01-ENGINE-KB-Q5_0-Q8_0-DIAGNOSTIC`

**Protocol revision:** `487eef0`

**Implementation revision:** `f45b345`

**Raw evidence:** [`strat01_gigachat_engine_kb_q5q8_diag_20260921`](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_kb_q5q8_diag_20260921/)

**Adjudication SHA-256:** `fd2ba64075f02fa26ff9210ff06a80cd6c2e3729966311d991235edc699416eb`

**Run-manifest SHA-256:** `83f363f63cf4232f21a19b89bfdc7d91e5968b70e0ad5ea2c74acdb857f99c7d`

**Claim labels.** Operator metrics, activation bytes, frozen replays, controls, identities, and execution counts are **MEASURED**. The explanation that Q8_0 quantization absorbs the remaining upstream input differences is **MEASURED at this boundary** because all three serialized activation payloads are identical. Sufficiency beyond K-B remains unmeasured.

## Result

The frozen [protocol](STRAT_01_GIGACHAT31_ENGINE_KB_Q5_0_Q8_0_DIAGNOSTIC_PROTOCOL_20260921.md) compared the current dequantize-Q5_0-to-F32 path with pinned llama.cpp Q8_0 activation quantization plus Q5_0×Q8_0 dot semantics. The C implementation was independently checked against a helper built from clean llama.cpp revision `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`. No donor graph was executed.

| comparison | NRMSE | normalized maximum | frozen gate | result |
|---|---:|---:|---|---|
| C Q5_0×Q8_0 vs pinned helper, exact-reference input | `4.802399901907019e-8` | `8.816135785418076e-8` | `2e-6 / 1e-5` | PASS |
| C Q5_0×Q8_0 vs pinned helper, upstream-double input | `4.802399901907019e-8` | `8.816135785418076e-8` | `2e-6 / 1e-5` | PASS |
| Q5_0×Q8_0 exact-reference input vs immutable reference output | `4.802399901907019e-8` | `8.816135785418076e-8` | `2e-6 / 1e-5` | PASS |
| Q5_0×Q8_0 upstream-double input vs immutable reference output | `4.802399901907019e-8` | `8.816135785418076e-8` | `0.002 / 0.01` | PASS |
| current dequant-F32 exact-reference input vs immutable reference output | `0.0003151131202874722` | `0.0006382882308642687` | tight operator gate | FAIL |

The changed coordinate is sufficient at the K-B boundary. The previous residual is not a tensor-layout or Q5_0 codec error: it is the omitted Q8_0 activation quantization/runtime dot semantic.

## Why the upstream-double arm also closes

The serialized Q8_0 activations are byte-identical across all three frozen inputs:

| pair | changed blocks | changed bytes |
|---|---:|---:|
| exact reference vs upstream double | `0 / 1024` | `0 / 34,816` |
| exact reference vs accepted float | `0 / 1024` | `0 / 34,816` |
| upstream double vs accepted float | `0 / 1024` | `0 / 34,816` |

Thus the small differences among accepted-float, upstream-double, and exact-reference `q-0` are below the Q8_0 bin boundaries for every 128-element head row. Once the pinned runtime operator quantizes those activations, all three arms present the same bytes to K-B and therefore produce the same output up to the independently measured C/helper accumulation difference.

This does not make upstream RMSNorm irrelevant. Its benefits were measured at several other boundaries, and the next valid experiment must retain it while propagating the corrected K-B operator through the remainder of block 0.

## Controls and provenance

- repaired apparatus preflight: `APPARATUS_READY_NO_DONOR_EXECUTION`, `errors=[]`;
- diagnostic execution: `errors=[]`, `donor_graph_executions=0`;
- current upstream-double and accepted-float outputs replay their frozen targets byte-for-byte;
- C and pinned-helper Q8_0 payloads agree byte-for-byte for both primary inputs;
- one-byte input mutation fires: NRMSE `0.0003396019524150514`, normalized maximum `0.003234023090164913`;
- wrong head stride fires: NRMSE `0.9980692319152357`, normalized maximum `1.0408742613621036`;
- Q5_0 high-bit corruption fires: NRMSE `1.264013660135612`, normalized maximum `1.1492454974672124`;
- omitted Q8_0 quantization fires through the current-path tight-gate failure;
- artifact, tensor descriptor, inputs, sources, payload sizes, and pinned checkout identities all validate;
- no timing or rate claim was emitted.

An earlier apparatus-only attempt ended VOID because Ninja/ccache could not remove a locked dependency file while compiling the unnecessarily broad `llama` target. It executed no donor graph and consumed no cell. The repaired helper links only `ggml-cpu`, builds serially, and passed before the non-VOID run.

## Consequence and stop rule

Do not repeat this diagnostic, revisit Q5_0 stored-row decoding, alter K-B orientation, or relax gates. Do not claim full attention, FFN, Rung 2B, Rung 2C, generation, quality, RAM, or rate from this operator-only result.

The next distinct cell is one combined propagation confirmation: retain the already measured double RMSNorm semantics, replace only K-B execution with pinned Q5_0×Q8_0 semantics, and propagate through the existing block-0 attention and dense FFN boundaries in both schedules. Freeze its protocol before changing production. No `SPEED_LEDGER.md` entry is due.
