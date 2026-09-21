# STRAT-01 GigaChat 3.1 engine Rung 2B — dense block-0 SwiGLU FAIL

**Date:** 2026-09-21

**Outcome:** `FAIL_ENGINE_RUNG2B`

**Cell:** `STRAT-01-ENGINE-RUNG2B`

**Accepted execution revision:** `b4897d2ea4c0577b01c317c866d832f194e26a01`

**Raw evidence:** [first reference-side VOID](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2b_20260921/), [second reference-side VOID](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2b_repair1_20260921/), [final apparatus preflight](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2b_apparatus_repair4_20260921/), [accepted numerical run](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2b_repair2_20260921/)

**Claim labels.** Artifact identity, process outcomes, checkpoint metrics, continuity, controls, hashes, and execution counts are **MEASURED**. The first-failure boundary is **MEASURED**. The hypothesis that small accepted upstream differences cross Q8_K activation-quantization boundaries is an **INFERENCE** and requires the separately frozen-input diagnostic described below.

## Question and scope

The frozen [Rung 2B protocol](STRAT_01_GIGACHAT31_ENGINE_RUNG2B_PROTOCOL_20260921.md) asks whether `benchmarks/phase60/engine.c` preserves the accepted GigaChat 3.1 block-0 dense FFN from the already closed `ffn_inp-0` boundary through FFN RMSNorm, parallel Q4_K gate/up projections, SwiGLU, Q6_K down projection, and residual `l_out-0`. The paired schedules are `prefill8` and `cached7p1`; the protocol permits one non-VOID accepted-artifact C execution and makes no rate claim.

The artifact is 6,474,702,976 bytes with SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`. The pinned reference is llama.cpp commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`. The accepted run records engine source SHA-256 `7708e42109bf484324b507e4b00aff74b69ff5ba302cc6e199f0cc6381108987`, Rung-2B header SHA-256 `7df3858efa489b09ea553971915546e85ecb9c27a5389dd347d9967c47e96030`, and reference source SHA-256 `cc9d229a5c72ab24ab0e499acc269c4bc3cf981ca8fb5e82657aa90709856831`.

## Preserved apparatus record

The first launch is `VOID_ENGINE_RUNG2B`: the shared reference callback selector still requested Rung-2A's schedule-dependent `kq-0`, so cached 7+1 composition refused before the C donor arm ran. The second launch is also `VOID_ENGINE_RUNG2B`: after narrowing callback selection to the seven Rung-2B tensors, the shared serializer still demanded the inherited `Kcur-0` cache witness. It likewise failed before the C donor arm ran. These are apparatus failures and provide no numerical evidence.

The repaired reference records cache semantics as inherited from the already accepted Rung 2A and does not regenerate its cache witness. The final apparatus-only preflight is `APPARATUS_READY_NO_DONOR_EXECUTION` with empty errors. C Rung-2A and Rung-2B self-tests, the 73,024-check legacy self-test, Python tests, reference build, and reference self-test all pass. The intervening `apparatus_repair3` record is a test-assertion VOID caused only by escaped source quoting; it executed no donor and is retained as provenance.

Consequently, the accepted run contains exactly one non-VOID C artifact execution. Both reference and C producers return zero, both trace trees validate, and external adjudication runs with empty apparatus errors.

## Numerical result

Both schedules produce identical metrics and the same first-failure boundary:

| checkpoint | NRMSE | normalized maximum | gate | result |
|---|---:|---:|---:|---|
| `ffn_norm-0` | 0.0007914143404266998 | 0.0006736163184616577 | 0.002 / 0.01 | PASS |
| `ffn_up-0` | 0.0030612619849613985 | 0.0017181935481242877 | 0.002 / 0.01 | FAIL |
| `ffn_gate-0` | 0.0026914051074363363 | 0.001061259056391299 | 0.002 / 0.01 | FAIL |
| `ffn_swiglu-0` | 0.0028168146094835135 | 0.0012920635186312326 | 0.002 / 0.01 | FAIL |
| `ffn_out-0` | 0.0065417572665433 | 0.00321304984415826 | 0.002 / 0.01 | FAIL |
| `l_out-0` | 0.0037069931330366616 | 0.0013351534091505797 | 0.001 / 0.005 | FAIL |

Only the NRMSE limit fails at each rejected checkpoint; normalized maxima remain inside their limits. Of 12 arm/checkpoint comparisons, 2 pass and 10 fail. Prefill-versus-7+1 `l_out-0` continuity is exact inside both implementations: NRMSE and normalized maximum are zero for the C engine and the pinned reference. This excludes the tested schedule/cache split as the cause.

All four planted semantic controls reject strongly: gate/up swap NRMSE `0.4032666357591153`, omitted SiLU `1.0595124882875306`, mutated Q6_K down input `1.0`, and omitted final residual `0.8255974804777428`. Thus the accepted result is not explained by those coarse implementation mistakes.

## Interpretation and no-duplication boundary

The measured boundary is after the passing FFN normalization and at both parallel Q4_K projections. Rung 2A previously accepted the same engine's Q4_K×Q8_K operator on its registered attention coordinates, so this result does not invalidate that cell. It establishes that composed block-0 FFN fidelity from the accepted upstream state is outside the stricter Rung-2B NRMSE gate.

Do not rerun Rung 2B, relax its thresholds, regenerate Rung 2A, or advance to Rung 2C. A new diagnostic may reuse the immutable captured `ffn_norm-0` payloads and the same artifact to cross-feed reference and C normalized inputs into the existing Q4_K projection path. It must distinguish:

1. error already present in each implementation's normalized input;
2. discontinuous amplification caused by independent Q8_K activation quantization;
3. a matrix-specific Q4_K layout, scale, or accumulation error in the 1536→8960 FFN projections.

That diagnostic is a changed estimand, must execute no accepted full donor graph, and must be separately preregistered. Rung 2C remains blocked until the dense boundary is repaired and confirmed under a new production protocol. No `SPEED_LEDGER.md` update is due.

