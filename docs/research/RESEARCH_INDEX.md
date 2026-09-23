# Research control index

**Purpose.** This is the short control-plane for the repository research record.
Read it before proposing or running work.  It is a navigation and
no-duplication index, not a replacement for a canonical result, raw artifact,
or preregistered brief.

**Snapshot assembled:** 21 September 2026.  A status is one of:

- **MEASURED** — the stated estimand was adjudicated; scope is binding.
- **SOURCE-DERIVED** — metadata, arithmetic, or source inspection only.
- **PROPOSED** — protocol or implementation exists, but has no result.
- **VOID** — apparatus/control failed before the estimand; it is not evidence.

## 1. Reading order and precedence

1. The newest canonical probe/result and any newer audit for the *same cell*.
2. The newest dated strategic status: [STATUS_20260921](STRATEGIC_10B_20260916/STATUS_20260921.md).
3. The chronological donor record: [donor-adaptation INDEX](donor_adaptation/INDEX.md) and
   [SPEED_LEDGER](donor_adaptation/SPEED_LEDGER.md).
4. The no-duplication map: [AXIS_COVERAGE_AND_NO_DUPLICATION_MAP](donor_adaptation/audits/AXIS_COVERAGE_AND_NO_DUPLICATION_MAP.md).
   For the earlier broad inventory, use the dated [research catalog](../../.planning/codebase/RESEARCH_CATALOG.md); its 16 September snapshot is not the current STRAT-01 pilot status.
5. The strategic design snapshot: [ROADMAP](STRATEGIC_10B_20260916/ROADMAP.md), with its
   source register [EVIDENCE](STRATEGIC_10B_20260916/EVIDENCE.md).
6. Briefs, proposals, README/HANDOFF material, and old plans only describe intent unless a
   newer result promotes them.

When two documents disagree, record the discrepancy and follow the newest scoped canonical
result; do not average claims or silently choose the attractive one.

## 2. Goal and non-negotiable gates

The target is one identifiable pretrained approximately-10B artifact, executed by the project C
engine (extensions allowed), with useful quality and at least 50 accepted end-to-end tok/s
(100 tok/s is a stretch target).  The same artifact must be used for quality and rate.

| Gate | Required reading |
|---|---|
| Quality | Paired document-level BPB against its exact teacher; preferred delta <= +0.01, mandatory upper one-sided CI95 <= +0.02.  Then preregistered task/rollout/rank checks; BPB alone is not promotion. |
| Engine fidelity | Exact artifact identity, tokenizer and corpus; explicit format/layout; parity/semantic checks through the engine path actually timed. |
| Rate | Accepted emitted tokens, batch 1, declared context and environment; lower CI95 >= 50, not merely a central value.  TTFT/prefill, RAM, bytes/text and occupancy are separate records. |
| Scale discipline | Synthetic shapes, source-derived traffic arithmetic, and external/literature claims cannot substitute for a pretrained quality-plus-rate result. |

The complete contract and rationale are in [ROADMAP §2](STRATEGIC_10B_20260916/ROADMAP.md).

## 3. Current program state, by stream

| Stream / highest relevant IDs | Status | What is established | What it does **not** establish | Canonical record |
|---|---|---|---|---|
| **Synthetic target-scale engine** — E36/E39/E40, E63 Part A/B, §69 | **MEASURED-SCOPED** | E63 Part B clean mixed A10B shape: int8-carved median **49.37 tok/s**, CI **[49.18, 50.49]**; its own lower-bound gate is `DESK-MODEL-HELD`. | It is synthetic/noise, only 11.4% of charged weights are int8, it is not a full-int8 donor, not pretrained quality, and not the final >=50 gate. | [SPEED_LEDGER §69](donor_adaptation/SPEED_LEDGER.md), raw `benchmarks/donor_adaptation/engine/results/e63_part_b*.json`, [E63 probe](donor_adaptation/probes/E63_THE_TEN_BILLION_CELL_AT_ONE_BYTE.md) |
| **Post-hoc structure/format** — E64–E68 | **CLOSED or constrained by cell** | E64 run 2: post-hoc int8 carve is dearer at every registered rung. E65: fast synthetic rank fraction has a large real-donor quality cost. E66: the specified one-byte 7B conversion is near score-neutral but misses strict rank. E67 output-aware local selector has no material local signal; E68 shared rank-64 residual is only partial, layer-concentrated local signal. | No universal claim about every selector, rank basis, jointly trained residual, donor, or 10B model. E67/E68 are local-error diagnostics, not BPB/rate. | [E64](donor_adaptation/probes/E64_CARVE_ON_INT8.md), [E65](donor_adaptation/probes/E65_RANK_FRACTION_COST.md), [E66](donor_adaptation/probes/E66_ONE_BYTE_AT_SEVEN_BILLION.md), [E67](donor_adaptation/probes/E67_OUTPUT_AWARE_SELECTION.md), [E68](donor_adaptation/probes/E68_SHARED_RESIDUAL.md) |
| **Trained donor repairs** — H1 S3, H2I v4, H4/H4D/H5 | **MIXED; terminal at stated gates** | H1 S3 closes ternary 8-layer carve trainability. H2I one-byte training improves score/routing but fails its combined rank gate: `SCORE-ONLY`. H4 recovers some score/rank but generation remains at floor; H5 frozen H4+H2I composition is adverse. | No target-scale, rate, export, or fresh-joint-training conclusion. H2I’s export seam is not an implementation queue because its composed adjudication failed. | [H1](donor_adaptation/probes/H1_THE_CARVE_TRAINED.md), [H2I](donor_adaptation/probes/H2I_ONE_BYTE_CARVE_BASELINE.md), [H4](donor_adaptation/probes/H4_AGGRESSIVE_RANK_STEP_ZERO.md), [H5](donor_adaptation/probes/H5_CROSS_COMPOSITION_RESULT.md), [H2I export-gap audit](donor_adaptation/audits/H2I_ENGINE_EXPORT_GAP_AUDIT.md) |
| **STRAT-02 StdMoE W4 / router** | **CLOSED for the tested arms** | Teacher baseline is **0.604406515337738 BPB**. W4 fails the +0.02 upper-CI gate. The sole preregistered router-F32 follow-on improves W4 but still fails: upper CI **+0.0203953142210874**. W2 expert scout is noncompetitive. | No generic impossibility of W4, router F32, another donor, or training into a new geometry; no C/rate/T4 result. The 96-doc heldout was already used for this follow-on, hence is not final-virgin selection data. | [STRAT-02F result](donor_adaptation/probes/STRAT_02F_W4_ROUTER_F32_HELDOUT_RESULT.md); prior apparatus/results under `donor_adaptation/probes/STRAT_02*` and raw `benchmarks/donor_adaptation/density/results/strat02*` |
| **STRAT-03 shared residual / router** | **CLOSED diagnostic** | Byte-matched nonlinear shared SwiGLU plus x-only router fails all three local gates on five pinned FFN layers; apparatus is valid. | Not BPB, generation, engine rate, 10B, jointly trained nonlinear sharing, or another routing target/partition. | [STRAT-03 result](donor_adaptation/probes/STRAT_03_EXECUTABLE_SHARED_RESIDUAL_RESULT.md), raw `benchmarks/donor_adaptation/engine/results/strat03_executable_shared_residual*.json` |
| **STRAT-01 GigaChat 3.1 base** | **QUALITY + PIQA + ROLLOUT; C BLOCK 0 PRODUCTION PASS** | The original Rung-2B FAIL is preserved, diagnosed, repaired, and superseded for the current production implementation by `PASS_ENGINE_BLOCK0_PRODUCTION_INTEGRATION`: all 12 block-0 comparisons pass, continuity is exact, and causal controls are live. | Do not repeat Rung 2A/2B or their repair chain. Rung 2C must be separately frozen for one block-1 routed/shared MoE layer. Tokenizer, later layers, full logits/generation/C quality, RAM, rate, and HumanEval remain open. | [Production result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_BLOCK0_PRODUCTION_INTEGRATION_RESULT_20260922.md) |
| **STRAT-01 GigaChat 3.1 MTP** | **REFERENCE PILOT; quality/rate open** | Pinned Q4 target plus separately converted BF16 MTP GGUF loads and invokes `draft-mtp`; native paired greedy observation is 12.78 tok/s baseline vs 11.01 MTP, 25/38 proposals accepted. | No BPB, task, population acceptance, clean-box rate, >=50 result, greedy equivalence, or `phase60/engine.c` port. The divergence at token 26 is unresolved (batch form is sufficient in one control; accumulated-state/KV issue remains possible). | [MTP pilot audit](donor_adaptation/audits/STRAT_01_GIGACHAT31_MTP_PILOT_20260918.md), [strategic status addendum](STRATEGIC_10B_20260916/STATUS_20260918.md), raw `benchmarks/donor_adaptation/density/results/strat01_gigachat*` |
| **Other donor screens** | **SOURCE-DERIVED** | GigaChat, Nous, Granite/LFM and Susono screens record revision/metadata/licence/traffic considerations. | No local quality/rate unless a cited audit says otherwise; metadata does not qualify a checkpoint for the final claim. | `donor_adaptation/audits/STRAT_01_*METADATA_SCREEN*`, `STRAT_01_SPARSE_HYBRID_METADATA_SCREEN.md` |
| **Native SSM / Phase-64** | **PARKED** | Small trained-native machinery and selected properties exist; S0 has a scalar BPB observation. | No target-scale artifact or final property adjudication for S0; do not spend donor quota by treating it as an automatic successor. | [axis map](donor_adaptation/audits/AXIS_COVERAGE_AND_NO_DUPLICATION_MAP.md), `docs/PHASE64_*` |

## 4. Chronology that matters for avoiding repeats

| ID / period | Type | Disposition and reuse rule |
|---|---|---|
| E63 Part A | **MEASURED** | Loader/container/parity mechanics fire. Reuse the frozen path; do not substitute its checks for the Part-B rate result. |
| E63 Part B / §69 | **MEASURED** | The old “G-E63d owed/VOID” wording is superseded by §69. Do **not** rerun its fired A10B rate gate, loosen occupancy, or call ancillary E61c/3B work complete. |
| E64 run 1 | **VOID / MALFORMED** | Its selected cells mixed attention-kernel conditions. Preserve it as the reason `CONFIG` is mandatory; never quote its int8 values. |
| E64 run 2 | **MEASURED** | Closed only for post-hoc carve on that donor/format/slice. Training in format is a changed coordinate, not a rerun. |
| E65 | **MEASURED-SCOPED** | Do not extrapolate its non-monotone rank-cost ladder from D=1536 to target width. |
| E66 | **MEASURED-SCOPED** | Do not rerun the exact 7B one-byte quality/rank cell; changing donor, quantizer, trained format, or estimand is required. |
| E67/E68 | **MEASURED-SCOPED** | Do not repeat mass/output-greedy or the exact shared-linear geometry. A successor must name a different residual/selection mechanism and its byte/quality/engine gates. |
| H1 S3 / H2I / H4/H4D/H5 | **MEASURED, terminal for stated gates** | Do not export H2I, restart its frozen training, or combine frozen H4/H2I as a shortcut. Fresh joint training is not tested by H5 but needs a new brief. |
| STRAT-02 attempt voids | **VOID** | Format/path-alias/resource voids are apparatus history only. The later valid W4 and 02F results, not the voids, decide the tested arms. |
| STRAT-02F | **MEASURED FAIL_BPB** | One selected W4+router-F32 arm is closed. Do not mine the same heldout for a sequence of new arms. |
| STRAT-03 | **MEASURED FAIL** | The exact five-layer shared/router diagnostic is closed; it is not a global theorem against block/Jacobi, low-rank-plus-block, or jointly trained approaches. |
| STRAT-01 MTP pilot | **MEASURED pilot + open gates** | Do not interpret any current MTP observation as quality or final rate. Diagnose equivalence and score the effective artifact before new speed sweeps. |
| STRAT-01 base PIQA | **MEASURED `PASS_PIQA`** | Do not repeat the 1,838-item BF16/Q4 cell or mine PIQA for format selection. The protocol authorizes the frozen 96-document rollout next. |
| STRAT-01 base document rollout | **MEASURED `PASS_DOCUMENT_ROLLOUT`** | Do not repeat the 96-document BF16/Q4 greedy cell or tune its degeneration threshold. Two Q4-only `loop8x3` documents pass the frozen failure threshold of at least three; the next coordinate is C-engine compatibility/parity. |
| STRAT-01 engine rung 0 | **MEASURED `PASS_ENGINE_RUNG0`** | Do not repeat the accepted-file header census or identity hash. All 414 descriptors agree with the pinned reader; rung 1 has since closed the registered numerical cells. |
| STRAT-01 engine rung 1 | **MEASURED `PASS_ENGINE_RUNG1`** | Never repeat the three codec/scalar stored-row cells for `blk.0.attn_q.weight` (Q4_K), `blk.0.attn_k_b.weight` (Q5_0), or `blk.0.ffn_down.weight` (Q6_K) on the same artifact, inputs, and estimand. The changed coordinate is now frozen as [rung 2A](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2A_PROTOCOL_20260921.md): block-0 MLA attention with paired eight-token prefill and 7+1 F16-cache execution. Do not widen it to FFN or MoE before adjudication. |
| STRAT-01 engine rung 2A | **MEASURED `FAIL_ENGINE_RUNG2A`; CLOSED** | Preserve the two early apparatus voids and the third raw void; do not reinterpret them. The third run's two producers completed and emitted immutable payloads. Commit `89b15d5` corrected only lowercase GGML type metadata, hash-bound the raw records, and adjudicated offline with zero donor executions. Six of 24 tensor comparisons pass, 18 fail; all three cache checkpoints fail. Both C and reference continuity checks pass with zero error, localizing the disagreement before cache sequencing. Never repeat or retune this acceptance cell. Freeze a distinct Q/KV full-matrix projection diagnostic using captured evidence where possible; do not advance to 2B yet. Canonical [result and full provenance](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2A_RESULT_20260921.md). |

## 5. Pending work is not a result

| Item | State at this snapshot | Required prerequisite / decision |
|---|---|---|
| GigaChat producer BF16 GGUF | **ACQUIRED/VERIFIED INPUT, not a score** | Local size 21,356,281,984 B and SHA-256 `e7a6409be0ac197babf21c48cfc8a96486d035c1e7a784feadd817b7b883c08e` match the pinned producer blob. This does not prove full parity with the source safetensors checkpoint. |
| GGUF document scorer | **PASS_INTERNAL_GGUF; FINAL GATES OPEN** | On 96 paired documents (32/category; 202,247 tokens; 746,161 bytes), producer BF16 is 0.5634397268 BPB and Q4 is 0.5755673276: Δ +0.0121276008, upper one-sided CI95 +0.0130809692 under the preregistered +0.02 internal limit. All category upper bounds are also below +0.02. Scores, manifests, and `heldout_adjudication.json` are under `benchmarks/donor_adaptation/density/results/strat01_gguf_quality_v1/`. This corpus is not fresh and the reference is the producer BF16 GGUF, so this is not the final quality gate. Runtime `llama.cpp` text tokenizer mismatched source IDs on all documents, leaving C-engine tokenizer parity open. |
| GigaChat GGUF quality protocol | **96-DOC INTERNAL PASS; SUPERSEDED BY SOURCE/FRESH/TASK PASSES FOR PROMOTION** | [Protocol and result](donor_adaptation/probes/STRAT_01_GIGACHAT31_GGUF_QUALITY_PROTOCOL_20260918.md) records apparatus controls and the internal GGUF pass. It did not itself prove source identity or final quality; those later gates are now recorded separately below. The execution path must still run the accepted artifact in `phase60/engine.c` at ≥50 tok/s lower CI95. |
| GigaChat source→GGUF binding | **PASS_SOURCE_BINDING (BASE MODEL)** | [Protocol and result](donor_adaptation/probes/STRAT_01_GIGACHAT31_SOURCE_BINDING_PROTOCOL_20260919.md): all six source shards pass pinned hashes; a clean `llama.cpp` conversion at `5b335f4` yields the same 414 tensor inventory as producer BF16. All 21,350,179,072 tensor bytes are identical, zero mismatches; per-tensor manifest digest `c4d3e3dc…fd309`. Metadata differences are limited to descriptive/header fields. This binds the base model only, not omitted MTP, C tokenizer, tasks or rate. |
| GigaChat fresh BPB | **PASS_FRESH_BPB** | [Protocol and result](donor_adaptation/probes/STRAT_01_GIGACHAT31_FRESH_BPB_PROTOCOL_20260919.md): 96 new documents, 109,989 tokens, 393,115 bytes, zero prior ID/content overlap. BF16 0.6226312179 BPB; Q4 0.6349546541; Δ +0.0123234362, upper CI95 +0.0135548442 ≤+0.02. All category bounds pass; prose is closest at +0.0182247781. This BPB result did not itself establish task/rollout; those subsequent gates now pass. C tokenizer/operator parity and engine/rate remain open. |
| GigaChat task/rollout | **`PASS_PIQA`; `PASS_DOCUMENT_ROLLOUT`; HumanEval pending** | [PIQA result](donor_adaptation/probes/STRAT_01_GIGACHAT31_PIQA_RESULT_20260920.md): BF16 1466/1838, Q4 1456/1838, required 1437. The [frozen 96-document rollout](donor_adaptation/probes/STRAT_01_GIGACHAT31_DOCUMENT_ROLLOUT_RESULT_20260921.md) passes with 2 Q4-only `loop8x3` documents against the ≥3 failure threshold; BF16 has one degeneration in `technical_general`. Both arms generated 256 tokens/document with no EOS, `run32`, or `empty`. HumanEval remains `PENDING_SANDBOX`. |
| GigaChat Rung-2A projection diagnostic | **MEASURED `ATTRIBUTED_Q8K_ACTIVATION_QUANTIZATION`; CLOSED** | [Frozen offline result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2A_PROJECTION_DIAGNOSTIC_RESULT_20260921.md): pinned GGML Q4_K×Q8_K reproduces both captured reference projections bit-for-bit (zero error), while dequant-F32 reproduces the C outputs and retains the old Q/KV NRMSE failures on the reference input. F64 accumulation does not close them. Zero donor executions. Implement the exact operator semantics and confirm offline before a new changed-coordinate attention gate; never repeat the old Rung-2A cell. |
| GigaChat engine Q4_K×Q8_K repair | **MEASURED `PASS_ENGINE_Q4K_Q8K_REPAIR`; CLOSED** | [Canonical result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REPAIR_RESULT_20260921.md): standalone C Q8_K bytes are exact on the model-free oracle population; both immutable Q/KV projections pass by wide margins (NRMSE `6.45e-8` and `7.32e-8`). Twenty oracle/negative/legacy controls and the 73,024-check kernel self-test pass; zero donor executions. Preserve the pre-execution directory-binding VOID and never repeat this cell. A separately frozen changed-coordinate block-0 attention confirmation is next; no speed claim. |
| GigaChat changed-coordinate block-0 attention confirmation | **MEASURED `PASS_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION`; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2A_Q4_REPAIR_CONFIRMATION_RESULT_20260921.md): both schedules pass tight `attn_norm`/Q/KV gates and exact C continuity; all tensors through `Vcur-0` plus all caches pass old gates. First remaining failure is `kqv_out-0` (NRMSE `0.01199015`) in both arms, followed by `ffn_inp-0`. One C execution, zero reference reruns. Do not repeat; next separate attention reconstruction from special-layout V-B semantics. |
| GigaChat `kqv_out-0` attribution | **MEASURED `PARTIAL_VB_Q8K_ATTRIBUTION`; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_KQV_OUT_DIAGNOSTIC_RESULT_20260921.md): project and pinned Q8_K agree tightly and improve target NRMSE from `0.01199014` to `0.00125348`, but miss the tight target gate. Zero donor executions. Do not repeat or install the production repair yet. |
| GigaChat pre-V-B latent capture | **MEASURED `ATTRIBUTED_RESIDUAL_TO_ATTENTION_RECONSTRUCTION`; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_PRE_VB_LATENT_CAPTURE_RESULT_20260921.md): reconstructed latent fails at NRMSE `2.16e-4`; project/pinned V-B and final layout pass on true `kqv`, with pinned outputs exact. One reference prefill execution. Do not repeat. |
| GigaChat attention-stage diagnostic | **MEASURED `MIXED_ATTENTION_STAGE_RESIDUAL`; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_ATTENTION_STAGE_DIAGNOSTIC_RESULT_20260921.md): softmax passes; scalar QK and value reduction fail independently. Preserve padded-shape VOID; canonical adjudication reused its trace with zero additional donor executions. |
| GigaChat F16 converter audit and repair | **MEASURED `PASS_F16_CONVERTER_REPAIR`; CLOSED** | [Repair result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_F16_CONVERTER_REPAIR_RESULT_20260921.md): zero mismatches on 217,620 frozen values and 16,777,216 exhaustive exponent-−25 cases. The preceding [audit](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_F16_CONVERTER_AUDIT_RESULT_20260921.md) isolated the one-line defect. Do not repeat. |
| GigaChat F16 vector-dot diagnostic | **MEASURED `F16_CONVERSION_ONLY_SUFFICIENT`; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_F16_VEC_DOT_DIAGNOSTIC_RESULT_20260921.md): scalar F16×F16 passes both tight gates; pinned vector outputs are exact; causal controls pass. Preserve both prior launches as VOID and do not repeat. |
| GigaChat attention + V-B production repair | **MEASURED `PASS_ENGINE_ATTENTION_VB_REPAIR`; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_ATTENTION_VB_REPAIR_RESULT_20260921.md): 24/24 tensors and 3/3 caches pass, both schedules have no remaining failure, and C continuity is exact. One C artifact execution, zero reference reruns. Do not repeat. |
| GigaChat engine Rung 2B dense SwiGLU | **MEASURED FAIL; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2B_RESULT_20260921.md): `ffn_norm-0` passes; Q4_K up/gate are the first failures in both schedules. One non-VOID C execution, exact continuity, all negative controls live. Do not rerun or tune gates; use captured normalized inputs for the separately scoped diagnostic. |
| GigaChat Rung-2B cross-input Q4_K diagnostic | **MEASURED `REFERENCE_INPUT_PASSES_Q4K_PATH`; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2B_CROSS_INPUT_DIAGNOSTIC_RESULT_20260921.md): exact reference input yields up/gate NRMSE `1.62e-7`/`1.48e-7`; C input replay is byte-exact and reproduces the FAIL. 46/48 Q8 blocks change. Do not alter the Q4_K matrix path; diagnose upstream RMSNorm semantics next. |
| GigaChat Rung-2B RMSNorm accumulator diagnostic | **MEASURED `DOUBLE_RMSNORM_INSUFFICIENT_FOR_PROJECTION_GATES`; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2B_RMSNORM_ACCUMULATOR_DIAGNOSTIC_RESULT_20260921.md): double accumulation exactly reproduces reference normalization but barely changes the C discrepancy; up/gate still fail at NRMSE `0.00306126`/`0.00269141`. Do not patch only FFN RMSNorm or repeat. Localize the same pinned semantics at the upstream attention RMSNorm sites next. |
| GigaChat upstream RMSNorm propagation diagnostic | **MEASURED `UPSTREAM_DOUBLE_RMSNORM_INSUFFICIENT_FOR_PROJECTION_GATES`; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_UPSTREAM_RMSNORM_PROPAGATION_DIAGNOSTIC_RESULT_20260921.md): attention norm becomes exact and terminal input improves to NRMSE `4.97e-4`, but up/gate still fail at `0.0026265`/`0.0023148`. First material residual is Q5_0 K-B absorption after near-exact `q-0`; diagnose Q5_0×Q8_0, do not repeat RMSNorm. |
| GigaChat K-B Q5_0×Q8_0 diagnostic | **MEASURED `Q5_0_Q8_0_SUFFICIENT_FOR_KB_BOUNDARY`; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_KB_Q5_0_Q8_0_DIAGNOSTIC_RESULT_20260921.md): pinned semantics match the immutable output at NRMSE `4.80e-8`; current dequant-F32 remains at `3.1511e-4`. Exact-reference, upstream-double, and accepted-float inputs serialize to identical Q8_0 bytes. Do not repeat. |
| GigaChat combined RMSNorm + K-B propagation | **MEASURED `COMBINED_RMS_Q5Q8_CLOSES_PROJECTION_GATES`; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_COMBINED_RMS_Q5Q8_PROPAGATION_RESULT_20260922.md): 24/24 Rung-2A tensors, 3/3 caches, 12/12 continuity checks, and 6/6 FFN comparisons pass. Up/gate NRMSE fall to `4.0201e-4`/`3.4697e-4`; terminal `ffn_inp-0` is `1.6236e-5`. One captured donor graph, zero new offline donor/reference graphs. Preserve both adjudicator VOID records; do not repeat. Production integration confirmation is next, not Rung 2C. |
| GigaChat block-0 production integration | **MEASURED `PASS_ENGINE_BLOCK0_PRODUCTION_INTEGRATION`; CLOSED** | [Result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_BLOCK0_PRODUCTION_INTEGRATION_RESULT_20260922.md): exact combined norm/up/gate hashes; 12/12 checkpoint comparisons pass through SwiGLU, Q6_K down, and terminal `l_out-0`; 12/12 continuity checks are zero-error and all four causal controls reject. One production invocation, zero reference reruns. Do not repeat. |
| GigaChat Rung 2C block-1 MoE | **MEASURED `FAIL_ENGINE_RUNG2C`; BOUNDARY ATTRIBUTED** | [Rung result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2C_RESULT_20260923.md): both arms first fail at `kqv_out-1`; routing passes. The zero-donor [cross-input result](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2C_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md) is `QUERY_AND_KV_RESIDUALS_INDEPENDENTLY_SUFFICIENT`: exact reference Q/KV passes attention+V-B, but either C query or C KV fails. Do not repeat producers, change attention/V-B, or tune gates. The frozen [layer-1-start propagation apparatus](donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2C_LAYER1_START_CROSS_INPUT_DIAGNOSTIC_APPARATUS_RESULT_20260923.md) is qualified with zero donor executions; one invocation is open. |
| GigaChat MTP rate continuation | **OPEN, but deprioritized** | Resolve batch-shape/history/KV divergence and demonstrate quality on the effective path. A further MTP BF16 sweep alone cannot close the measured 12.78-to-50 gap. |
| E61c / planned 3B ancillary cells | **UNMEASURED** | E63 runner aborted after a configuration assertion; these are separate cells, not missing repetitions of E63d. A new scoped repair must be briefed. |
| Final target artifact | **QUALITY-CANDIDATE SELECTED; BLOCK-0 PASS; BLOCK-1 FAIL; FULL MODEL/RATE OPEN** | The accepted artifact passes dense block 0 but the first complete block-1 cell fails from `kqv_out-1` onward. The router surface itself passes. HumanEval, tokenizer, remaining layers/operators, full logits/generation/C quality, RAM, and accepted-token rate remain open. |

## 6. Raw-artifact navigation

- Donor quality workers, manifests and scores: `benchmarks/donor_adaptation/density/results/`.
- Engine experiment JSON and audit sidecars: `benchmarks/donor_adaptation/engine/results/`.
- Canonical probes and frozen briefs: `docs/research/donor_adaptation/probes/` and
  `docs/research/donor_adaptation/briefs/`.
- Donor screens, provenance and no-duplication audits: `docs/research/donor_adaptation/audits/`.
- Native SSM run records: `docs/PHASE64_*`, `docs/PHASE64_RUNG1_RUN_RECORD.md` and
  `benchmarks/phase60/` / `benchmarks/phase64/`.

Raw files are evidence only when their canonical result identifies the artifact, control state and
adjudication.  A downloaded weight file, an executable, a log, or a provisional JSON alone is not
a completed experiment.

## 7. Mandatory pre-brief control card

Before assigning a new experiment ID, include all eight answers:

1. Nearest prior cell and canonical artifact link.
2. Exact changed coordinate: donor/family, scale/shape, format, training, domain, engine path,
   or estimand.
3. One falsifiable sentence saying why that prior result does not answer this cell.
4. Frozen artifact/corpus/tokenizer and paired controls, including engine `CONFIG`.
5. Separate gates for quality, rank/rollout, engine parity, and rate.
6. Labels for every claim: measured, source-derived, proposed, or void.
7. Void handling and a stop rule; repair narrowly without erasing the void.
8. Raw-result location plus the canonical document that adjudicates it.

If the changed coordinate cannot be named, the work is a duplicate and must not run.

## 8. Staleness and contradiction warnings

- [ROADMAP](STRATEGIC_10B_20260916/ROADMAP.md) and [EVIDENCE](STRATEGIC_10B_20260916/EVIDENCE.md)
  are 16 September design/source snapshots, not a live experiment ledger.
- The top of the donor [INDEX](donor_adaptation/INDEX.md) contains earlier GigaChat extraction and
  “not yet runnable” language. The newer [MTP pilot audit](donor_adaptation/audits/STRAT_01_GIGACHAT31_MTP_PILOT_20260918.md)
  and [STATUS addendum](STRATEGIC_10B_20260916/STATUS_20260918.md) supersede that scope.
- Earlier SPEED_LEDGER sections call `G-E63d` owed or void. [§69](donor_adaptation/SPEED_LEDGER.md)
  records the later admissible measurement; only its ancillary E61c/3B cells remain unmeasured.
- The MTP audit contains more than one observation: generic-build 23/39 and native paired 25/38
  proposals are different runs, not competing estimates. Neither is population acceptance.
- The pinned llama.cpp maps `tokenizer.ggml.pre=gigachat` to a GPT-2 regex unlike the source
  tokenizer. Its native text-to-ID path fails parity on the first frozen document; previous CLI
  generation measures remain pilot observations, not proof of source-tokenizer fidelity.
- STRAT-02F's mathematical point delta is below +0.02, but its preregistered *upper CI* fails;
  the verdict is therefore `FAIL_BPB`.
- E63 near-50 and E40 above-100 are format/shape-specific synthetic rates; neither contradicts
  the absence of a qualifying pretrained target artifact.
