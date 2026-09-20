# Research control index

**Purpose.** This is the short control-plane for the repository research record.
Read it before proposing or running work.  It is a navigation and
no-duplication index, not a replacement for a canonical result, raw artifact,
or preregistered brief.

**Snapshot assembled:** 20 September 2026.  A status is one of:

- **MEASURED** — the stated estimand was adjudicated; scope is binding.
- **SOURCE-DERIVED** — metadata, arithmetic, or source inspection only.
- **PROPOSED** — protocol or implementation exists, but has no result.
- **VOID** — apparatus/control failed before the estimand; it is not evidence.

## 1. Reading order and precedence

1. The newest canonical probe/result and any newer audit for the *same cell*.
2. The dated strategic status: [STATUS_20260920](STRATEGIC_10B_20260916/STATUS_20260920.md).
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
| **STRAT-01 GigaChat 3.1 base** | **QUALITY + PIQA GATES PASS; rollout/engine/rate open** | Exact source binding passes; fresh paired BPB passes with upper CI95 +0.0135548442; paired PIQA passes with BF16 1466/1838 and Q4 1456/1838 against a required 1437. | No document-rollout, HumanEval, MTP-quality, C-engine parity, clean-box rate, or >=50 tok/s conclusion. | [source binding](donor_adaptation/probes/STRAT_01_GIGACHAT31_SOURCE_BINDING_PROTOCOL_20260919.md), [fresh BPB](donor_adaptation/probes/STRAT_01_GIGACHAT31_FRESH_BPB_PROTOCOL_20260919.md), [PIQA result](donor_adaptation/probes/STRAT_01_GIGACHAT31_PIQA_RESULT_20260920.md) |
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

## 5. Pending work is not a result

| Item | State at this snapshot | Required prerequisite / decision |
|---|---|---|
| GigaChat producer BF16 GGUF | **ACQUIRED/VERIFIED INPUT, not a score** | Local size 21,356,281,984 B and SHA-256 `e7a6409be0ac197babf21c48cfc8a96486d035c1e7a784feadd817b7b883c08e` match the pinned producer blob. This does not prove full parity with the source safetensors checkpoint. |
| GGUF document scorer | **PASS_INTERNAL_GGUF; FINAL GATES OPEN** | On 96 paired documents (32/category; 202,247 tokens; 746,161 bytes), producer BF16 is 0.5634397268 BPB and Q4 is 0.5755673276: Δ +0.0121276008, upper one-sided CI95 +0.0130809692 under the preregistered +0.02 internal limit. All category upper bounds are also below +0.02. Scores, manifests, and `heldout_adjudication.json` are under `benchmarks/donor_adaptation/density/results/strat01_gguf_quality_v1/`. This corpus is not fresh and the reference is the producer BF16 GGUF, so this is not the final quality gate. Runtime `llama.cpp` text tokenizer mismatched source IDs on all documents, leaving C-engine tokenizer parity open. |
| GigaChat GGUF quality protocol | **96-DOC INTERNAL PASS; SOURCE/FRESH/TASK/ENGINE GATES OPEN** | [Protocol and result](donor_adaptation/probes/STRAT_01_GIGACHAT31_GGUF_QUALITY_PROTOCOL_20260918.md) records apparatus controls and the internal GGUF pass. It does not prove full source-checkpoint identity or authorize a final claim. Next quality work must bind the BF16 reference and freeze independent heldout/task evaluations; the separate execution path must run the accepted artifact in `phase60/engine.c` at ≥50 tok/s lower CI95. |
| GigaChat source→GGUF binding | **PASS_SOURCE_BINDING (BASE MODEL)** | [Protocol and result](donor_adaptation/probes/STRAT_01_GIGACHAT31_SOURCE_BINDING_PROTOCOL_20260919.md): all six source shards pass pinned hashes; a clean `llama.cpp` conversion at `5b335f4` yields the same 414 tensor inventory as producer BF16. All 21,350,179,072 tensor bytes are identical, zero mismatches; per-tensor manifest digest `c4d3e3dc…fd309`. Metadata differences are limited to descriptive/header fields. This binds the base model only, not omitted MTP, C tokenizer, tasks or rate. |
| GigaChat fresh BPB | **PASS_FRESH_BPB** | [Protocol and result](donor_adaptation/probes/STRAT_01_GIGACHAT31_FRESH_BPB_PROTOCOL_20260919.md): 96 new documents, 109,989 tokens, 393,115 bytes, zero prior ID/content overlap. BF16 0.6226312179 BPB; Q4 0.6349546541; Δ +0.0123234362, upper CI95 +0.0135548442 ≤+0.02. All category bounds pass; prose is closest at +0.0182247781. Task/rollout, C tokenizer and engine/rate remain open. |
| GigaChat task/rollout | **`PASS_PIQA`; ROLLOUT APPARATUS PASS; FULL ARMS NEXT** | [PIQA result](donor_adaptation/probes/STRAT_01_GIGACHAT31_PIQA_RESULT_20260920.md): BF16 1466/1838, Q4 1456/1838, required 1437; paired descriptive Δ accuracy -0.0054407, CI95 [-0.0168662, +0.0059848]. The [frozen protocol](donor_adaptation/probes/STRAT_01_GIGACHAT31_TASK_ROLLOUT_PROTOCOL_20260919.md) authorizes the 96-document rollout. Its [execution brief](donor_adaptation/briefs/BRIEF_STRAT_01_GIGACHAT31_DOCUMENT_ROLLOUT.md) records `PASS_APPARATUS`: real source-token binding and exact BF16/Q4 batch128↔64 repeatability on the paired smoke. The 96-document estimand remains unmeasured. HumanEval functional scoring remains pending a real sandbox. |
| GigaChat MTP rate continuation | **OPEN, but deprioritized** | Resolve batch-shape/history/KV divergence and demonstrate quality on the effective path. A further MTP BF16 sweep alone cannot close the measured 12.78-to-50 gap. |
| E61c / planned 3B ancillary cells | **UNMEASURED** | E63 runner aborted after a configuration assertion; these are separate cells, not missing repetitions of E63d. A new scoped repair must be briefed. |
| Final target artifact | **OPEN** | The next genuine decision is whether a reviewed, hash-pinned donor geometry has a plausible quality gate path *and* a separately declared engine/rate path. No inherited experiment launches automatically. |

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
