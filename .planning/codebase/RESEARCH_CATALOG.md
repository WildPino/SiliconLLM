# SiliconLLM — research catalog and no-duplication handoff

**Snapshot:** 2026-09-16. **Latest material:** E63's first admissible clean A10B sweep closed `G-E63d` as `DESK-MODEL-HELD` at 49.37 tok/s and measured `G-E63e` at 0.9952; the runner then stopped at E61c's fp32/ternary CONFIG mismatch, so the ancillary Part B sweeps are incomplete. H1 session 3 v2 is COMPLETE/PASS; H2I Phase B v4 is terminal `SCORE-ONLY` (score passes, strict rank does not). The user's remembered endpoint was E64; the tree contained E65, completed E66 and H2I.

## Read this first

1. Search by mechanism/shape/format first: `docs/research/donor_adaptation/audits/AXIS_COVERAGE_AND_NO_DUPLICATION_MAP.md`.
2. Canonical donor chronology: `docs/research/donor_adaptation/INDEX.md`.
3. Frozen target-donor compatibility/upper-bound disposition: `docs/research/donor_adaptation/audits/TARGET_DONOR_DECISION.md` (not a benchmark or launch authorization).
4. Speed chronology/arithmetic: `docs/research/donor_adaptation/SPEED_LEDGER.md`.
5. Current owner actions: `docs/COMMUNICATION.md`.
6. Then open the cited probe, brief/addendum, runner and result JSON.

The axis map also separates four lineages that must not be merged: the closed frozen-substrate work,
the parked trained CPU-native/Phase-64 ladder, active pretrained-donor adaptation, and prior-art/design
documents that contain no local measurement.

## Current state in one page

| question | current answer | authority |
|---|---|---|
| Can the CPU engine run a 10B shape above 50/100 tok/s? | **Yes as a shape:** E40 `R128` measured ~113–130 tok/s with live FFN, but weights were synthetic/noise. | canonical E40 probe/ledger |
| Can a pretrained donor be post-hoc carved to the required 1.17% FFN and retain quality? | **No for this donor:** E38's perfect per-token oracle is 4.131817 BPB, above 4.069819 chance. | canonical E38 |
| Is the 1B/weight rung faithful? | **Yes at 0.5/1.5/3/7B, without a scale law:** at 7B A2 is only +0.000378 BPB from fp32; E62's non-monotonicity still forbids extrapolation. | E60/E62/E66 |
| Does a 10B carved-int8 artifact exist and compute correctly? | **Yes:** E63 Part A, exact arithmetic/parity. The later clean A10B Part B sweep measured its synthetic-noise 10B shape at **49.37 tok/s**, 95% bootstrap CI **[49.18, 50.49]**, with 45/45 preflight and 10/10 timing cells below the frozen 4.39% occupancy bar. `G-E63d` is `DESK-MODEL-HELD`; `G-E63e` ratio is 0.9952 [0.9557, 1.0177]. This is not a trained-model or ≥50-tok/s success. The runner stopped before ancillary E61c/3B results. | canonical E63 §16 / `e63_part_b.json` |
| What does carve cost on int8? | **E64 run 1 void. Audited run 2 says dearer by +1.13–1.43 BPB across the ladder; at k=3 the int8 carve itself costs +1.912 BPB and lands above chance.** | canonical E64 probe §8 / ledger §66 |
| Is the carve trainable? | **Yes on the ternary 8-layer branch:** H1 S3 `COMPLETE/PASS/CARVE-IS-TRAINABLE`; trained experts dominate recovery and the trained-router gate fires. No one-byte, rank, rate, 10B or scale claim. | canonical H1 S3 result + adjudication |
| Is one-byte + carve the right next trainable object? | **Yes on the matched 8L proxy:** R8 changes H0 by only −0.0000169 BPB, while hard k16 costs +0.209071; Phase B eligible. Rank is still broken at 9/160 free, 99/160 teacher-forced. | H2I Phase A |
| What was H2I Phase B's terminal verdict? | **`SCORE-ONLY`:** v4 hard BPB 0.905344219843237 beats applied one-byte 1.0190764652622473 by −0.11373224541901028 (`TRAINING-HELPS`) and router beats STATIC, but teacher-forced top-1 95/160 fails strict >99; combined gate false. V1–v3 remain `VOID_OPERATIONAL`; no H2I rerun, export, engine, rate, 10B, scale or target-donor promotion. | canonical H2I result + adjudication; brief addendum N |
| What does the rank fraction used by fast R128 cost post-hoc? | **At D=1536, r/D=1/32 costs +1.705835 BPB = 51.66% dense→chance gap; no width extrapolation.** | canonical E65 |
| What is the H4 aggressive-rank ternary step zero? | **CPU controls pass; Stage A RUNNING:** rank48 q/o factorization fires on 56 organs, full self-test passes, dense intact is `0.7675949641196625` BPB (`160/160` free and teacher-forced), and H4 INIT is `2.8035765393907672` BPB (`5/160` free, `21/160` teacher-forced). P1 is qualitatively right, but its registered BPB band `3.2–5.5` is falsified. Launch check 2026-09-16 ~09:21 UTC: acct3 authenticated as sirwildpino; private dataset once-created, server `READY`, 16/16 inventory/bytes exact, manifest SHA-256 `a97e444359684b2bd769932248493873fe7a7f4b4e63f3f811cf3caf17f2c331`, direct API GPU quota `0/30h`; kernel submitted once and `KernelWorkerStatus.RUNNING`. Pinned config: 4000 steps / bs2 / accum8 / lr2e-4 / every250 / max2.8h / seed1717 / one T4. No terminal checkpoint or CPU fp32 adjudication yet; all outcome bands unscored. | `probes/H4_AGGRESSIVE_RANK_STEP_ZERO.md` |
| Does a compressed pretrained 7B actually function on `engine.c`? | **Yes at one byte:** A2 0.674405 BPB vs fp32 0.674027; 4/5 greedy trajectories exact, 137/160 overall. It misses the strict 150/160 rank bar and has no rate claim. | canonical E66 run 2 / audit |

## Rules that prevent wasted work

- Do not re-measure a closed question unless the engine, donor, slice, or estimand changed and the new scope is preregistered.
- Do not repeat H4 step zero: its rank48 factorization, self-test, dense intact control and ternary INIT anchor are frozen in `probes/H4_AGGRESSIVE_RANK_STEP_ZERO.md`. Stage A is currently RUNNING; when complete, adjudicate only from its terminal checkpoint and CPU fp32 evaluation, and do not substitute E65 QO-48 for `h4_eval_init.json`.
- Do not reopen E38's post-hoc selector route with another router/carve search; an unattainable oracle already beat no usable signal.
- Do not infer 10B quality from E62's 0.5/1.5/3B int8 points; the measured damage is non-monotone. Measure the target or state it unknown.
- Do not infer rate from E63 Part A or E64/E65 quality. `G-E63d` is now the separate, measured clean-rate gate on **synthetic 10B noise weights**, not a pretrained-model goal pass.
- Do not count E63's earlier 2026-09-16 preflight refusals as timing attempts: no cell ran in those windows. A later fully admissible A10B sweep **did** run and closed `G-E63d`/`G-E63e`; do not rerun it or ask for more user process cleanup. The full runner did not finish: ancillary E61c refused at `quant=fp32` because `rep()` demanded ternary, and 3B was not reached. Repair those as a separately scoped measurement, never as a G-E63d retry.
- Do not treat E64 run-1 numbers as evidence. Run 2 is canonical; its quality result must not be turned into a timing, trained-format claim or scale extrapolation.
- Do not re-run E66's controls, score or rank: run 2 is canonical and closes the registered 7 B one-byte fidelity question. Change donor, quantizer, head format, slice or estimand first.
- For the next healing branch, preserve E66's result that one-byte format damage is negligible and E64's result that the post-hoc carve on that format is not; the trainable object is selection/routing, not the byte conversion itself.
- Do not launch the old `H2T` ternary-body proposal as written. E66 removes the ternary-format damage by keeping the faithful one-byte rung, while E64 shows the carve/routing hole remains and is larger there. A successor must be one-byte + trained selection/routing, with weight-only and activation-quantized partners kept separate.
- Do not repeat H2I Phase A: its matched write-once result is `0.810005595` uncarved R8 and `1.019076465` hard-k16, with rank 9/160 free and 99/160 teacher-forced. The Phase B threshold is the H2I value, not H1's ternary applied row. E64's all-28L format ordering and H2I's 8L ordering differ; neither may be scaled by layer count.
- Do not export H2I. The tagged-v2 container can represent the mixed object, but `qwen_export.py` cannot yet compose H0 factors, only eight carved layers, trained R8 masters, untouched fp32 matrices and exact fp32 trained routers. More importantly, v4 failed the combined score+rank gate, so the mapped seam in `audits/H2I_ENGINE_EXPORT_GAP_AUDIT.md` is closed and retained only to prevent duplicate analysis.
- Always assert engine `CONFIG`; always use a same-binary/same-arm control for a same-binary claim; never resume from a result file being validated.

## Experiment registry — foundations before the E-series

| ID | area | cataloged result / role | primary locations |
|---|---|---|---|
| P1 | nibble packing | packing correctness/benchmark foundation | `docs/research/donor_adaptation/probes/P1_NIBBLE_PACKING.md`, `benchmarks/donor_adaptation/nibble_pack.h` |
| P2 | block traffic | expert-path/block-kernel traffic decomposition | `docs/research/donor_adaptation/probes/P2_EXPERT_PATH_DECOMPOSITION.md`, `benchmarks/donor_adaptation/p1/` |
| P3 | donor shape | donor shape on engine | `docs/research/donor_adaptation/probes/P3_DONOR_SHAPE_ON_ENGINE.md` |
| R1 | donor runtime | runtime baseline/shape | `docs/research/donor_adaptation/probes/R1_DONOR_RUNTIME.md` |
| R2A | principal angle | low-rank prescreen; weighted-use geometry | `docs/research/donor_adaptation/probes/R2A_PRINCIPAL_ANGLE.md`, `benchmarks/donor_adaptation/r2/` |
| T1 | ternarization | donor ternarization baseline | `docs/research/donor_adaptation/probes/T1_DONOR_TERNARIZATION.md` |
| T2 | rule | activation-aware ternarization rule; GPTQ/data-aware controls | `docs/research/donor_adaptation/probes/T2_TERNARIZATION_RULE.md` |
| T2B | organs | organ coverage/ternarization comparison | `docs/research/donor_adaptation/probes/T2B_ORGAN_COVERAGE.md` |
| T3 | rotation | rotation under ternarization | `docs/research/donor_adaptation/probes/T3_ROTATION.md` |
| D0/D0C | density | coactivation and carve granularity; BPB can improve while ranking is floor noise | `docs/research/donor_adaptation/probes/D0_COACTIVATION.md`, `D0C_GRANULARITY.md` |
| D4 | reconstruction | Hessian/activation-weighted layer reconstruction | `docs/research/donor_adaptation/probes/D4_RECONSTRUCTION.md` |
| F1 | gates | gate predictor/controller apparatus and audits | `docs/research/donor_adaptation/probes/F1_GATE_PREDICTOR.md`, `docs/research/donor_adaptation/audits/` |

## Experiment registry — E1–E19: correctness, donor reality, first quality wall

| ID | verdict / status | do not repeat because |
|---|---|---|
| E1 | **PASS** engine BPB matches PyTorch | standing BPB protocol; reuse its slice/ID discipline |
| E2 | RMSNorm fold verified | fold/parity foundation; only revisit with a changed exporter |
| E3 | original target-scale ceiling corrected/withdrawn in ledger | old 1.17ms/ceiling arithmetic is not a current rate claim |
| E4 | attention reduction lever measured/corrected | attention mechanism established; see E40 for completed lever sweep |
| E5 | **OVERHEAD-DOMINATED** | decomposition is closed for its registered question |
| E6 | **GENERATION-CONFIRMED** | donor can generate on runtime; does not establish quality after conversion |
| E7 | **REAL-WEIGHTS-CONFIRMED** | synthetic proxy checked against 7B donor; use E15/E16 for quality |
| E8 | **CHAIN-CONFIRMED** | dependency chain identified; magnitude predictions not reusable as law |
| E9 | **GLUE-CONFIRMED** | one-thread SwiGLU glue fixed/measured |
| E10 | **CORE-BOUND** | packed kernel ceiling characterized |
| E11 | **NO-LIFT** | LUT was not a free speed lever; crossover/working-set matters |
| E12 | **CHANCE-LINE** | default ternarization sits at chance across tested scales |
| E13 | **LAYOUT-CONFIRMED** | blocked tile-major layout fixed LUT collapse |
| E14 | **CHEAP-BUT-NOT-NEUTRAL** | int8 activation cost is known on its registered donor/format; E59 is the trained-kernel follow-up |
| E15 | **DOES-NOT-PREDICT** | the quoted 7B artifact did not predict donor behavior |
| E16 | **SCORE-CROSSES-RANK-DOES-NOT** | fixing ternarization score does not preserve greedy ranking at 7B |
| E17 | **HEAD-IS-NOT-THE-MECHANISM** | head ranking/quality remains a bottleneck but this arm does not explain it |
| E18 | **CLIFF-NOT-SLOPE** | agreement floor and ranking ladder are discontinuous; 50 tok/s is a small active-weight budget |
| E19 | **CARVE-DOES-NOT-RANK** | FFN-only post-hoc carve cannot reach target and does not preserve donor ranking |

## Experiment registry — E20–E42: structural axes and engine economics

| ID | verdict / status | handoff |
|---|---|---|
| E20 | **RULE-EXHAUSTED / DRIFT-DOMINATES** | ranking and BPB disagree; do not search another ternary rule as if it were untested |
| E21 | **ATTENTION-YES-HEAD-NO** | activation-weighted q/o rank is promising; head remains resistant |
| E22 | **FITS-BUT-NOT-IN-THE-FORMAT** | cheap composition is super-additive and format/engine support matters |
| E23 | **ROUTER-HOLDS / ROUTER-COSTS** | oracle carve is a ceiling; real router cost must be priced |
| E24 | **DEPTH-RECOVERS** | depth can restore some budget, not donor quality |
| E25 | **RANK-PAYS-WHAT-IT-WEIGHS** | factored matvec exists and parity is gated; rate only after engine path is measured |
| E26 | **GATHERED-WEIGHT-COSTS-MORE** | gathered carve traffic is not equivalent to streamed traffic |
| E27 | **FLOOR-IS-NOT-ENOUGH** | low active-weight floor alone does not make donor quality survive |
| E28 | **CONTAINER-COSTS** | numerator/denominator was not constant at target shape |
| E29 | **RESIDUAL-RANKS** | residual layers can rank; E27's broad conclusion is narrowed, not erased |
| E30 | **AT-THE-WALL** | fast kernel reaches ~0.909 of measured wall at target shape |
| E31 | **GATHER-COSTS** | row-granularity gathered byte delivers ~58% of streamed rate; granularity is the lever |
| E32 | **ACTIVATION-COSTLY** | `--lutblk` activation quantization cost is +0.048991 BPB at registered cell |
| E33 | **LOCALITY-PARTIAL** | coarse carve helps less than desk model; target-shape result not resolvable |
| E34 | **FLOOR-IS-THE-WALL** | T10 with FFN gone is 20.03 tok/s, below 50; attention/head floor matters |
| E35 | **DEPTH-IS-HALVED** | measured 50 tok/s envelope gives L*≈19.33 under registered model |
| E36 | **TEN-B-NEAR-FIFTY** | 10B shape/parameter budget exists as a speed model, not quality evidence |
| E37 | **SPARSITY-DEGRADES** | applying the carve to trained weights costs ~+0.553692 BPB on ternary FFN |
| E38 | **SELECTION-IS-DEAD** | perfect selector oracle at 1.17% is above chance; post-hoc selector route is closed |
| E39 | **RANK-BUYS-SPEED** | same 10B parameter count, rank moved into attention and improved rate |
| E40 | **ATTENTION-LEVERS-EXHAUSTED** | R128 reaches ~113–130 tok/s with live FFN; this closes “next speed lever” searching |
| E41 | **VERDICT-UNRESOLVABLE** | partition result does not survive its registered control/addendum |
| E42 | **NO REGISTERED VERDICT** | predictability control went void; do not reuse it as a positive/negative finding |

## Registry — E43–E59: ledger-only / speed-mechanism continuation

These entries are represented primarily by briefs, engine runners/results and `SPEED_LEDGER.md`; not every one has a matching published probe in the current tree.

| ID | focus and status |
|---|---|
| E43 | vocabulary/token value; units half done, speed half permanently void |
| E44 | measured interval discipline and token worth; feeds the bar interpretation |
| E45 | speed dependence on weight values; controls/replications constrain value-based explanations |
| E46 | token cost at real context; context/stream slope accounting |
| E47 | local runner/log artifact (`e47_head_counts.py`, `e47_run.log`); no standalone probe/brief found |
| E48 | context slope decomposition; attention/context contribution |
| E49 | attention kernel was not switched on in the earlier probe; leads to E50 correction |
| E50 | fast kernel becomes default; `CONFIG` provenance requirement becomes explicit |
| E51 | softmax/libm `errno` cost; local math attribution |
| E52 | busy-box/occupancy cost; establishes paired/clean interval discipline |
| E53 | real exponential vs libm; direction evidence, no standalone rate claim in final ledger |
| E54 | droop is time/work-before-it, not a simple memory-load law |
| E55 | demonstration table with an interval that survives; interval/decision presentation |
| E56 | no canonical E56 probe/brief located; do not invent an experiment between E55 and E57 |
| E57 | first trained-model speed numbers; quality must be scored against same-format peers, not HuggingFace floor |
| E58 | **NO-HEADROOM**; prior ×1.75 packed headroom withdrawn to ×1.11–1.17 after denominator correction |
| E59 | **FAST-AND-LOSSY**; fast LUT kernel gives ~×1.253/×1.333 but changes 42–74% of trained tokens; kernel cannot be enabled blindly |

## Registry — E60 onward: current endgame

| ID | authoritative state | consequence |
|---|---|---|
| E60 | **rung built:** 1 B/weight trained artifact is 63.87 tok/s at 0.5B with +0.000066 BPB; 1.5B 21.39 tok/s; three measured bandwidths, not one | precision cliff is between 1 B and 0.5 B; use format-specific bandwidth |
| E61 | **CHAIN NOT BINDER:** breaking int8 FMA chain gives +5.07% (not predicted 12–22%); packed gains ×1.2745; four-chain bandwidth curve survives | value/chain ceiling is not enough; 1 B stream near 34–35 GB/s is the useful rung |
| E62 | **THE-COST-DOES-NOT-RANK:** 1B damage 0.000066/0.001252/0.000579 at 0.5/1.5/3B; non-monotone | no exponent or 10B extrapolation; 7B must be measured |
| E63 Part A | **PATH-EXISTS-AND-IS-EXACT:** 10B carved-int8 artifact 10,015,507,256 B; ids/logits inert controls pass; packed-vs-int8 carve top-1 100% | Part A itself is not a tok/s result; Part B A10B subsequently measured rate |
| E63 interim/addenda and clean Part B A10B | Earlier contended paired ratio ≈0.9826 corrected and composed ≈49 tok/s were estimates, not absolute readings. Three later preflights refused before timing. The subsequent clean 45/45 preflight and 10/10-cell A10B sweep closed `G-E63d` at 49.37 [49.18, 50.49] tok/s (`DESK-MODEL-HELD`) and `G-E63e` at 0.9952 [0.9557, 1.0177]. Full Part B stopped on E61c's fp32/ternary CONFIG mismatch. | Do not rerun A10B or relabel its noise-weight rate as trained-model success; E61c and 3B remain separate ancillary debts |
| E64 run 1 | **VOID / MALFORMED control:** cross-kernel `serial` vs `avx4` mismatch amplified through top-k selection (~1500×) | no cells may be quoted from run 1 |
| E64 run 2 | **CANONICAL:** `G-E64a2` full ladder exact under asserted `attn=serial`; `G-E64b` passes; `G-E64d=CARVE-IS-DEARER-ON-INT8`, int8-minus-ternary +1.13–1.43 BPB; `k=3` int8-carved 4.128510 BPB > chance | quality-only, one donor, post-hoc; no trained-format, scale or rate claim; apparatus commit lag is recorded in probe §8.1 |
| H0 | **format trainability evidence:** rank/carve-related training at 1.5B improves post-hoc quality, but free-running remains weak/partial at the measured checkpoints | training-in-format is distinct from post-hoc application |
| H1 | **COMPLETE / guard PASS / CARVE-IS-TRAINABLE:** S3 HARD trained-8L 0.9234896239116439 vs applied 1.0966361325809948 (delta −0.17314650866935088); S2 was 0.9625929082574385. Experts dominate recovery, trained router adds a material −0.02520343583515794, and `G-H1e` fires. S3 v1 remains operationally void before resume; v2 is the sole scientific continuation. Canonical records: `h1_eval_h1_s3.json` and `h1_eval_h1_s3_adjudication.json`. | ternary 8-layer trainability is closed. M.5.1–4 pass; M.5.5 is not independently adjudicated from the final pair. Do not select a periodic checkpoint. No one-byte, rate, rank, 10 B, scale or export claim |
| E65 | **RANK FRACTION COST:** r/D=1/32 on real 1.5B donor = 2.473430 BPB, +1.705835 = 51.66% dense→chance; rank damage non-monotone | no D=4096 extrapolation; this is a post-hoc floor, not a trainability verdict |
| E66 | **COMPLETE — SCORE SURVIVES, RANK BAR DOES NOT:** A2 0.674405 BPB, +0.000378 vs fp32; fold worth 0.0000376; repaired rank run 2 is 137/160 with 4/5 full trajectories exact | rank run 1 void; promote rank only from run 2; no rate, 10 B or scale-law claim |
| H2I Phase A | **PHASE-B-ELIGIBLE:** exact R8 8L baseline 0.810006 vs H0 0.810022; hard k16 1.019076, carve +0.209071. Rank 9/160 free, 99/160 teacher-forced, mean 3.675. | write-once matched baseline; do not remeasure. Train R8 selection/router and require score + rank improvement; no rate/10 B claim |
| H2I Phase B | **V4 TERMINAL `SCORE-ONLY`; V1–V3 `VOID_OPERATIONAL`:** frozen guard/evaluator rc 3 records hard BPB 0.905344219843237 vs applied 1.0190764652622473 (−0.11373224541901028, `TRAINING-HELPS`) and a useful trained router, but strict teacher-forced rank is 95/160 rather than >99. The combined gate is false. | closed score+rank cell; do not rerun or select a checkpoint. Do not export, claim rate/10B/scale, or promote a target donor from this branch |
| H4 step zero / Stage A | **CPU STEP-ZERO COMPLETE / STAGE A RUNNING:** rank48 ternary q/o on the pinned 1.5B donor passes G-H4a and G-H4b/c/d/f/g; INIT is 2.8035765393907672 BPB, 5/160 free, 21/160 teacher-forced. P1's qualitative ordering holds, numeric BPB band is falsified. Apparatus v2 adds nonterminal step-250 checkpoints; full self-test including G-H4h and independent 15-file bundle hash audit pass, manifest SHA-256 `a97e444359684b2bd769932248493873fe7a7f4b4e63f3caf17f2c331`. Kernel `sirwildpino/h4-rank48-stagea-v2` was `RUNNING` at the launch check; no terminal result yet. | do not repeat step zero or repush Stage A; adjudicate only its terminal checkpoint with CPU fp32; no 10B/rate/transfer/export claim |

## Open queue, ordered by information gain

1. **Do not reopen H2I:** v4 closes its one-byte score+rank cell as `SCORE-ONLY`; v1–v3 remain `VOID_OPERATIONAL`. Do not rerun, repush or choose a progress checkpoint.
2. **Keep the H2I terminal evidence scoped:** score and routing learned, but the strict teacher-forced rank clause failed. Its periodic step-2250 checkpoint is loadable (2,246 applied updates; ZIP CRC clean; 56 expected arrays), yet it cannot be selected to reopen the terminal verdict.
3. **Do not demand more user process cleanup or rerun `G-E63d`:** after the earlier refusals, its frozen guard passed and the A10B gate closed. The E61c/3B ancillary measurements did not complete and need a separately scoped repair if still useful; the fp32 CONFIG mismatch is not a reason to repeat A10B.
4. **Keep the H2I engine-export seam closed:** it requires a combined score+rank pass which v4 did not achieve. Do not patch E63d's pinned binary in place.

## Corpus inventory

- Donor-adaptation documentation: 69 probes, 87 briefs, 20 audits (including `TARGET_DONOR_LOCAL_INVENTORY.md`, `TARGET_DONOR_DECISION.md`, and the cross-axis map), 6 decisions and 9 prior-art notes, plus `INDEX.md` and `SPEED_LEDGER.md`.
- Donor-adaptation benchmark tree: hundreds of scripts/logs/results across density, ternary, engine, P1/R2/F1/S1; treat `archive/` as historical unless a canonical document points into it.
- Phase 64: 49 source/spec files under `benchmarks/phase64/`, including MVE data/logit/train stages and WS3–WS6 audits.
- Broader research includes CPU architecture, memory bandwidth, long-context/SSM retrieval and scale-up docs under `docs/research/`.
