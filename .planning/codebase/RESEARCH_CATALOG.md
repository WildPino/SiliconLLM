# SiliconLLM — research catalog and no-duplication handoff

**Snapshot:** 2026-09-15. **Latest material:** H1 session 3 v2 and H2I Phase B v4 are running on separate Kaggle accounts. H2I versions 1–3 are operationally void; v3 exposed a CPU-generator/CUDA-allocation mismatch before training, and v4 uses the narrowly preregistered repair with all scientific payloads unchanged. The user's remembered endpoint was E64; the tree contained E65, completed E66 and H2I. E63d's first admissible rate measurement has not yet started.

## Read this first

1. Search by mechanism/shape/format first: `docs/research/donor_adaptation/audits/AXIS_COVERAGE_AND_NO_DUPLICATION_MAP.md`.
2. Canonical donor chronology: `docs/research/donor_adaptation/INDEX.md`.
3. Speed chronology/arithmetic: `docs/research/donor_adaptation/SPEED_LEDGER.md`.
4. Current owner actions: `docs/COMMUNICATION.md`.
5. Then open the cited probe, brief/addendum, runner and result JSON.

The axis map also separates four lineages that must not be merged: the closed frozen-substrate work,
the parked trained CPU-native/Phase-64 ladder, active pretrained-donor adaptation, and prior-art/design
documents that contain no local measurement.

## Current state in one page

| question | current answer | authority |
|---|---|---|
| Can the CPU engine run a 10B shape above 50/100 tok/s? | **Yes as a shape:** E40 `R128` measured ~113–130 tok/s with live FFN, but weights were synthetic/noise. | canonical E40 probe/ledger |
| Can a pretrained donor be post-hoc carved to the required 1.17% FFN and retain quality? | **No for this donor:** E38's perfect per-token oracle is 4.131817 BPB, above 4.069819 chance. | canonical E38 |
| Is the 1B/weight rung faithful? | **Yes at 0.5/1.5/3/7B, without a scale law:** at 7B A2 is only +0.000378 BPB from fp32; E62's non-monotonicity still forbids extrapolation. | E60/E62/E66 |
| Does a 10B carved-int8 artifact exist and compute correctly? | **Yes:** E63 Part A, exact arithmetic/parity. Part B is hash-pinned and CONFIG-gated; its 2026-09-15 preflight refused before timing because 16/45 occupancy samples breached the bar. Clean rate is still owed. | canonical E63 §15 / preflight JSON |
| What does carve cost on int8? | **E64 run 1 void. Audited run 2 says dearer by +1.13–1.43 BPB across the ladder; at k=3 the int8 carve itself costs +1.912 BPB and lands above chance.** | canonical E64 probe §8 / ledger §66 |
| Is the carve trainable? | **Yes on an 8-layer branch:** H1 `TRAINING-HELPS`; router still moves, no full-stack claim. | canonical H1 |
| Is one-byte + carve the right next trainable object? | **Yes on the matched 8L proxy:** R8 changes H0 by only −0.0000169 BPB, while hard k16 costs +0.209071; Phase B eligible. Rank is still broken at 9/160 free, 99/160 teacher-forced. | H2I Phase A |
| Is H2I Phase B running? | **Yes:** Kaggle v4 is `RUNNING` with the device-safe real-control repair and rebuilt manifest `330fa237…`. Versions 1–3 are `VOID_OPERATIONAL`; none entered training or attempted an optimizer update. | H2I brief addenda B–J / readiness, launch and void records |
| What does the rank fraction used by fast R128 cost post-hoc? | **At D=1536, r/D=1/32 costs +1.705835 BPB = 51.66% dense→chance gap; no width extrapolation.** | canonical E65 |
| Does a compressed pretrained 7B actually function on `engine.c`? | **Yes at one byte:** A2 0.674405 BPB vs fp32 0.674027; 4/5 greedy trajectories exact, 137/160 overall. It misses the strict 150/160 rank bar and has no rate claim. | canonical E66 run 2 / audit |

## Rules that prevent wasted work

- Do not re-measure a closed question unless the engine, donor, slice, or estimand changed and the new scope is preregistered.
- Do not reopen E38's post-hoc selector route with another router/carve search; an unattainable oracle already beat no usable signal.
- Do not infer 10B quality from E62's 0.5/1.5/3B int8 points; the measured damage is non-monotone. Measure the target or state it unknown.
- Do not infer rate from E63 Part A or E64/E65 quality. `G-E63d` is the separate clean-rate gate.
- Do not count E63's 2026-09-15 preflight refusal as a timing attempt: no cell ran and canonical `e63_part_b.json` was absent. Reuse the committed hardened runner at the next quiet window; do not repeat Part A/parity/interim as substitutes.
- Do not treat E64 run-1 numbers as evidence. Run 2 is canonical; its quality result must not be turned into a timing, trained-format claim or scale extrapolation.
- Do not re-run E66's controls, score or rank: run 2 is canonical and closes the registered 7 B one-byte fidelity question. Change donor, quantizer, head format, slice or estimand first.
- For the next healing branch, preserve E66's result that one-byte format damage is negligible and E64's result that the post-hoc carve on that format is not; the trainable object is selection/routing, not the byte conversion itself.
- Do not launch the old `H2T` ternary-body proposal as written. E66 removes the ternary-format damage by keeping the faithful one-byte rung, while E64 shows the carve/routing hole remains and is larger there. A successor must be one-byte + trained selection/routing, with weight-only and activation-quantized partners kept separate.
- Do not repeat H2I Phase A: its matched write-once result is `0.810005595` uncarved R8 and `1.019076465` hard-k16, with rank 9/160 free and 99/160 teacher-forced. The Phase B threshold is the H2I value, not H1's ternary applied row. E64's all-28L format ordering and H2I's 8L ordering differ; neither may be scaled by layer count.
- Do not assume a passing H2I bundle can already be exported. The tagged-v2 container can represent the mixed object, but `qwen_export.py` cannot yet compose H0 factors, only eight carved layers, trained R8 masters, untouched fp32 matrices and exact fp32 trained routers. The router is especially not equivalent: today's exporter ternarizes it. See `audits/H2I_ENGINE_EXPORT_GAP_AUDIT.md`; implementation is conditional on the combined H2I score+rank gate.
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
| E63 Part A | **PATH-EXISTS-AND-IS-EXACT:** 10B carved-int8 artifact 10,015,507,256 B; ids/logits inert controls pass; packed-vs-int8 carve top-1 100% | rate gate `G-E63d` remains void/owed; Part A is not a tok/s result |
| E63 interim/addenda | paired int8/packed ratio in contended run ≈0.9826 corrected; composed estimate ≈49 tok/s, CI ~45.5–51.1; no absolute claim. Addendum D pins runner/engine/artefacts/sidecars and gates every future `CONFIG`. The 2026-09-15 45 s preflight refused at 16/45 breaches, before any cell. | a genuinely quiet window is still required; reuse the committed runner, not earlier controls |
| E64 run 1 | **VOID / MALFORMED control:** cross-kernel `serial` vs `avx4` mismatch amplified through top-k selection (~1500×) | no cells may be quoted from run 1 |
| E64 run 2 | **CANONICAL:** `G-E64a2` full ladder exact under asserted `attn=serial`; `G-E64b` passes; `G-E64d=CARVE-IS-DEARER-ON-INT8`, int8-minus-ternary +1.13–1.43 BPB; `k=3` int8-carved 4.128510 BPB > chance | quality-only, one donor, post-hoc; no trained-format, scale or rate claim; apparatus commit lag is recorded in probe §8.1 |
| H0 | **format trainability evidence:** rank/carve-related training at 1.5B improves post-hoc quality, but free-running remains weak/partial at the measured checkpoints | training-in-format is distinct from post-hoc application |
| H1 | **TRAINING-HELPS; S3 RUNNING:** trained 8L 0.962593 vs applied 1.096636. S3 kernel v2 is the first scientific continuation; v1 is operationally void before resume because Kaggle exposed the old dataset version under READY. Addendum R freezes an executable write-once guard around the unchanged CPU evaluator. | wait for terminal v2; then download the final pair once and invoke `h1_s3_adjudicate.py`. Never select a periodic checkpoint. No one-byte, rate, rank or 10 B claim |
| E65 | **RANK FRACTION COST:** r/D=1/32 on real 1.5B donor = 2.473430 BPB, +1.705835 = 51.66% dense→chance; rank damage non-monotone | no D=4096 extrapolation; this is a post-hoc floor, not a trainability verdict |
| E66 | **COMPLETE — SCORE SURVIVES, RANK BAR DOES NOT:** A2 0.674405 BPB, +0.000378 vs fp32; fold worth 0.0000376; repaired rank run 2 is 137/160 with 4/5 full trajectories exact | rank run 1 void; promote rank only from run 2; no rate, 10 B or scale-law claim |
| H2I Phase A | **PHASE-B-ELIGIBLE:** exact R8 8L baseline 0.810006 vs H0 0.810022; hard k16 1.019076, carve +0.209071. Rank 9/160 free, 99/160 teacher-forced, mean 3.675. | write-once matched baseline; do not remeasure. Train R8 selection/router and require score + rank improvement; no rate/10 B claim |
| H2I Phase B | **V4 RUNNING; V1–V3 VOID_OPERATIONAL:** v4 passed local bundle, identity, quota, inactive-kernel and exact remote-inventory gates. Its bundle changes only `h2i_qat.py`; all eight scientific payloads are unchanged. The seven CPU adjudicator blobs and exact terminal commands are frozen before output in `h2i_phase_b_v4_adjudication_freeze.json`. | wait for terminal v4 event; validate log/artifacts and adjudicate once only if the final outputs pass. Score runs before rank; evaluator exits 2/3 are scientific outcomes, not rerun triggers. No checkpoint selection |

## Open queue, ordered by information gain

1. **Wait event-driven for H1 S3 v2 and H2I v4:** both are `RUNNING` on separate accounts. Do not repush, poll rapidly or select a checkpoint from progress diagnostics.
2. **When H2I returns, validate v4 and run its single CPU fp32 adjudication:** require deployable BPB `<1.019076465`, free rank `>9/160`, teacher-forced `>99/160`, and mean rank `<3.675`. If score fails, do not run rank as rescue.
3. **When H1 returns, validate the first periodic checkpoint and run one CPU fp32 evaluation:** compare only against addendum M's frozen gates.
4. **Get a genuinely quiet 10–15 minute CPU window for `G-E63d`:** its 45 s all-samples guard must pass before timing. The 2026-09-15 refusal ran zero cells and is not a result.
5. **Only if H2I's combined score+rank gate passes, build its engine export seam:** follow `audits/H2I_ENGINE_EXPORT_GAP_AUDIT.md`; first preserve the fp32 trained router exactly, then test router compression as a separate treatment. Do not patch E63d's pinned binary in place.

## Corpus inventory

- Donor-adaptation documentation: 69 probes, 87 briefs, 18 audits (including the cross-axis map), 7 decisions and 9 prior-art notes, plus `INDEX.md` and `SPEED_LEDGER.md`.
- Donor-adaptation benchmark tree: hundreds of scripts/logs/results across density, ternary, engine, P1/R2/F1/S1; treat `archive/` as historical unless a canonical document points into it.
- Phase 64: 49 source/spec files under `benchmarks/phase64/`, including MVE data/logit/train stages and WS3–WS6 audits.
- Broader research includes CPU architecture, memory bandwidth, long-context/SSM retrieval and scale-up docs under `docs/research/`.
