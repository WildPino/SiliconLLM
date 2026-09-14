# SiliconLLM — research catalog and no-duplication handoff

**Snapshot:** 2026-09-14. **Latest discovered material:** H1, E64 run 2, E65, E66 preregistration. The user's remembered endpoint was E64; the tree contains later work, so this catalog includes it and labels its authority/status.

## Read this first

1. Canonical narrative: `docs/research/donor_adaptation/INDEX.md`.
2. Speed chronology/arithmetic: `docs/research/donor_adaptation/SPEED_LEDGER.md`.
3. Current owner actions: `docs/COMMUNICATION.md`.
4. Then open the cited probe, brief/addendum, runner and result JSON.

## Current state in one page

| question | current answer | authority |
|---|---|---|
| Can the CPU engine run a 10B shape above 50/100 tok/s? | **Yes as a shape:** E40 `R128` measured ~113–130 tok/s with live FFN, but weights were synthetic/noise. | canonical E40 probe/ledger |
| Can a pretrained donor be post-hoc carved to the required 1.17% FFN and retain quality? | **No for this donor:** E38's perfect per-token oracle is 4.131817 BPB, above 4.069819 chance. | canonical E38 |
| Is the 1B/weight rung faithful? | **Yes at measured 0.5/1.5/3B points, but no scale law:** E62 says damage is non-monotone. 7B remains unmeasured. | E60/E62; E66 prereg only |
| Does a 10B carved-int8 artifact exist and compute correctly? | **Yes:** E63 Part A, exact arithmetic/parity; its clean rate is still owed. | canonical E63 |
| What does carve cost on int8? | **E64 run 1 void. E64 run 2 current uncommitted evidence says dearer by ~1.31–1.43 BPB across ladder; needs promotion.** | working-tree JSON/log, not yet ledger |
| Is the carve trainable? | **Yes on an 8-layer branch:** H1 `TRAINING-HELPS`; router still moves, no full-stack claim. | canonical H1 |
| What does the rank fraction used by fast R128 cost post-hoc? | **At D=1536, r/D=1/32 costs +1.705835 BPB = 51.66% dense→chance gap; no width extrapolation.** | canonical E65 |
| What is the next real 7B quality check? | E66 one-byte-at-7B, preregistered and apparatus-ready; result absent. | E66 brief/runner |

## Rules that prevent wasted work

- Do not re-measure a closed question unless the engine, donor, slice, or estimand changed and the new scope is preregistered.
- Do not reopen E38's post-hoc selector route with another router/carve search; an unattainable oracle already beat no usable signal.
- Do not infer 10B quality from E62's 0.5/1.5/3B int8 points; the measured damage is non-monotone. Measure the target or state it unknown.
- Do not infer rate from E63 Part A or E64/E65 quality. `G-E63d` is the separate clean-rate gate.
- Do not treat E64 run-1 numbers as evidence. Run 2 is the only current candidate, and its JSON is uncommitted.
- Do not spend GPU hours on another donor conversion before E66 and the H1 decision are adjudicated; H1 specifically distinguishes post-hoc damage from training-in-format recovery.
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
| E63 interim/addenda | paired int8/packed ratio in contended run ≈0.9826 corrected; composed estimate ≈49 tok/s, CI ~45.5–51.1; no absolute claim | clean idle hour still required; attention is 72.3% of A10B-K3 token, FFN 11.4% |
| E64 run 1 | **VOID / MALFORMED control:** cross-kernel `serial` vs `avx4` mismatch amplified through top-k selection (~1500×) | no cells may be quoted from run 1 |
| E64 run 2 | **CURRENT UNCOMMITTED EVIDENCE:** `G-E64a` full ladder exact under asserted `attn=serial`; `G-E64b` passes; `G-E64d=CARVE-IS-DEARER-ON-INT8`, int8-minus-ternary +1.31–1.43 BPB | promote only after probe/ledger audit and commit; still quality-only, no rate |
| H0 | **format trainability evidence:** rank/carve-related training at 1.5B improves post-hoc quality, but free-running remains weak/partial at the measured checkpoints | training-in-format is distinct from post-hoc application |
| H1 | **TRAINING-HELPS:** trained 8L 0.962593 vs applied 1.096636; experts supply most early movement, router becomes helpful and still moves | do not declare finished; full stack/free-running and format choice remain open |
| E65 | **RANK FRACTION COST:** r/D=1/32 on real 1.5B donor = 2.473430 BPB, +1.705835 = 51.66% dense→chance; rank damage non-monotone | no D=4096 extrapolation; this is a post-hoc floor, not a trainability verdict |
| E66 | **PRE-REGISTERED, NO RESULT:** one-byte-at-7B, quality/rank only, `attn=avx4` asserted, controls against E62/E16 | run only after preserving prereg; no rate and no claim until result/probe exists |

## Open queue, ordered by information gain

1. **Promote/audit E64 run 2:** update `probes/E64_CARVE_ON_INT8.md`, `INDEX.md` and `SPEED_LEDGER.md` with the repaired-control result, preserving run 1 as void and labeling run 2 quality-only.
2. **Run/adjudicate E66:** 7B one-byte quality/rank. Controls are `G-E66a` (1.5B int8 E62), `G-E66b` (7B ternary E16), `G-E66c` (`CONFIG`), then score/rank gates. No rate.
3. **Get the clean CPU hour for `G-E63d`:** measure the actual 10B carved-int8 rate; do not replace this with another composed estimate.
4. **Decide H1 continuation / H2T:** use H1's router-vs-experts decomposition and E64's dearer-int8 result to choose whether the next training branch is one-byte or ternary; preserve the distinction between applied and trained.
5. **Only then spend T4 budget on the selected healing run:** H1/H0 show trainability, but no 10B training fits a T4; the T4 budget is for healing a smaller representative branch and validating the recipe.

## Corpus inventory

- Donor-adaptation documentation: 67 probes, 86 briefs, audits/decisions/prior-art plus `INDEX.md` and `SPEED_LEDGER.md`.
- Donor-adaptation benchmark tree: hundreds of scripts/logs/results across density, ternary, engine, P1/R2/F1/S1; treat `archive/` as historical unless a canonical document points into it.
- Phase 64: 49 source/spec files under `benchmarks/phase64/`, including MVE data/logit/train stages and WS3–WS6 audits.
- Broader research includes CPU architecture, memory bandwidth, long-context/SSM retrieval and scale-up docs under `docs/research/`.

