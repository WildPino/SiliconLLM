# C50 injection protocol: do auxiliary experts on v0 store and extract F0's facts? (a mechanism test)

27 October 2026. Design and review: [C50_INJECTION_REVIEW_20261027](C50_INJECTION_REVIEW_20261027.md) (`0c22742`).
Registered before any of the cell's F0 generations, samples or trainings runs. The independent review of this protocol
and its code is at the end: 18 objections, all taken.

## Question and thesis line

**Question:** v0 is trained on F0's text about facts it does not know. Does a sparse expert branch then answer
reworded questions about those facts, more than about twin facts it never saw?

**Thesis line (rule 4):** no outcome reaches the thesis's scale. The cell works on about 250 facts; 2× is on the order
of 10⁸. It is a mechanism test, with a mechanism-test budget. It prices F0's tokens per extracted fact, and that price
is what turns the choice of teacher for 2× into a number.

## Steps

All commands are `python -I benchmarks/chatbot50/<script>`. Every path is under `E:/…/inj/`.

| # | Where | Command | Output |
| --- | --- | --- | --- |
| 1 | local, after S3 block 2 and cell M part 2 | `c50_inj_data.py gen --model f0 --prompts E:/…/t1/fresh/inputs/<p>.tsv --out inj/f0_fresh/<p> --max-new 32` for nq and tqa | F0 on the fresh panel (~1 h) |
| 1b | local | `llama-batched-bench` (build b11040, CUDA) on F0: `-c 8192 -b 2048 -ub 512 -npp 128 -ntg 64 -npl 1,4,16,32 -ngl 99 --cpu-moe -t 6` | F0's batched decode rate, the design's missing price (~30 min) |
| 2 | local | `c50_inj_data.py pool` | the pool, the known-item panel, `p1.tsv`, `qa.tsv` |
| 3 | local | `gen --model f0 --prompts inj/p1.tsv --out inj/f0_p1 --max-new 96` | F0's first passage, from the question alone (~3 h) |
| 3′ | Kaggle | `inj-sample` | v0's 8 draws at temperature 0.7 → `inj/v0_sample/qa.samples.jsonl` |
| 4 | local | `c50_inj_data.py split` | the filters, then the clusters (shared gold or question), then the split. **STOP if a half has fewer than 100** |
| 5 | local | `gen --model f0` on `inj/more.tsv` → `inj/f0_more` (`--max-new 128`), on `type.tsv` → `f0_type` (4) and on `reword.tsv` → `f0_reword` (48); `gen --model t0` on `reword.tsv` → `t0_reword` (48) | p2-p5 for every fact (the twin half's are never trained), the answer type, two rewordings (~9 h) |
| 6 | local | `c50_inj_data.py build` | training files, evaluation sets, `manifest.json` |
| 7a | Kaggle, then local | `inj-mve`, then `c50_inj_verdict.py --pick lambda --inputs inj --mve <mve.json> --params inj/params.json` | λ0 |
| 7b | Kaggle, then local | `inj-eval-v0`; `inj-dev-<e\|l>-<1e-4\|3e-4\|1e-3>`; then `--pick lr --dev inj/dev` | the lr for E and for L; a §12 DECISION row before step 8 |
| 8 | Kaggle | `inj-train-<arm>-<seed>` for e-1, e-2, edense-1, l-1, l-2, s-1, qa1-1 | checkpoints at 25, 50 and 100 exposures, each with its sha256 in `train.json` |
| 9 | Kaggle | `inj-eval-<arm>-<ck25\|ck50\|final>` for every arm; `inj-eval-e-final_rot` (on e-1); `inj-general-<v0\|e\|l>` | collected as `evals/<arm>/<mark>/`, `evals/<arm>/general/`, `train/<arm>/` |
| 10 | local | `c50_inj_verdict.py --inputs inj --evals inj/evals --train inj/train --json results/c50_inj_verdict.json` | the outcome |

**The picks, by rule, from the records:**
- **λ0** is the smallest of {1e-6, 1e-4, 1e-2} whose 30 MVE steps (at lr 3e-4) meet all three conditions:
  - identity holds;
  - mean fire over steps 10-30 is ≤ 0.15;
  - no layer is below 0.01 at step 30.
  - If none qualifies: STOP.
- **lr, separately for e and l:** the one of {1e-4, 3e-4, 1e-3} with the highest dev accuracy (each set's accuracy on
  the dev facts, averaged over rew, orig and cloze), minus v0's. Ties go to the lower lr.
- **The dev facts never enter the verdict.**

## Rule (`c50_inj_verdict.py`, self-test 15/15)

The self-test covers:
- the rule on planted numbers, stable over 8 seeds;
- the whole reading path on a planted folder, with an intact provenance chain and with a broken one.

**Per fact:**
- accuracy on its valid rewordings (their mean), S2's alias scorer, greedy;
- accuracy on the original question;
- accuracy on the cloze.

**Contrasts:**
- **Δ(arm)** = mean acc(injected) − mean acc(twin).
- **Arm − 0** = Δ(arm) − Δ(v0), paired on the same facts.
- **Bootstrap:** facts redrawn within each half, 10,000 draws.
- **Arms E and L** are the per-fact means of their two seeds.
- **Checkpoint:** the final one is read; ck25 and ck50 give the dose curve.

| Check | Rule |
| --- | --- |
| **STOP** | any of: <br>• a broken provenance link (manifest → training text, lr and λ0 = `params.json` → each mark's sha256 → each evaluation's weights and inputs; the rot flag); <br>• a training not COMPLETE, or not v0 at step 0; <br>• v0's open book on the **rewordings** below 0.50; <br>• \|Δ(v0)\| > 0.08; <br>• v0's open book on the rewordings differs between the halves by more than 3 SE |
| E **passes** | E − 0 ≥ 0.10 with CI95 > 0, **and** E − 0 on T0's rewordings alone has CI95 > 0 |
| Other arms **pass** | arm − 0 ≥ 0.10 with CI95 > 0 |
| **The fact holds** | E − S has CI95 > 0; Δ(S) − Δ(0) ≤ half of E − 0; the answer bag (twin rewordings that e-1 answers with an injected gold) ≤ 0.05 |
| **Fire** (e-1) | replay ≤ 0.15, and no layer below 0.01 over the last 20 steps |
| **Selective** (e-1) | fire on the injected rewordings ≥ 1.5 × fire on the twin ones. Fire is counted only over each prompt's own span: the shared chat template and instruction are cut |
| **Guards** (e-1 and l-1 against v0) | known items, gsm8k and chat: A4's paired rule. Docs top-1 accuracy: CI95 low > −0.005 |
| **Stored** | cloze: arm − 0 ≥ 0.10 with CI95 > 0, for e, edense or l |

**Outcome**, the first row that applies:

| Outcome | Next |
| --- | --- |
| E passes, but the fact does not hold | redesign the text |
| **PASS:** E passes, fire within limits, guards hold, selective | a capacity cell; the choice of teacher goes to the user, priced |
| PASS, NOT SELECTIVE | router and addressing design |
| E passes, fire or guards fail | the failing part, one cell |
| E fails, E-dense passes | one cell on the fire schedule |
| E and E-dense fail, L passes | a memory layer, or a WISE-style side memory |
| all fail, stored | formats and QA mixing along the dose curve |
| all fail, not stored | one longer dose; then retrieval, priced by the open book |

**Reported, not gated:**
- each half against v0;
- QA1 − E;
- the original-question contrasts;
- E-rot;
- the dose curve;
- known items at every checkpoint;
- rejections and attrition by half;
- v0's open book on the original questions and on each rewording source;
- F0's tokens per extracted fact at each checkpoint, for E (all passages) and QA1 (the first passage).

**Power:** at about 125 facts per half and accuracy near 0.3, the standard error of E − 0 is about 0.04.
- At a true effect of 0.15, P(pass) is about 0.89.
- At 0.11, it is about 0.60.

## What it takes from the design (rule 1)

| The design asks | Here | Why |
| --- | --- | --- |
| Guards at every checkpoint | known items at every checkpoint; gsm8k, chat and docs at the final only, for e-1 and l-1 | gsm8k's 384-token generations cost about 40 min per run. The primary reads the final only |
| Fire on injected, twin and replay | for e-1 only. Selectivity is now injected against twin, not injected against replay | the second seed is an accuracy noise check. Twin prompts share the format, so they isolate the fact |
| Selectivity at the answer positions | not measured | the prompt span is where the gate must recognise the subject; answer positions need a per-row trace inside generation |
| 2 seeds | for E and L; one for E-dense, S and QA1 | budget; those arms are read against 0.10, not against each other |
| An lr pilot | on 30 dev facts; 60 steps against about 250 in the main runs; the MVE at lr 3e-4 only | budget; the pick is coarse, and a FAIL with a best lr at the grid's edge is reported as such |
| Facts per GB | not measured | as in the review |

## Predictions (scored, not gated)

| Prediction | Value |
| --- | --- |
| E − 0 | +0.15..+0.45 |
| E-dense − 0 | ≥ E − 0.05 |
| L − 0 | ≤ E + 0.05 |
| QA1 − 0 | 0.05..E |
| Δ(0) | within ±0.04 |
| Δ(S) − Δ(0) | ≤ 0 |
| E-rot | loses at least half of E − 0 |
| Fire of E | replay ≤ 0.12; injected ≥ 2× twin |
| Open book on the rewordings | ≥ 0.70 |
| Pool after the filters | 200-300 facts |
| F0 batched decode at 32 sequences | 3-10× its rate at 1 |

## Budget (ESTIMATE)

- **Local F0:** about 13.5 h, after block 2 and cell M part 2.
- **Local T0:** about 1 h.
- **Kaggle:** about 40 T4 × 2 session-hours over the three accounts:

| Item | Session-hours |
| --- | --- |
| Sample | 1 |
| MVE | 1.5 |
| Dev pilot | 5 |
| 7 trainings | 23 |
| 22 evaluations and 3 general runs | 10 |

- **A crash or a dead layer is a VOID:** recorded and reviewed before any rerun.

## Independent review (rule 6)

A fresh sub-agent, with no access to the conversation, read the thesis, the directive, the rules, the design, this
protocol and the code. Verdict: "register after fixes". All objections were taken:

| # | Objection | Severity | Response |
| --- | --- | --- | --- |
| 1 | `inj-dev-e-1e-4` splits into three parts | blocking | `split("-", 1)` |
| 2 | dev-e needs λ0 from a `params.json` that only the lr pick writes | blocking | two-stage pick: λ0 after the MVE, lr after the dev runs |
| 3 | prompt fire is averaged over the shared template, which pushes the selectivity ratio toward 1 | blocking | fire is counted over each row's own span (shared prefix and suffix cut) |
| 4 | rewordings can be string recall: the 8-gram rule cannot fire on short questions, and the rejection was one-sided | blocking | p2-p5 are generated for every fact; one rule for every half (a shared 4-gram of content words with the fact's own passages, or content Jaccard ≥ 0.5 with p3's question); attrition by half reported; T0's rewordings alone are a co-gate |
| 5 | the raw-text swap can leave a gold alias in S's passages | should-fix | a passage where an alias survives is dropped |
| 6 | the calibration STOP cannot fire (both halves at ~0) | should-fix | a balance check on v0's open book over the rewordings, at 3 SE |
| 7 | the open book used the original question | should-fix | open book on the rewordings is the planted positive, reported by source |
| 8 | facts with no rewording counted as fire 0 | should-fix | NaN, and means over facts with rewordings |
| 9 | no provenance chain | should-fix | text sha256 and marks' sha256 in `train.json`; the verdict checks every link and STOPs on a break |
| 10 | the same fact could sit in both halves | should-fix | clusters by normalized gold or question, split as units |
| 11 | a one-step dead-layer STOP ends a run | should-fix | 20 consecutive steps; a VOID, reviewed |
| 12 | silent drops: the batched F0 measurement, selectivity's change, the answer bag, the price along the dose | should-fix | step 1b; listed above; the answer bag is gated; the price is reported per checkpoint |
| 13 | the picks run under other schedules | note | stated in the table above; the MVE runs at 3e-4 |
| 14 | the lr pick pooled rows, so rewordings weighed double | note | per-set mean, then the mean over sets |
| 15 | `measure()` was not tested | note | a planted-folder smoke, with an intact and a broken chain |
| 16 | step 5 had no output folders | note | written above |
| 17 | the MDE claim did not match the rule | note | the power is given at 0.15 and at 0.11 |
| 18 | exposures differed by fact | note | each fact's documents are cycled up to the largest count, so every fact gets 100 |
