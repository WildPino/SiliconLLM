# Stage 2, part 2, third test: the stack trained end to end on its scales. A TEST OF MECHANISM. PROTOCOL

2026-11-09. Ling-mini, one RTX 3060, then CPU panels. Decided on S2M's result: part 2 trains the stack together, by a method that runs on flash, and says which and at what cost on flash. Code in `benchmarks/stage2/`: `s2n_core.py` (the student), `s2n_flash_step.py` (cell F), `s2n_route_hist.py`, `s2n_teach.py`, `s2n_llama_docs.py`, `s2n_train.py`, `s2n_export.py`, `s2n_run.py`, `s2n_chain.py`, `s2n_verdict.py`. Selftests: core 14, export 5, verdict 32, all PASS. Every superseded dry-run reading is kept beside the one that counts. Dry run (`s2n_chain.py dry`) before the push.

## Question

S2M's stack ks_stk4 (bal's experts, organs at IQ2_KS, head stored) is −12.18 [−16.22, −8.13]. S2M's amendment 1 (below) splits it: the stack adds up to bal4 (−4.74) plus the organs at k = 4 (−7.41), and the organs lose 4.39 between k = 8 and k = 4. Does a loss that sees the whole stack at both k win back the organs' k = 4 loss, and more, with the codes as they are, by a method that runs on flash?

**Judged on points, chat and top-1, never on a loss.**

**Why a test of mechanism (rule 4).** Stage 1's label needs the CI95's lower end at −5 or above; with ks_stk4's half-width of about 4, WITHIN BUDGET needs a point near −1. No arm here is expected to fit. The cell decides whether end-to-end training of scales belongs in flash's recipe, and at what granularity.

## The method on flash, and its cost

S2E's test 4 (`results/s2f_feasibility.json`) found three methods that fit: M1 block by block (7.7 / 86.4 h at 1.5 / 16.8 M positions), M2 end to end on per-row scales with codes frozen (3.2 / 35.2 h, two T4, 28.8 of 30 GB), M5 block by block with the loss 4 blocks downstream (11.1 / 124 h). **M2 is the only one whose loss sees the stack.** Every organ training so far was M1 and lost to IQ2_KS post hoc.

**M2 on two T4 does not fit this stack, on the desk:** the page's 1.2 GB margin, less the stored head (+0.37 GB), bal's per-256 scales (0.78 GB) and IQ2_KS's extra bits (0.04 GB), is about 0, before fp32 logits and 32 layers of residuals (about 1.07 GB at 4,096 tokens). **The method is M2 placed on the 3060, as the page also sized it** (25.6 GB of codes streamed per pass, no hours): codes and scales in host RAM, one layer at a time to the GPU, bf16, forward, recompute and backward per layer. **Cell F measures it at flash's shapes first; if F fails, nothing else here runs, and the result says M2 does not run on flash.**

| Flash's recipe, if M2 is kept | GPU h | CPU h | Source |
| --- | ---: | ---: | --- |
| Teacher signal: S3's F0 top-32, 10 M positions in hand; to 16.8 M | 22 | | S3's 32 h per 10 M |
| Experts: bal by M1 at 16.8 M | 86.4 | | S2E, measured rate |
| Organs: imatrix and IQ2_KS | | ~6 | S2M's 66 min on mini, scaled by bytes |
| lr pick, 3 × 0.5 M positions | 7.2 | | F's rate (F × 366) |
| **M2 streamed, per row, 16.8 M positions** | **80.6** (80.6-80.8) | | **cell F, read** |
| Export | | ~2 | |
| **Total** | **196** | ~8 | |

## Cells

**F, flash's step (GPU, idle machine, before anything else).** Flash's shapes with random codes in this stack's form: experts as 2-bit codes with fp16 scales per 256; organs as bf16 values, as the training holds them; the head bf16 on the GPU. The training's step exactly: 2 sequences of 2,048 positions, documents packed with S3's corpus's length (mean 350) and a block-diagonal mask, k = 8; expert load sampled from mini's measured shares (`s2n_route_hist.py`, layer by depth), the busiest expert's load over the mean recorded. 5 steps timed after 2 of warm-up. **Fits** if peak reserved GPU ≤ 11.5 GB, host ≤ 70 GB and the **median** step × 4,096 ≤ 240 h (what the ~360 h recipe leaves); min and max beside it. Recorded beside: the dequantization's share of the median step, an estimate (elements unpacked per step, counted, times the unpack's cost per element measured apart), which separates this code's cost from the method's.

**Teacher (GPU).** S3's teacher tool on mini's Q4_K_M, the panels' ref, over the first 1.6 M positions of S3's corpus v1 (4,578 documents): top-32 log-probabilities at k = 8, flash's teacher's form. Read: 1,600,146 positions, 96 shards, exit 0. 1,501,450 train (4,293 documents); the rest held out, split by alternate documents into **pick** (lr; 143 documents, 55,990 positions) and **acts** (judging; 142, 42,706). On acts, +0.003 top-1 is about 2.8 standard errors if 5% of positions change their top-1. P0 at k = 4 scores +0.97, so a k = 8 teacher is the right target for both forms.

**Arms (GPU, then all seven panels on CPU, generation at k = 4 as ks_stk4):**

| Arm | What trains |
| --- | --- |
| **e2e_row, judged** | one multiplier per row of every organ and routed-expert matrix: what M2 holds on flash |
| e2e_grp | experts at bal's own per-256 scales, organs per row: what flash's memory limit costs |
| glob, trivial | five scalars: attention, shared expert, dense FFN, routed gate/up, routed down; the same 1.5 M positions |

- **Start:** ks_stk4, every tensor as the panels read it; bal's Q4_0 blocks outside the 2-bit form (8,094) kept as exact values (read back against gguf-py: maximum difference 0.0). Multipliers start at 1; codes, router, norms, embeddings, head frozen.
- **Loss:** truncated KL over the teacher's top 32 plus one bin for the rest mass, the student's full log-softmax in fp32. LLM-QAT Table 5: logits alone 63.1; hidden or attention terms hurt.
- **Sequences:** documents in a seeded order packed into 2,048 positions, never split, attention inside a document. A step is 2 sequences, about 4,096 positions; one epoch, about 366 steps.
- **Routing:** sequences alternate k = 8 and k = 4 mass 16; at k = 4 a document's first 64 positions route at k = 8, as the panels prefill their prompts (median 29 to 96 tokens) at k = 8 and decode at k = 4.
- **Optimizer:** Adafactor. It clips the update's RMS over a tensor, so a typical multiplier near 1 moves about lr per step; a single one can move more. lr picked from 1e-3, 3e-3, 1e-2 (typical reach over the epoch: 0.37, 1.1, 3.7) on the first 150 k positions, by held-out **pick** top-1. **Multipliers are held 1-D**, one second moment each: the dry run's first damage reading, with expert multipliers as [experts, rows] tensors, had Adafactor's factored (row × column) estimate give rarely routed experts steps scaled by their small row average: multipliers up to 3.09 and sign flips (down to −0.17) in 47 steps at lr 1e-2. On flash, one fp32 moment per multiplier is about 0.2 GB per row form. **One scalar per tensor over the multipliers** (t × m; see the damage control). Recorded per arm, over trained scales only, organs and experts apart: counts under 0.5 and over 1.5, min, max, the tensor scalars' range; held-out acts also at the pick's length (150 k positions), beside the end.
- **lr and reach:** the damage control runs at lr 1e-2. If the pick chooses 3e-3 or 1e-3, the arms move rows 3 to 10 times less than the control did; the result says so.
- **Scales:** applied in fp32 and rounded once, as the export writes them.
- **Export:** organs: bf16(value × multiplier) in ks_stk4's BF16 container. Experts: every Q4_0 block's d × its multiplier in fp16.

## Lines (registered; `s2n_verdict.py`)

- **e2e_row:** stage 1's label and chat gate, reported; FITS is not expected.
- **Training:** e2e_row − ks_stk4, paired CI95: **E2E PAYS** (lower > 0 and margin > 1.90), **E2E HURTS** (upper < 0), **NOT SHOWN**. A PAYS with margin ≤ 3.80 needs a second data order (e2e_row2, seed + 1), then PAYS only if both do.
- **The organs' k = 4 loss:** e2e_row − (bal4 + ks_org), ks_org at k = 8: **WINS BACK THE ORGANS' k = 4 LOSS** (lower > 0), **DOES NOT WIN IT BACK** (upper < 0), NOT SHOWN; ks_stk4's is −4.41. Beside it, generation at k = 4 (nq, tqa, gsm8k, chat), e2e_row − ks_stk4: **GENERATION GAINS / LOSES / NOT SHOWN**.
- **k = 8 panels:** arc + mmlu, e2e_row − ks_stk4: **HURTS AT k = 8** if the upper end < 0, else k = 8 HOLDS.
- **Documents:** dBPB per document, e2e_row − ks_stk4, paired over the 72 documents: **BETTER / WORSE ON DOCUMENTS**, NOT SHOWN.
- **Granularity:** e2e_grp − e2e_row: **GROUPS PAY** (lower > 0 and margin > 1.90), else NOT SHOWN.
- **Trivial, read only if E2E PAYS:** e2e_row − glob: **ROWS PAY** (lower > 0, margin > 1.90); else **A GLOBAL GAIN DOES IT** if glob − ks_stk4 ≥ 2/3 of e2e_row − ks_stk4; else NOT SHOWN.

## Controls

| Control | Rule | Failing means |
| --- | --- | --- |
| Parity, k = 8 and k = 4 mass 16 | untrained torch student against llama.cpp's score tool on S2M's documents (165,858 positions), same file, form and tool: top-1 agreement ≥ 0.95 and KL to ref within 5%, recomputed by the verdict | VOID |
| Planted, gross | torch at k = 4 mass 0 against llama.cpp at mass 16 must fail parity | parity not discriminating, said |
| Planted, subtle (reported with size) | k = 8 with bal's 8,094 exception blocks dropped; k = 8 with the shared expert's down rows of layers 5-14 × 0.97, the size the training moves | what parity cannot see is stated |
| Damage and recovery (dry run) | shared expert's down rows of layers 5-14 × 0.8; only those rows' multipliers train, lr 1e-2, 150 k positions; the same rows trained the same way from ks_stk4 as it is (damage_ref). **Repair = both land on the same weights:** the damaged run's multipliers over the reference's, geometric mean over rows, ≥ 1.125 (half way to 1.25), and the damaged run ends on held-out acts within half the damage's drop of the reference | the step cannot move a scale where the loss wants it: no arm runs |
| Training acts | held-out acts mean top-1 up by ≥ 0.003 | VOID |
| Scales positive | every trained effective scale > 0 (a sign flip is not a scale) | VOID |
| Export read back | every patched tensor recomputed from ks_stk4 and the scales, byte for byte; every other byte ks_stk4's; planted bytes caught (selftest) | VOID |
| Export kept | the exported file read as it is keeps ≥ 80% of the training's held-out acts top-1 gain | VOID |

**Parity, first reading (old code, `s2n_parity_*_first.json`).** k = 8: top-1 0.9676, KL 0.7852 against llama.cpp's 0.7836. k = 4 mass 16: 0.9748, KL ratio 1.002. Planted mass 0: 0.6461, KL ratio 0.819. The top-1 floor was first written as 0.97, before any measurement. The same student in fp32 reads 0.9730 at k = 8: the student's bf16 explains half a point. The rest is read as llama.cpp's arithmetic (Q8 activations on the Q4_0 experts and the Q6_K head), not measured apart; a defect in the student would move the KL, and the KL is equal within 0.23%. Floor set at 0.95, 1.8 points under the lowest true reading.

**Parity, dry run (the code that trains, `s2n_parity_*.json`).**

| Form | top-1 | KL ratio to llama.cpp | Parity |
| --- | ---: | ---: | --- |
| k = 8 | 0.9676 | 1.0021 | pass |
| k = 4 mass 16 | 0.9748 | 1.0023 | pass |
| planted: k = 4 mass 0 | 0.6461 | 0.819 | **fails, caught** |
| planted: k = 8, 8,094 exception blocks dropped | 0.8602 | 1.175 | **fails, caught** |
| planted: k = 8, shared down rows of layers 5-14 × 0.97 | 0.9623 | 1.030 | passes, **not caught** |

The ×0.97 plant moves top-1 by 0.5 points and the KL by 3%, inside both floors: parity cannot see a change of the size the training makes, so it guards the student's arithmetic, not its scales; the export checks guard those. Mini's measured expert load for F (24 documents, 48,903 tokens, k = 8): busiest expert over the mean 6.9 to 24.6 by layer, median 13.4.

**Damage, first reading (`s2n_dry_damage_first.json`): failed as registered, and the control was the wrong one.** It trained every multiplier from the damaged start (lr 1e-2, 47 steps, 152,549 positions). The ×0.8 cost held-out acts 0.89 points of mean top-1 (0.7243 → 0.7154). After training, acts read 0.7572, **3.3 points over the undamaged start** (k = 4 form KL 0.880 → 0.558), but the damaged rows' mean multiplier reached 1.008 against 1.125: the loss bought far more elsewhere than undoing a 0.9-point damage, and the other organ multipliers moved ±4.6% on average. A control that lets 15 M multipliers compensate cannot say whether the step moves a scale where the loss wants it. The same run exposed the factored-moment defect above.

**Damage, second reading (`s2n_dry_damage_second.json`): only the 20,480 damaged rows train.** They ended at a mean of 1.013 (0.64 to 1.36), yet acts reached 0.7293, over the undamaged start's 0.7243 (k = 4 form KL 0.906 → 0.761, against 0.880 undamaged). So the loss does not want those rows at 1: ks_stk4 is not the optimum along them, and "back to 1 / 0.8" was an unchecked premise. If the rows' optimum from the undamaged start is m, from the damaged one it is m / 0.8. **Third form, the one registered:** the same rows trained from ks_stk4 too (damage_ref), and repair read as both runs landing on the same weights, row by row. It can fail: a step that does not move, or moves without regard to the loss, leaves the ratio near 1.

**Damage, third reading (`s2n_dry_damage.json`, `s2n_dry_damage_ref.json`): NOT REPAIRED, as registered; the chain stopped.** Ratio of the damaged run's rows over the reference's: geometric mean **1.019** (10th-90th percentile 0.967-1.075) against 1.125. Held-out acts: the damaged run ends at 0.7303, the reference at 0.7326, a gap of 0.23 points within half the drop (0.44): that half passes. Rows of the two runs correlate 0.93, but they share the data order, so they share its noise. Reading, after the third review: the per-element normalized step moves a row by about lr × its own gradient's signal-to-noise ratio, which grows with the row's output. The step repaired where the loss pays. By the shared down row's norm, the ratio is 1.015 in the bottom half and 1.112 in the top 1% (1.155 in the top 0.1%), and the damage's held-out gap closed by 64% (k = 8 KL), 43% (k = 4 form KL) and 74% (top-1). A uniform ×0.8 is a tensor-wide direction, and per-row parameters reach it slowly. Not re-scored with a weighted ratio: that would be a fourth form chosen after the data.

**Design change (rule 5), then the registered third form again, thresholds unchanged.** One trainable scalar per tensor over its multipliers; the effective scale is t × m, what the export writes and the control records. t moves a tensor-wide direction at about lr per step; on flash it is a few thousand scalars. If it fails, no arm runs (the GPU chain checks it).

**Damage, fourth reading (`s2n_dry_damage.json`, `s2n_dry_damage_ref.json`, tensor scalars): REPAIRED, as registered.** Ratio of the damaged run's rows over the reference's: geometric mean **1.188** (5th-95th percentile 1.133-1.250) against 1.125, so 77% of the way to 1 / 0.8 in log terms. Rows of the two runs correlate 0.97. Tensor scalars: damaged run 1.10-1.27, reference 0.94-1.09. Held-out acts: the damaged run ends at 0.7326, the reference at 0.7329, a gap of 0.03 points against half the drop (0.44). No scale is non-positive in either run (damaged 0.76-1.70, reference 0.64-1.47). Both runs again end above the undamaged start (0.7243; k = 4 form KL 0.880 → 0.749 and 0.746), as in the second reading.

**F, read (`results/s2n_flash_step.json`, idle machine, RAM at 3200 MT/s): M2 per row FITS on flash.** Per row: median **70.9 s** per step (70.8-71.1), **80.6 h** for 4,096 steps against 240; peak GPU reserved 9.9 GB against 11.5, host 38.5 GB against 70; 49.3 M trainable scales; the dequantization about 27% of the step (58.9 h without it); sampled busiest expert over the mean 11.8 (median), 19.3 (max). Per 256 (e2e_grp's form): **does not fit the 3060**, 15.0 GB reserved (13.5 allocated, over the card's 12 GB: the driver spilled to host memory, so its median 411 s, 468 h, is not a clean hours reading); 390.6 M trainable scales. The judged method is per row, so condition 1 holds: M2 per row runs on flash at about 81 h, 196 GPU h for the recipe above. e2e_grp still runs on mini as the granularity question; if GROUPS PAY, flash's per-256 step needs a card over 12 GB or M5.

**The step, measured on mini (diagnosis, not a timing).** One step, profiled: 37 s, more than half of it weight-sized elementwise work, unpacking 2-bit codes in four strided passes. Unpacking is now one table lookup, bit-identical to the old code on random codes with exceptions, 2.6× faster per call; the step reads 27-36 s, median about 31. The rest is mostly elementwise work and routing.

## Predictions

| | Predicted |
| --- | --- |
| F | fits on memory; median 60-160 s per step: 70-180 h; FITS 75%. From a smoke run at 2 layers (one MoE), not idle, not a timing: 2.7 s per step, unpacking 23% of it, 5.8 GB reserved; mini's whole step is 31 s. **Read: per row 70.9 s, 80.6 h, 9.9 GB, FITS (hit); per 256 does not fit, not predicted** |
| e2e_row | Δ −11 to −6, chat 20-26; OVER 55%, UNDECIDED 45%, WITHIN ~0 |
| Training | E2E PAYS 45%, NOT SHOWN 50%, HURTS 5% |
| Organs' k = 4 loss | WINS BACK 25%, NOT SHOWN 65%; GENERATION GAINS 40% |
| k = 8 panels | HURTS 15% |
| Documents | BETTER 50% |
| Granularity | GROUPS PAY 35% |
| Trivial (if PAYS) | ROWS PAY 40%, GLOBAL 25% |
| Subtle plants | exceptions dropped not caught 80% (read: caught, a miss); shared × 0.97 not caught 70% (read: not caught, a hit) |

EfficientQAT's E2E-QP after Block-AP (Table 4, Llama-2-7B w2g64) closed a fifth of the gap to FP16 with per-group steps; per row is weaker by 32-128 times in parameters, hence the prediction.

## What a result changes (rule 5)

- **F fails on memory, or on hours with (1 − dequantization share) × median × 4,096 still over 240 h:** M2 does not run on flash; part 2's recipe is M1 or M5 only, and this training does not launch.
- **F fails on hours, but would fit without the dequantization:** this code does not run M2 on flash in time; the training does not launch, and the next step is a fused dequantize-and-multiply kernel, timed again by F, before M2 is dropped.
- **E2E PAYS:** M2 enters flash's recipe at F's hours. **HURTS:** it leaves, and flash's training is bal's M1 alone. **NOT SHOWN:** not shown at 1.5 M positions, 1/11 of flash's; M2 does not enter the recipe on this cell, and the result names the budget as the open reason.
- **WINS BACK:** the organs at 2 bits cost their k = 8 price only, and the next cut works on that. **DOES NOT:** the organs' k = 4 loss stays in the budget and the next protocol keeps one organ stored.
- **GROUPS PAY:** flash's M2 needs per-256 scales; F's memory decides whether the 3060 holds them, else M5. **A GLOBAL GAIN DOES IT:** flash's M2 trains five scalars.
- **In every case** the stack stays over budget; the next protocol keeps one organ stored, the one whose restoration on the best stack here gives most points per ms (CPU only).

## Reviewers' objections (rule 6) and answers

First round, 10 objections, all answered in draft 2: a test of mechanism, a teacher in flash's form, M2 moved to the 3060 and measured by F, a full cost table, both k in the loss, an e2e_grp arm, controls that can fail, a five-scalar trivial arm, amendment 1 before the push.

| # | Second round (draft 2): objection | Answer in draft 3 |
| --- | --- | --- |
| 1 | Blocking: Adafactor moves a multiplier ~lr per step; at lr 3e-4 over 36 steps the damage test cannot pass, and the lr grid freezes the arms | Verified in torch's source. lr grid 1e-3 to 1e-2 with each reach written; damage × 0.8 at lr 1e-2 (reach 0.37) with thresholds inside the reach |
| 2 | Blocking: F times one 2 × 2,048 micro-step while training ran one ~350-position document per micro-step (about 12 streaming passes per step); uniform expert load; organs as codes; fastest step | Documents packed into 2,048 with a block-diagonal mask in both; F samples mini's measured expert load; organs bf16 in F; median, with a 240 h ceiling |
| 3 | Blocking: no line answers the rewritten question; "against the parts" duplicates the training line | WINS BACK line against bal4 + ks_org, and a generation-at-k = 4 line; the duplicate removed. An organs-only arm would attribute the gain, not change the recipe: not added |
| 4 | Multiplier cast to bf16 in the forward; export rounds differently | Applied in fp32 and rounded once in the student; the export check measures what the container keeps |
| 5 | Export control tautological | Export kept: the exported file, read back as the panels read it, must keep 80% of the held-out gain |
| 6 | Floor moved by deleting the failed record; tools differ between k; verdict says 0.97 | First readings kept (`_first`), the verdict recomputes from numbers. Both k use the same tool and ubatch (S2L's score tool, ubatch 512; S2M's k = 8 run). Subtle plants added |
| 7 | glob trained 150 k positions: handicapped by budget | Same 1.5 M positions |
| 8 | "Acts" passes on any increase; pick and acts on the same set | Held-out split in two; acts needs +0.003 |
| 9 | Training form is not the panels' (prefill at k = 8) | First 64 positions of a document at k = 8 in the k = 4 form |
| 10 | Documents panel has no rule | Documents line, paired over documents |
| 11 | Costs | Updated below; F's prediction raised |

| # | Third round (draft 4, after the dry run): objection | Answer in draft 5 |
| --- | --- | --- |
| 1 | Blocking: the damage control's consequence softened in prose; the verdict VOIDs every arm with it failed; the GPU chain does not check it | Prose removed; the GPU chain stops unless parity and the damage control pass; table, verdict and chain agree |
| 2 | Blocking: NOT SHOWN mapped to "M2 leaves" while the control blames the budget | NOT SHOWN read as "not shown at 1.5 M positions"; HURTS alone removes M2 |
| 3 | The "noise-dominated" reading is not what the records show: the step repaired the rows where the loss pays (ratio by row norm, gap closed 43-74%); a uniform error is a tensor-wide direction that per-row parameters reach slowly. Proposed: one scalar per tensor over the multipliers, the control rerun in its registered form | Reading corrected; tensor scalars added; the control rerun, thresholds unchanged; not re-scored with a weighted ratio |
| 4 | The expert path has no post-fix record; a quiet element can still step ~10 × lr | A sign flip VOIDs the arm; counts per class recorded |
| 5 | glob's scalars move tensor-wide directions at ~lr, rows at lr × their ratio: ROWS PAY / GLOBAL partly measures Adafactor | The tensor scalars give e2e_row and e2e_grp the same tensor-wide reach as glob |
| 6 | F's 50% split is not tied to the ceiling; the share is an estimate | Rule now (1 − share) × hours against 240 h; labelled an estimate |
| 7 | lr picked at 47 steps for a 366-step run | Held-out acts recorded at 150 k positions and at the end |
| 8 | The control speaks for lr 1e-2 only | Stated beside the lr |
| 9 | Multiplier summary diluted by untrained scales | Trained scales only |

## Amendment 1 of S2M: ks_org at k = 4

Run on CPU while this protocol was written; result in S2M's result, `s2z_k4_verdict.json`. Valid. ks_org4 −7.41 [−10.90, −3.99], chat 22: UNDECIDED, chat gate fails. **The organs lose at k = 4**, ks_org4 − ks_org −4.39 [−7.44, −1.42], mostly chat (30 passes to 22). **Excess at k = 4 IT ADDS UP**, ks_stk4 − bal4 − ks_org4 −0.02 [−4.53, +4.54]. S2M's "compounding" was the organs at k = 4; P0 alone at k = 4 is +0.97. As S2M's amendment registered for ADDS UP, the target is ks_org4's loss with bal's experts in place: the question is rewritten, the WINS BACK line added, both k stay in the loss (IQ2_KS's imatrix saw k = 8 only).

## Literature (rule 2)

| Source | What it gives | Known failure |
| --- | --- | --- |
| EfficientQAT, arXiv 2407.11062, §3.3, §4.1, Tables 1, 2, 4 | E2E-QP: weights fixed, step sizes trained end to end, one epoch, 4,096 × 4,096 samples, lr 2e-5 at 2 bits; round-to-nearest 453.49 → 9.33 average perplexity, after Block-AP 8.53 → 7.68 | a 2-bit gap remains; depends on data |
| LLM-QAT, arXiv 2305.17888, Eq. 4, Table 5 | logit distillation | hidden or attention terms hurt (37.7, 61.1) |
| Adafactor, torch's implementation (`torch/optim/_adafactor.py`) | relative step: update × lr × RMS(parameter), update RMS clipped to 1 over the tensor; second moment factored into row and column sums for tensors of 2 or more dimensions | small parameters move slowly; one element can move more than lr when the rest of its tensor is quiet; the factored estimate misreads a row that gets gradient only sometimes (measured here, dry run) |
| QEP 2504.09629, CBQ 2312.07950 (read for S2G) | error grows through quantized blocks | full correction overfits small calibration |

## What this takes from the design (rule 1)

| Design element | Here | Why, at flash's scale |
| --- | --- | --- |
| Block-AP before E2E-QP | IQ2_KS post hoc for organs, bal's M1 for experts | stage 2's block trainings of organs lost to IQ2_KS post hoc |
| Per-group steps | per row, judged; per-256 as e2e_grp | per-256 on flash is ~400 M scalars; F measures whether they fit |
| Codes trained | frozen | M2; only M1 and M5 move codes on flash |
| Router, norms | frozen | one change at a time |
| Head at 2 bits | stored | as S2M |
| 16.8 M positions | 1.5 M | mechanism budget; flash's hours are at 16.8 M |
| Seeds | one, a second order in the gray zone | the noise is S2L's single pair, 1.90 |

## Thesis line (rule 4)

No outcome brings the stack within budget. E2E PAYS puts a measured, flash-runnable step in part 2's recipe at F's hours; the stack then sits at the S2M row's 43.05 ms, 23.23 tok/s on the desk, 2.56 ms over the 40.49 line.

## Parts specific to Ling

The numbers, the shared expert, one dense layer, the fused QKV, q/k norms per head (which erase a common scale on a head's Q or K rows: glob's QKV scalar acts on V only). Not specific: M2, streaming, packing, the truncated KL, multipliers over any codes.

## Cost

GPU, read so far: teacher 1.4 h; route shares 0.1 h; parity and plants 1.3 h; damage, first reading 0.9 h. Damage, second reading 0.4 h; third 0.8 h; fourth (damage, damage_ref) 1.1 h; F 1.0 h at an idle machine. Ahead: lr pick 1.4 h; e2e_row, e2e_grp, glob about 3.6 h each (366 steps of about 31 s, two held-out passes) with their export checks: about 18 h in all, one job at a time. CPU: panels about 5 h per arm, 15 h.
