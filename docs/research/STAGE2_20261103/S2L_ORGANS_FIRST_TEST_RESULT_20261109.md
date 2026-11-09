# Stage 2, part 2, first test: the organs without a router. RESULT

2026-11-09. Protocol `S2L_ORGANS_FIRST_TEST_PROTOCOL_20261107.md`, amendment 1. A test of mechanism, on Ling-mini. Records: `benchmarks/stage2/results/s2y_verdict.json`, `s2y_run.json`, `s2y_train_{as_both,st_both2}.json`, `s2y_pick_{1,3,10}.json`, `s2y_ks.json`, `s2y_ks_levels*.json`, `s2y_export_*.json`.

## Outcome

All five sections valid. **IQ2_KS, without training, beats every training of the organs so far.** The grid moves them. Bal stands in its own container. The first stack is worse than its parts.

| Section | Label | Number |
| --- | --- | --- |
| Noise | | gap 1.90 points (st_both −13.36, st_both2 −15.26; paired CI95 [−4.56, +0.57]); top-1 gap 0.0027 |
| Grid | **THE GRID MOVES THE ORGANS** | as_both − sb_q40 +5.74 [+2.63, +8.91]; top-1 +0.066 [+0.060, +0.073] |
| Bal | **BAL STANDS IN ITS OWN CONTAINER** | bal4 − tq404 +6.97 [+3.97, +9.99]; k = 8 +4.78 [+1.50, +8.15] |
| Stack | **OVER BUDGET**, chat gate fails | stk4 −16.70 [−20.47, −12.78], chat 12; excess −5.70 [−10.11, −1.42]: **WORSE THAN ITS PARTS** |
| st_ffn | **IT IS SPREAD**; **OUTLIERS WEIGH** | see below |
| IQ2_KS | **IQ2_KS BEATS THE TRAINING** | against as_both, at 2.1875 bits |

| Arm | Δ points [CI95] | Chat, degenerate | Top-1 / KL / dBPB on documents |
| --- | --- | --- | --- |
| ref | | 28, 0 | |
| st_both (test 2c) | −13.36 [−16.98, −9.59] | 15, 2 | 0.631 / 1.569 / +0.467 |
| st_both2, second seed | −15.26 [−18.62, −11.87] | 11, 1 | 0.628 / 1.523 / +0.452 |
| sb_q40, st_both as Q4_0 | −13.50 [−17.30, −9.55] | 14, 5 | 0.631 / 1.574 / +0.470 |
| **as_both, judged** | **−7.76 [−11.36, −4.04]** | 21, 2 | 0.697 / 1.148 / +0.331 |
| **ks_org, IQ2_KS, no training** | documents and chat only | **30, 0** | **0.795 / 0.407 / +0.037** |
| stk4, bal + st_attn, k = 4 | −16.70 [−20.47, −12.78] | 12, 0 | 0.685 / 1.026 / +0.180 |

## The run

- **Trainings:** 20 layers logged on both, each under post-hoc. The asym lr pick on layer 2 read 0.238, **0.233**, 0.253 at scales 1, 3, 10: scale 3, as run, was the best. It ran on the idle GPU once both trainings had ended, before the chain's turn.
- **Exports:** every arm exact, every other tensor equal to its reference. Tool check equal on 30 items.
- **IQ2_KS:** the imatrix lowers the weighted error on 100 of 100 organ tensors (0.081 against 0.121); relative error 0.114, not a decoding fault. Added after the verdict, `s2y_ks_levels.py`: every block of 32 in ks_org's organs holds at most 4 values, and the experts are P0's types. The same count reads 3 on sb_bf and 32 on ff_keepout, so it separates.
- **Containers:** sb_q40 − st_both −0.14 [−2.19, +1.81]; tq404 − t4 +0.25 [−1.00, +1.53]; tq40 − tern +0.06. Activation precision moves little here.

## What it says

- **The noise is 1.9 points on one pair.** Every one-seed margin of the stage is above it: bal +6.97, the grid +5.74, the stack's excess −5.70, x19k4 over bal4 +4.14. It is a single difference, not a spread.
- **The grid gives the organs +5.7, three times what step 1's errors suggested,** with top-1 rising with the points. as_both's CI95 crosses −5: undecided on its own.
- **S2K's PAYS stands.** With ternary in Q4_0, bal4 still leads by +6.97. Its Open note is closed.
- **The stack compounds.** Bal at k = 4 and trained attention cost −4.74 and −6.25 apart, −16.70 together: 5.7 worse than the sum. nq and chat carry it: nq 13.7 against 19.7 (bal4) and 25.3 (st_attn), chat 12 against 24 and 22.
- **st_ffn's collapse sits in the shared expert at all depths, not in the dense FFN.** Restored as stored: the dense FFN gives back 0.04 of top-1 and 0.17 of chat; the shared expert of layers 1-9 0.50 and 0.74; of layers 10-19 0.46 and 0.87. Either half back is enough for most of it. Read as what the stack does without a group, not as shares.
- **Outliers weigh, and do not explain it.** In BF16, the 2.1% of shared groups the experts' rule marks give back 0.178 of top-1, against 0.027 and 0.028 for the same count at random; chat 7 against 4 and 5, out of 40. The collapse stays. Shared lr 9e-3 remains the suspect: it was not tested.
- **IQ2_KS post hoc puts the organs at the experts' level:** top-1 0.795, KL 0.407, dBPB +0.037 [+0.027, +0.048], chat 30 against ref 28; bal's experts alone read 0.802, 0.430, +0.043. Ternary post hoc with a row scale read 0.048. It has a scale per 32 weights where the trainings had one per row. The finer scales are a candidate cause, not a measured one.

## Predictions

| | Predicted | Read | |
| --- | --- | --- | --- |
| as_both | Δ −12 to −7, top-1 0.65-0.74, chat 16-24 | −7.76, 0.697, 21 | hit |
| Grid | MOVES 45%, NOISE 50% | MOVES | 45% |
| Gap, top-1 gap | 0.3-2.5, ≤ 0.015 | 1.90, 0.0027 | hit |
| stk4 | Δ −13 to −8, chat 18-24; OVER 70% | −16.70, 12; OVER | miss, worse; 70% |
| Excess | −2 to +2, ADDS UP 70% | −5.70, WORSE | miss |
| st_ffn | SPREAD 55%; ff_noearly most | SPREAD; most on top-1, not on chat | 55% |
| Outliers | WEIGH 60%; 0.2-0.6 | WEIGH; 0.178 | 60%; miss, lower |
| ks_org | top-1 0.70-0.85, KL 0.4-1.2, dBPB +0.10-0.50; BEATS 60% | 0.795, 0.407, +0.037; BEATS | dBPB miss, better |
| Containers | sb_q40 within ±1.5; bal +4 to +7.5, STANDS 85% | −0.14; +6.97, STANDS | hit |

## What this does not say

ks_org ran documents and chat only, so it has no Δ in points. It ran in a BF16 container: IQ2_KS's own kernel multiplies by Q8_K activations, unmeasured here; elsewhere containers moved under 0.3. Its 11% extra time is one cell at 4096². The stack has never run with IQ2_KS organs, nor experts at IQ2_KS. One seed pair; the lr pick reads block loss on one layer.

## Thesis line

The organs no longer need a training that has not worked: an existing format brings them to the experts' level on documents and chat. They do not reach the line together: stacking costs 5.7 points beyond the sum. On the desk, organs at IQ2_KS take 11% more of their 13.74 ms (attention 7.76 desk, shared 5.25, dense 0.73 desk): **41.29 + 0.25 + 1.51 = 43.05 ms, 23.23 tok/s**, a sum of cells with no quality behind it, 2.56 ms over the 40.49 line. The stack row as registered, FFN organs stored: 45.46 ms, 22.00 tok/s, over budget.

## Parts specific to Ling

The numbers, the outlier counts, the shared expert beside routed experts, one dense layer, the fused QKV, the imatrix's text. Not specific: the grid, IQ2_KS and its imatrix, the restoration method, the container arms.

## Recommendation

**Next, on CPU and without training: the first stack at 2 bits everywhere but the head, bal's experts at k = 4 with attention, shared expert and dense FFN at IQ2_KS with the imatrix, all seven panels. Beside it, as its trivial control, the experts at IQ2_KS too, post hoc.** About 10 h of CPU, no GPU. If the stack is within budget, it is the configuration to beat, at 23.23 tok/s on the desk. If it is over, part 2 trains that stack together, organs starting from IQ2_KS's values and attention trained with bal's experts in place, as the stack's WORSE asks. If the control matches bal, the experts' training is not what carries them. The shared expert's lr arm is not needed for this: IQ2_KS gives the shared expert back without training.

**What would disprove it:** the IQ2_KS stack losing its organs' gain once the experts are at 2 bits, with an excess as large as stk4's. Then the format is not the lever, the compounding is, and part 2 trains the stack whole whatever its start.
