# Stage 2: the fourth code, step 2. RESULT

2026-11-07. Protocol `S2K_FOURTH_CODE_STEP2_PROTOCOL_20261105.md`, amendments 1-3. A test of mechanism. Records: `benchmarks/stage2/results/s2v_verdict.json`, `s2v_train.json`, `s2v_export.json`, `s2v_export_void1.json`, `s2x_run.json`.

## Outcome: THE FOURTH CODE PAYS AFTER TRAINING

At equal bytes and equal training, routed experts in {−1.5, −0.5, 0.5, 1.5} beat ternary at k = 4 by **+7.22 points [+4.18, +10.31]**. One seed each: the CI covers items, not seeds.

| Arm | Δ points [CI95] | Label | Chat, degenerate | Chat tokens | Top-1 / KL / dBPB on documents |
| --- | --- | --- | --- | ---: | --- |
| t4, ternary, k = 4 (start) | −11.97 [−15.85, −7.91] | OVER BUDGET | 16, 2 | 56.8 | 0.721 / 0.867 / +0.147 |
| **bal4, k = 4, judged** | **−4.74 [−8.32, −1.11]** | UNDECIDED | 24, 0 | 31.9 | 0.802 / 0.430 / +0.043 |
| tern, k = 8 | −8.52 | | 20, 0 | | 0.721 / 0.867 / +0.147 |
| bal, k = 8 | −3.69 [−6.53, −0.79] | UNDECIDED | 27, 0 | 41.8 | 0.802 / 0.430 / +0.043 |
| x19k4, exact slot on ternary, for scale | −0.61 [−3.75, +2.47] | WITHIN BUDGET | 31 | 35.5 | 0.825 / 0.331 / +0.022 |

Margin at k = 8, against the −8.52: +4.83 [+1.54, +8.27]. Documents are scored at k = 8 only, as in test 3b.

| Panel | ref | t4 | bal4 | tern | bal |
| --- | ---: | ---: | ---: | ---: | ---: |
| nq | 29.0 | 16.5 | 19.7 | 18.0 | 20.5 |
| tqa | 48.7 | 29.3 | 37.5 | 30.5 | 39.5 |
| gsm8k | 81.0 | 79.0 | 86.0 | 87.0 | 82.0 |
| arc | 53.2 | 51.2 | 52.7 | 51.2 | 52.7 |
| mmlu | 49.3 | 43.3 | 46.8 | 43.3 | 46.8 |
| chat | 70.0 | 40.0 | 60.0 | 50.0 | 67.5 |

## The run

- **Training:** 12.57 h on the RTX 3060 at lr 0.001, 13.7 h with the pick, against 8.3 h expected. Layers 3-19 took about 2,450 s each; other jobs shared the GPU for part of it, so this is not a rate. It passed the protocol's 10 h without a reviewer, by amendment 1's decision.
- **Export:** the first was VOID, kept: one group's scale had crossed zero and the code wrote it as zeros (amendment 2). The second passes the control as registered, worst error 0.00385 against 0.0039. All non-expert tensors have the same types as ternary's file.
- **Panels:** the first run crashed with no output on Q4_0's path; binaries rebuilt, byte-equal on P0 (amendment 3).
- **Controls:** training lowered block loss on every layer. Planted checks held dry.

## What it says

- **Every panel moves the same way but gsm8k at k = 8,** and top-1 rises with the likelihood: 0.721 to 0.802, KL 0.867 to 0.430. It is not chat alone.
- **Block loss ratio 0.793:** it starts at 0.62 on layer 1 and rises to 0.92 on layer 19. The advantage thins with depth.
- **Trivial control:** step 1 found the same gain on a Gaussian. This is what a bell shape gives at equal bytes, not a finding about Ling. Here the panels judge it, and they agree.
- **Exact slots still lead on quality:** x19k4 beats bal4 by +4.14 [+1.48, +6.83], at 3.92 ms against at most 0.25 ms. Bal under an exact slot was not tried.
- **bal4 is not within budget by label,** and at −4.74 it would use nearly all of it.
- **Short chat answers are not explained:** bal4 31.9 tokens and x19k4 35.5, against ref 52.0 and t4 56.8. On both arms that stay coherent, k = 4 shortens the answers. bal to bal4 also gains 4 on gsm8k and loses 3 chat passes; 100 and 40 items.
- **One group of 256 weights trained to about zero** (layer 6, scale −1.0e-5). No group was protected against this.

## Predictions

| | Predicted | Read | |
| --- | --- | --- | --- |
| Block loss, bal over tern | 0.75-0.92 | 0.793 | hit |
| Δ bal, k = 8 | −7.5 to −5 | −3.69 | miss, better |
| Δ bal4, k = 4 | −10.5 to −7.5 | −4.74 | miss, better |
| Chat at k = 8 | 21-26 | 27 | miss, better |
| Margin at k = 4 | +0.5 to +3 | +7.22 | miss, better |
| Outcome | NO DIFFERENCE 55%, PAYS 40%, WORSE 5% | PAYS | 40% |

The correction −0.5·s·sum(x) costs 0.25 ms on one thread at flash's shape, an upper bound (S2E, amendment 3). The four codes cost no time (cell 7: 0.999, 1.002).

## What this does not say

The panels read Q4_0 holding the same values, so no 2-bit format has run. Bal here has a scale per 256 weights and a flag on 5,588 outlier groups; IQ2_BN has a scale per row and no flag. Row scales cost ternary experts nothing (test 2), and nothing has measured that on bal. Mini at 1.5 M positions is not flash.

**Open, 2026-11-07:** "same values" is not same arithmetic. Q4_0 multiplies by Q8_0 activations and TQ2_0 by Q8_K, so the margin mixes the grid with activation precision, by an unmeasured amount. S2L, amendment 1, measures it: ternary experts as Q4_0 (tq40, tq404). If bal4 − tq404's CI95 reaches 0, PAYS becomes NOT SHOWN.

**Closed, 2026-11-09 (S2L result):** bal4 − tq404 +6.97 [+3.97, +9.99], BAL STANDS IN ITS OWN CONTAINER; the container moves ternary by +0.25 [−1.00, +1.53]. PAYS stands. The noise between two seeds of an organ training was 1.90 points.

## Thesis line

The bytes do not change. At k = 4 the experts' loss falls from −11.97 to −4.74 for at most 0.25 ms. It does not reach the line alone: the organs are still out, and the budget has almost nothing left for them.

## Parts specific to Ling

The numbers, the outlier groups, layer 1's massive dimension, the lr. Not specific: the grid, the exact Q4_0 export, the method.

## Recommendation

**As registered: part 2's routed experts train in bal, and an engine writer and the corrected kernel path enter the plan.** Their scale layout gets decided first, with one arm at a scale per row.

**What would disprove it:** a second seed that closes the margin, or a real 2-bit layout that loses it.
