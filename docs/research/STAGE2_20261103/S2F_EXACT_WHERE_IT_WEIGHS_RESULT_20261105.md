# Stage 2, part 1, test 3b: experts exact where they weigh. RESULT

2026-11-05. Protocol `S2F_EXACT_WHERE_IT_WEIGHS_PROTOCOL_20261104.md`. The verdict opened with test 2b-bis's PASS. Records: `benchmarks/stage2/results/s2x_run.json`, `s2x_verdict.json`.

## Outcome: LITTLE COMES BACK

The judged arm x15 brings back 0.85 of the 8.52 points that ternary experts lose, and does not beat a random expert. Part 2 stays a training, as registered.

| Arm | What | Δ points [CI95] | Chat | Back | Desk ms | Top-1 / KL / dBPB on documents |
| --- | --- | --- | --- | ---: | ---: | --- |
| tern | all experts ternary | −8.52 | 20 | | | 0.721 / 0.867 / +0.147 |
| **x15, judged** | top expert exact, layers 15-19 | **−7.67 [−11.44, −3.75]** | 21, 1 degenerate | 0.85 | 1.03 | 0.770 / 0.588 / +0.052 |
| r15, control | random expert exact, layers 15-19 | −7.39 [−10.95, −3.72] | 23 | 1.13 | | 0.748 / 0.714 / +0.090 |
| x19, reported | top expert exact, layers 1-19 | **−1.61 [−4.61, +1.39]** | 28 | 6.91 | 3.92 | 0.825 / 0.331 / +0.022 |
| x15k4, reported | as x15, k = 4 | −10.45 [−14.59, −6.18] | 22 | −1.93 | 1.03 | 0.770 / 0.588 / +0.052 |
| x9, reported | top expert exact, layers 9-19 | −3.89 [−7.03, −0.82] | 27, 1 degenerate | 4.63 | 2.27 | 0.793 / 0.463 / +0.020 |
| r9, reported | random expert exact, layers 9-19 | −7.08 [−10.83, −3.25] | 23 | 1.44 | | 0.756 / 0.660 / +0.073 |
| r19, amendment 2 | random expert exact, layers 1-19 | −6.35 [−9.73, −2.95] | 24 | 2.17 | | 0.765 / 0.622 / +0.073 |
| t4, amendment 2 | all experts ternary, k = 4 | **−11.97 [−15.85, −7.91]** | 16, 2 degenerate | −3.45 | | 0.721 / 0.867 / +0.147 |
| x19k4, amendment 2 | as x19, k = 4 | **−0.61 [−3.75, +2.47]** | 31 | 7.91 | 3.92 | 0.825 / 0.331 / +0.022 |

x15 against r15: −0.28 [−2.45, +1.79]. The line was Δ ≥ −4.26 and beating the control. Controls hold: off, replay, slot error, plant.
Predictions: x15 −6 to −3.5, miss; LITTLE 35%; r15 −9 to −7, hit; x19 −4 to −1.5, hit.

## Read at k = 4 (amendment 2, added 2026-11-06)

The ceiling runs at k = 4, so the starting point beside −8.52 is **t4, −11.97**.

- **x19k4 stays a priced option,** by the registered rule: Δ −0.61 ≥ −4.26, and x19 beats r19 by +4.74 [+1.58, +7.82]. It labels WITHIN BUDGET, at 3.92 ms, and brings back 11.36 points from t4. x15k4 brings back 1.52 at 1.03 ms.
- **x9 beats r9 by +3.19 [+0.67, +5.75] with Δ −3.89:** the first thing that would disprove this result's recommendation has happened. x9 ran at k = 8 only, so its k = 4 number does not exist.
- **Caution on x19k4's +1.00 over x19: all of it is chat** (77.5 against 70). Its chat answers run 35.5 tokens, against ref's 52.0, x19's 59.7 and t4's 56.8, and the cause is unexplained. With x19's chat the arm reads −1.86. It says k = 4 costs nothing visible once the top slot is exact, not that k = 4 helps.
- **The k = 4 runs did decode at k = 4:** the records show slots dropped (moe_kept under moe_slots) and the outputs differ from x19's; x19k4's gsm8k equals x19's 85 by count only. Score panels are copied, as registered.

Predictions of amendment 2: x19k4 −5.5 to −3, miss, better; r19 −5 to −2.5, miss, worse; x19 beats r19 60%, hit; t4 −12.5 to −10.5, hit.

## What it says

- **Likelihood moved, the panels did not.** x15 beats r15 on top-1 (0.770 against 0.748) and dBPB (+0.052 against +0.090), but gives the same points. A gain on likelihood alone would have misled here.
- **The damage is spread over layers.** Five layers bring back 10%, eleven 54%, nineteen 81%, at chat 28, and that arm fits the quality budget.
- **x19 costs 3.92 ms.** On the idle session's ceiling that gives 19.6-20.7 tok/s, against 21.27-22.57 without it. Exact experts do not fit under 24.7.
- **k = 4 on top of ternary hurts:** t4 loses 3.45 points against tern and x15k4 2.78 against x15, and both are over budget. x15k4's gsm8k of 66 is not a mechanism: t4, with fewer exact weights, reads 79 on the same 100 items.

## Thesis line

Exact experts at the one affordable depth do not repair the ternary loss. Reaching quality with ternary experts needs training, which is what test 2c is measuring.

## Parts specific to Ling

Flat sigmoid routing with 8 of 256 experts, the shared expert beside them, and which layers carry the return. The hook and the rule need only a top-k MoE.

## Recommendation

**Part 2 stays a training: leave exact experts out of the configuration, let the running chains finish (x9, r9, test 2c's trainings and routing), then run test 2c's exports and panels and deliver the stage.**
No new test is needed for this. The 24.7 line now also depends on the attention kernel (9-12 ms against the 5.15 it needs), so the delivery must weigh that against training.

**What would disprove it:** x9 beating r9 by more than the noise with Δ ≥ −4.26, which would make a middle depth worth pricing; or test 2c's stacked organs leaving the budget even after training, which would leave no route to quality.

**Status, 2026-11-06:** both happened. x9 beats r9, and test 2c's stack is out of budget (S2G_STACKED_ORGANS_RESULT_20261106.md). x15 stays out, as decided. Exact slots at depth go to the delivery as a priced option, at k = 4, and the recommendation above stands only for the x15 depth.