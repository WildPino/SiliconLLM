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

x15 against r15: −0.28 [−2.45, +1.79]. The line was Δ ≥ −4.26 and beating the control. Controls hold: off, replay, slot error, plant.
Predictions: x15 −6 to −3.5, miss; LITTLE 35%; r15 −9 to −7, hit; x19 −4 to −1.5, hit. x9 and r9, reported, are still running and will be added.

## What it says

- **Likelihood moved, the panels did not.** x15 beats r15 on top-1 (0.770 against 0.748) and dBPB (+0.052 against +0.090), but gives the same points. A gain on likelihood alone would have misled here.
- **The damage is spread over layers.** Five layers bring back 10%; nineteen bring back 81%, at chat 28, and that arm fits the quality budget.
- **x19 costs 3.92 ms.** On the idle session's ceiling that gives 19.6-20.7 tok/s, against 21.27-22.57 without it. Exact experts do not fit under 24.7.
- **k = 4 on top of ternary hurts:** gsm8k falls to 66 from 81, and the arm is over budget.

## Thesis line

Exact experts at the one affordable depth do not repair the ternary loss. Reaching quality with ternary experts needs training, which is what test 2c is measuring.

## Parts specific to Ling

Flat sigmoid routing with 8 of 256 experts, the shared expert beside them, and which layers carry the return. The hook and the rule need only a top-k MoE.

## Recommendation

**Part 2 stays a training: leave exact experts out of the configuration, let the running chains finish (x9, r9, test 2c's trainings and routing), then run test 2c's exports and panels and deliver the stage.**
No new test is needed for this. The 24.7 line now also depends on the attention kernel (9-12 ms against the 5.15 it needs), so the delivery must weigh that against training.

**What would disprove it:** x9 beating r9 by more than the noise with Δ ≥ −4.26, which would make a middle depth worth pricing; or test 2c's stacked organs leaving the budget even after training, which would leave no route to quality.
