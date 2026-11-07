# Stage 2: the fourth code, step 2. A test of mechanism. PROTOCOL

2026-11-05. For stage 2, part 2. Ling-mini. GPU after test 2c's trainings, then CPU for the panels.

## Amendment 1 (2026-11-06, before the training)

The chain stopped at its limit: lr 0.001, projection 14.6 h, with the pick run beside the CPU chains (4,114 s against stage 1's 2,541 s). It was decided to run the training to its end with no limit, alone on the GPU after the idle cells: s2_tail_chain.py gpu --no-limit. Expected about 8.3 h, 9.5 h with the pick; the real time goes in the result. Nothing else changes.

## Amendment 2 (2026-11-07, after the training, before any panel)

The export failed its control: worst error 1.0 against 0.0039, so that export is VOID, kept as `results/s2v_export_void1.json`. One group fails of 19 layers' experts: layer 6, up_proj, expert 175, row 161, columns 1024-1279. Its trained scale crossed zero to −1.0e-5, and `bal_levels` divided only by positive scales, so it wrote zeros in place of ±5e-6 and ±1.5e-5. The code was wrong, not the rule. The fix divides by any scale that is not zero; d = −5.07e-6 is held exactly as an fp16 subnormal. The control stays as registered, and the export runs again. Worst error on every other block was 0.00385. Training took 12.57 h, past amendment 1's 8.3 h, with other jobs on the GPU for part of it. Nothing else changes.

## Amendment 3 (2026-11-07, before any panel output)

The re-export passes the control as registered: worst error 0.00385, sha `ec017f92`. The panel tool then crashed on bal's first panel, with no output: `ggml_gemv_q4_0_8x8_q8_0` stores to an unaligned stack slot, a path only a Q4_0 model reaches, and the same compiler fault seen in ik's flash path (S2E, amendment 3). `s2x_build_ua.ps1` rebuilds ggml-cpu alone with `-Wa,-muse-unaligned-vector-move`, and bal and bal4 use those binaries. Control, run on P0: outputs byte-equal to the pinned binaries on 30 nq items, 30 arc items and 6 chat items (`.jsonl`, `.pos.bin`). Weights, panels and the line do not change.

## Question

Step 1 found the routed experts' block loss 38% lower with {−1.5, −0.5, 0.5, 1.5} than with ternary, before any training, and equal to what a Gaussian gives. At equal bytes and equal training, does that survive on the panels?

## Thesis line (rule 4)

The bytes do not change, so neither does the desk's speed, apart from the correction, which is timed. Experts hold 15.3 B of mini's weights, and most of flash's. A pass changes the format part 2 trains; it does not bring the ceiling up. Mini at 1.5 M positions is not flash at its scale: this is a mechanism test with a mechanism budget.

## Literature (read in the papers)

ParetoQ, arXiv 2502.02631, Table 1 and §3.2.1, as in step 1: after training, 2 bits beat 1.58 on LLaMA-3 8B (71.2 against 69.0), with the SEQ grid. Known way to fail: Figure 5 favours a zero at 3 bits and above, and nothing there is at this training length. The estimators are stage 1 (b)'s: straight-through for the weights, LSQ for the scales.

## Design (`s1t_train.py --bal`, `s1t_export.py --bal`, `s2x_run.py`, `s2v_verdict.py`)

- **Reference: the ternary arm `tern` that gave −8.52,** not retrained: 19 layers, 1.5 M positions, one epoch, group 256, same seed and code. So the bal number compares with the −8.52 directly.
- **Bal:** the same command with `--bal`. w = s·q, q in the four levels, one scale per 256 weights, Lloyd start, floor-based rounding. The lr comes from the same grid on layer 1 (1e-3, 3e-3, 1e-2), by the same rule.
- **Outlier groups:** the same mask as ternary (5,588 groups on stage 1's arm). They keep ternary's grid: with no zero, a scale set by an outlier would write ±0.5·s on all 255 neighbours, and layer 1's massive dimension would go with them.
- **Export, quality only:** Q4_0 holds the grid exactly (d = s/2, q = ±1, ±3), so the panels read the trained weights, not a re-rounding. The quantizer would round these blocks by their maximum, so `write_bal` writes the bytes over its output and reads them back.
- **Panels:** the seven, at k = 8 (`bal`) and k = 4 (`bal4`, decode-only, k₂ = 16, as test 3b). Score panels are copied to k = 4, as amendment 2 of test 3b.
- **The correction's time:** −0.5·s·sum(x) per row, a cell of the next idle session (`s2l_cells.py`, see below).

## Line (registered; `s2v_verdict.py`, selftest PASS, 5 cases)

Judged at k = 4, the ceiling's routing: margin = points(bal4) − points(t4), paired CI95 over items.
- **THE FOURTH CODE PAYS AFTER TRAINING:** CI95 lower > 0.
- **IT IS WORSE AFTER TRAINING:** CI95 upper < 0.
- **NO DIFFERENCE AFTER TRAINING:** otherwise.

Reported beside it: the k = 8 margin against −8.52, block loss per layer, chat, top-1 and dBPB on the documents.

## Controls

| Control | Rule | Failing means |
| --- | --- | --- |
| Export | every expert tensor in the file equals the stored s·level per weight, within bf16's half step | VOID |
| Planted, dry | a 1% scale error in the blocks fails that check: read 0.0120 against a limit of 0.0039 | |
| Planted grid, dry | weights on the grid plus 5% noise: scale within 0.9%, four levels read back | |
| Training acts | block loss after training under its post-hoc start on every layer | VOID |
| Trivial | step 1's Gaussian twin: an untrained gain equal to a bell's. A margin here is judged on panels, not on block loss | |

## Predictions

| | Predicted |
| --- | --- |
| Block loss, bal over tern, mean of layers | 0.75-0.92: training closes part of the 0.62 |
| Δ bal, k = 8 | −7.5 to −5 |
| Δ bal4, k = 4 | −10.5 to −7.5 |
| Chat at k = 8 | 21-26 |
| Margin at k = 4 | +0.5 to +3 |
| Outcome | NO DIFFERENCE 55%, PAYS 40%, WORSE 5% |

A NO DIFFERENCE at 55% is not a predicted failure of the mechanism: the panels' CI is about ±2 points, so a real gain of 1 point cannot be told apart.

## What a result changes (rule 5)

- **PAYS:** part 2's routed experts train in bal, and an engine writer and the corrected kernel path enter the plan.
- **NO DIFFERENCE or WORSE:** part 2 stays ternary, and the kernel stays as it is. Block loss alone does not change the format.

## Parts specific to Ling

- **Specific:** the numbers, the outlier groups and layer 1's massive dimension, the lr.
- **Not specific:** the grid, the exact Q4_0 export, the method.

## What this protocol takes from the design (rule 1)

| Design element | Here | Why |
| --- | --- | --- |
| IQ2_BN's row scale | group 256, TQ2_0's | the −8.52 was trained at group 256; comparing with it is the point. Row scales need both arms retrained, over the budget |
| Ternary retrained beside it | not retrained | the same code and seed; only the form differs. One seed each: the difference between forms carries that seed's noise |
| The real 2-bit format | Q4_0 in the panels | same values; bytes and speed are not what panels measure |
| Outlier groups in bal | ternary grid, one flag per group | an extra bit per 256 weights in a real format; counted |
| Attention and shared expert | stay as stored | step 1's result |

## Cost

GPU: stage 1 took 9.4 h with the lr pick. Under 10 h, so rule 6 does not ask for a reviewer. If layer 1 and the pick project past 10 h, the run stops and the user decides. CPU: about 3 h of panels.
