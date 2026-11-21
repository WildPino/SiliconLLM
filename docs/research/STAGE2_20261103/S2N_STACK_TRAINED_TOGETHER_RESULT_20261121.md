# Stage 2, part 2, third test: the stack trained end to end on its scales. RESULT

2026-11-21. Protocol: `S2N_STACK_TRAINED_TOGETHER_PROTOCOL_20261109.md` (`1451a8a`). Verdict: `benchmarks/stage2/results/s2n_verdict.json`, from `s2n_verdict.py` (selftest PASS 32). A test of mechanism.

## What was read

**Validity, cell-wide: holds.** Parity at k = 8 and k = 4 mass 16 passes; the gross plant (k = 4 mass 0) is caught; the dropped exception blocks are caught, the shared × 0.97 is not (both as predicted). Damage repair passes (fourth reading, ratio 1.188). F: M2 per row runs on flash, 80.6 h.

| Arm | Valid | Δ vs ref | over ks_stk4 | chat | degenerate | acts top-1 (start 0.7243): at 150 k / end | docs dBPB, top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| ks_stk4 (untrained) | | −12.18 [−16.22, −8.13] | | 22 | 1 | | +0.117, 0.723 |
| **e2e_row, judged** | **VOID: 5 scales ≤ 0** | −13.42 [−17.43, −9.16] | −1.24 [−4.41, +1.96] | 16 | 4 | 0.7649 / 0.7577 | +0.069, 0.705 |
| e2e_grp | VOID: 5 scales ≤ 0 | −10.05 [−13.85, −6.25] | +2.13 [−0.97, +5.17] | 21 | 3 | 0.7682 / 0.7600 | +0.060, 0.715 |
| glob, five scalars | valid | −9.05 [−13.00, −5.10] | **+3.12 [+0.51, +5.84]** | 26 | 0 | 0.7326 / 0.7313 | +0.104, 0.730 |

Every registered line on e2e_row is VOID, and so is granularity. The trivial line was to be read only if E2E PAYS. The values below are reported, not judged.

- **The negative scales.** Both per-row arms end with 5 non-positive expert scales out of 15.1 M and 59.9 M (e2e_row: layers 2, 5, 13, 18, minimum −0.53). The organs stay positive (0.18 to 1.97). Expert scales spread wide: e2e_row has 23 k under 0.5 and 231 k over 1.5.
- **The lr was too hot for the full run.** It was picked at 150 k positions (1e-2: 0.7512 against 0.7467 at 3e-3) and run to 1.5 M in one epoch, with no repeated data. Both per-row arms peak on held-out acts at 150 k and lose 0.7-0.8 points by the end. The protocol stated the reach (4.5 at lr 1e-2) but registered no schedule.
- **The training metric and the panels part ways.** The per-row arms gain 3.3-3.6 points of held-out acts top-1 and cut the documents' dBPB by about 0.05. On the panels they do not gain: e2e_row's chat falls from 22 to 16, and its top-1 agreement on documents falls (0.723 → 0.705) while its BPB improves. glob gains 0.7 points on acts, and on the panels it leads e2e_row by **+4.35 [+1.20, +7.64]**.
- **The excess, with bal4 + ks_org at k = 8 as the parts** (the TAPPE-line reading; for glob and e2e_grp computed with the verdict's bootstrap, unregistered):

  | Arm | Excess |
  | --- | --- |
  | ks_stk4 | −4.42 [−8.30, −0.55] |
  | e2e_row | −5.66 [−10.02, −1.30] |
  | e2e_grp | −2.30 [−6.37, +1.79] |
  | glob | −1.31 [−5.31, +2.58] |

  With five scalars the excess is no longer WORSE. The stack is still out of budget: glob's upper end is −5.10.
- **Predictions:** e2e_row Δ −11 to −6, read −13.42 (miss). Training NOT SHOWN 50%, read VOID. Subtle plants: a miss and a hit, as already read in the dry run. F: hit.

## Conduct

The chains stopped eight times between 2026-11-10 and 2026-11-18: machine freezes and restarts, one Windows Update restart, one stop on request. Each restart resumed training from the arm's last checkpoint (`state.pt`: scales, optimizer and data cursor) and skipped finished panels. At most 25 steps were rerun each time. A resumed arm reads the same data order but is not bit-identical to an uninterrupted run.

## Thesis line

No arm brings the stack within budget. The best valid stack, glob, is −9.05 with chat 26. It costs five scalars on flash, nothing in hours. M2 per row, the step that fits flash at 80.6 h, has not been shown to add anything on the panels. Its arms are VOID for a known cause: an lr picked at a tenth of the run's length.

## Recommendation

**One amendment run: e2e_row again with the VOID's two causes removed. Scales are parametrized as exp(θ), positive by construction. lr is 3e-3 with a cosine decay to zero over the 1.5 M positions. Same data, teacher, export and panels. glob's +3.12 is registered as the bar: ROWS PAY if the rerun beats glob with lower > 0 and margin > 1.90.** Cost: about 5 h of GPU and 5 h of CPU, under the reviewer's 10 h.

- If ROWS PAY, M2 per row enters flash's recipe at 80.6 h.
- Otherwise it leaves, and flash's recipe takes the five scalars.

Either way, the stack stays out of budget. The next question is the experts at k = 4 (bal4 alone is −4.74), not the scales.

**What would refute it:** a valid per-row rerun that ties or beats glob on the panels while keeping its acts gain. That would make the disconnect above an artifact of the unstable lr, not of per-row capacity.
