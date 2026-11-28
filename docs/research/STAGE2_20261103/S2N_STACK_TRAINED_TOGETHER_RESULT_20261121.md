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

## Amendment 1 result (2026-11-27): ROWS PAY

Protocol: amendment 1 (`b39db26`). e2e_rowx is e2e_row with scales exp(τ + θ) and lr 3e-3 decaying by a cosine to 0. It ran 454 steps over 1.50 M positions in 4.2 h of GPU, in one uninterrupted chain. Verdict: `s2n_verdict.json`, `amendment_1` and `gains` (selftest PASS 36).

**Validity: holds.** Every control passes: no scale ≤ 0 (range 0.72-1.78, 3 over 1.5), acts gain, export read back, export keeps 0.0548 of the trained 0.0548, panels complete.

| Arm | Valid | Δ vs ref | over ks_stk4 | chat | degenerate | acts top-1 (start 0.7243): at 150 k / end | acts KL k8 / k4p | docs dBPB, top-1 agreement |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| ks_stk4 | | −12.18 [−16.22, −8.13] | | 22 | 1 | | | +0.117, 0.723 |
| e2e_row | VOID | −13.42 | −1.24 [−4.41, +1.96] | 16 | 4 | 0.7649 / 0.7577 | −0.142 / −0.353 | +0.069, 0.705 |
| glob | valid | −9.05 | +3.12 [+0.51, +5.84] | 26 | 0 | 0.7326 / 0.7313 | +0.007 / −0.116 | +0.104, 0.730 |
| **e2e_rowx** | **valid** | **−5.16 [−8.61, −1.77]** | **+7.02 [+3.89, +10.12]** | **28** | **0** | 0.7631 / **0.7791** | −0.224 / −0.420 | **+0.015**, 0.741 |

- **The registered line:** e2e_rowx − glob = **+3.89 [+1.20, +6.62]**, lower > 0 and margin > 1.90: **ROWS PAY.** M2 per row enters flash's recipe at 80.6 h.
- **Training metric and panels now agree.** e2e_rowx has the largest acts gain (+5.5 points, still rising at the end where e2e_row fell) and the largest panel gain. Its documents' top-1 agreement rises (0.723 → 0.741) where e2e_row's fell. The disconnect read on 2026-11-21 came from the unstable lr, as the refutation clause allowed.
- **The excess, bal4 + ks_org as the parts:** **+2.61 [−1.21, +6.36]**, against ks_stk4's −4.42. Trained together, the stack is no worse than the sum of its parts. The TAPPE line's second branch holds.
- **The stack is still out of budget:** stage 1's label is UNDECIDED (lower −8.61, the label needs −5). Chat is 28, the reference's own.
- **Where the loss sits, per panel:**

  | Arm | nq | tqa | gsm8k | arc | mmlu | chat |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | ref | 29.0 | 48.7 | 81.0 | 53.2 | 49.3 | 70.0 |
  | ks_stk4 | 14.5 | 31.5 | 62.0 | 50.3 | 44.7 | 55.0 |
  | e2e_rowx | 16.3 | 35.7 | 77.0 | 54.5 | 46.7 | 70.0 |

  Generation at k = 4 and chat come back almost entirely; the closed-book questions (nq, tqa) do not, at −12.7 and −13.0 points.
- **Predictions:** ROWS DO NOT PAY 65% (miss); e2e_rowx +1 to +5 over ks_stk4, read +7.02 (miss, above).

**Thesis line.** This is the first trained 2-bit stack with every organ at 2 bits and k = 4 that sits near the budget: −5.16 with the reference's chat. It costs M2 per row, 80.6 h on flash at 16.8 M positions. This cell ran 1.5 M positions, one eleventh of that.

**Recommendation: buy tokens.** Run e2e_rowx, unchanged, at flash's 16.8 M positions. That needs a teacher pass past the current 1.6 M positions on CPU, and about 47 h of GPU on mini, so it goes to the reviewer (rule 6). Judge it on stage 1's label with the chat gate, with nq + tqa read beside it. This is the TAPPE branch for "out, without the refutation".

**What would refute it:** a 16.8 M run that gains less than 1.90 over this one. Then the remaining loss, which sits in closed-book recall, is not bought with tokens at this teacher. The next lever is then downstream blocks or one stored organ.

## Amendment 3 result (2026-11-27): the experts hold the recall, and the stack adds up

The run was `s2n_split.py chain`, on CPU only. Results are in `s2n_split_verdict.json`, `s2n_split_export_rowx_{org,exp}.json` and `s2n_run.json`. Both arms are **valid**: each export reads back and every panel is complete.

| Arm | Recall vs ref | All vs ref | Chat |
| --- | ---: | ---: | ---: |
| top4dm16: exact experts, k = 4 | −1.58 | +0.97 | 30 |
| ks_org4: organs IQ2_KS | −6.67 | −7.41 | 22 |
| rowx_org: ks_org4 with e2e_rowx's organ scales | −6.33 | −3.43 | 27 |
| bal4: bal experts | −10.25 | −4.74 | 24 |
| rowx_exp: bal4 with e2e_rowx's expert scales | −9.33 | −4.41 | 26 |
| e2e_rowx: the stack, trained | −12.83 | −5.16 | 28 |

- **Organ scales, recall:** +0.33 [−1.33, +2.00], **NOT SHOWN**. All panels: +3.98 [+0.54, +7.50]. The recovery on the rest of the panels comes from the organ scales.
- **Expert scales, recall:** +0.92 [−0.33, +2.17], **NOT SHOWN**. All panels: +0.33 [−2.24, +2.91]. Scales on bal's frozen codes buy nothing measurable.
- **Who holds it:** rowx_org − rowx_exp = +3.00 [+0.83, +5.25], **EXPERTS HOLD THE RECALL**.
- **Interaction:** +1.25 [−1.33, +3.83], **ADDS UP**. The trained stack loses what its trained parts lose, and no more.
- **Predictions:**
  - organs NOT SHOWN 55%: hit;
  - experts NOT SHOWN 60%: hit;
  - EXPERTS HOLD 55%: hit;
  - ADDS UP 60%: hit.

**What it says.** Scales do not bring back recall in either part, so more positions of the same scales do not either. The registered row is EXPERTS HOLD, ADDS UP: **amendment 2 is dropped**, and the next lever is the experts' recall.

**Recommendation: exact experts where they weigh, on the trained organs.** The cell is x19k4's experts under rowx_org's organs (IQ2_KS with e2e_rowx's organ scales), judged on stage 1's label with the chat gate, recall beside.

**The additive estimate.** It is −3.43 + (−0.61) − 0.97, about −5.0 on all panels, and recall about −10.5. That is at the budget's edge, so the protocol states the decision each outcome changes (rule 5) before the run. Its cost is panels on CPU plus the RAM of the exact layers, with no training.

**What would refute it:** the combination landing well below the additive estimate, which would mean the organ scales trained on bal experts do not carry over to exact ones. Then the organ scales are retrained with those experts in place, at 1.5 M positions (about 4 h of GPU).

## Amendment 4 result (2026-11-28): the slot pays on recall, the stack is at −2.48 and still UNDECIDED

The run was `s2n_exact.py chain`, on CPU only. Results are in `s2n_exact_verdict.json` and `s2n_exact_run.json`. Both arms are **valid**: every panel is COMPLETE, and every CONFIG and MIX line confirms the slot. Two runner stops happened before any line was read, both on the time guard; see the protocol's notes.

| Arm | Δ vs ref [CI95] | Label | Recall vs ref | Chat (ref 28) | nq | tqa | gsm8k | arc | mmlu | Desk ms, tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ref | | | | 28 | 29.0 | 48.7 | 81.0 | 53.2 | 49.3 | |
| e2e_rowx | −5.16 [−8.61, −1.77] | UNDECIDED | −12.83 | 28 | 16.3 | 35.7 | 77.0 | 54.5 | 46.7 | 43.05, 23.23 |
| **sx19** | **−2.48 [−6.10, +1.20]** | **UNDECIDED** | **−7.58** | 26 | 22.8 | 39.7 | 85.0 | 55.5 | 48.2 | 46.97, 21.29 |
| sx9 | −6.98 [−10.69, −3.25] | UNDECIDED, chat gate fails | −10.58 | 20, 1 degenerate | 20.0 | 36.5 | 80.0 | 54.8 | 47.9 | 45.32, 22.07 |

- **The slot on the stack:**
  - sx19 recall +5.25 [+3.42, +7.08], **PAYS**. All panels +2.68 [−0.31, +5.79], NOT SHOWN.
  - sx9 recall +2.25 [+0.67, +3.92], **PAYS**. All panels −1.82, NOT SHOWN, and its chat falls from 28 to 20.
- **Depth:** sx19 − sx9 on recall +3.00 [+1.33, +4.67], **DEPTH PAYS**. On all panels +4.50 [+1.19, +7.81].
- **The curve:**
  - sx9: 0.99 recall points per ms of slot, and −0.80 points overall.
  - sx19: 1.34 recall points per ms, and +0.68 points overall.

  Early layers matter once the late ones are exact.
- **Carry-over:** +2.53 [−1.31, +6.27] over the parts' estimate (−5.01), **ADDS UP**. e2e_rowx's organ scales hold under exact slots.
- **Predictions:**

  | Prediction | Outcome |
  | --- | --- |
  | sx19 Δ −1.5 to −4.5 | hit |
  | sx19 UNDECIDED 65% | hit |
  | sx9 Δ −2.5 to −5 | miss, worse |
  | Slot PAYS on recall (70% and 50%) | hit |
  | DEPTH PAYS 40% | hit |
  | Carry-over ADDS UP 60% | hit |

**What it says.** The registered row is "neither fits, the slot PAYS". sx19 is the best trained 2-bit stack measured so far: −2.48, with chat inside the gate. It misses WITHIN BUDGET by about 1.1 points on the CI's lower end. Recall still loses 7.58:
- **Organs:** rowx_org's recall, −6.33 with every expert exact, puts about 4.75 of it on the organs beyond k = 4.
- **Experts:** about 1.25 is left on the experts.

The slot can no longer pay for the rest. The 9-19 point is dominated: it costs chat and gives nothing overall.

**Speed.** sx19 is 6.48 ms over the 40.49 line on the desk. As registered, the slot's ms stay out of the ceiling until quality closes.

**Recommendation: the organs' recall, at sx19, with the cheap organs first.** The cell is one CPU arm: sx19 with the FFN organs (shared expert and dense FFN) exact from P0, and attention kept at IQ2_KS with its scales.
- **Desk cost:** +3.92 ms (the stored-FFN row of S2_DELIVERY_20261105.md, §5), so 50.89 ms and 19.65 tok/s.
- **Lines:** stage 1's label with the chat gate. The gain over sx19 on recall and on all panels.
- **Cost:** about 3.5 h of CPU and a new export that mixes tensor types (organs exact, experts bal).

**What would refute it:** the arm not beating sx19 on recall. The organs' recall would then sit in attention, whose stored form costs 16.17 ms. The quality line on mini would close only far from the speed line, and that is a decision on the stage's targets, not another cell.

## Amendment 5 result (2026-11-28), test of mechanism: the FFN does not hold the remaining recall

The run was `s2n_ffn.py chain`, on CPU only. Results are in `s2n_ffn_export.json`, `s2n_ffn_verdict.json` and the "sxf" record of `s2n_exact_run.json`. The cell is **valid**:
- the export read back 60 patched tensors out of 278, with cosine to e2e_rowx's own between 0.921 and 0.947 (mean 0.936), and every other tensor byte-equal;
- every panel is COMPLETE, and every MIX line confirms the slot in 19 layers.

| Arm | Δ vs ref [CI95] | Label | Recall vs ref | Chat (ref 28) | nq | tqa | gsm8k | arc | mmlu |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ref | | | | 28 | 29.0 | 48.7 | 81.0 | 53.2 | 49.3 |
| sx19 | −2.48 [−6.10, +1.20] | UNDECIDED | −7.58 | 26 | 22.8 | 39.7 | 85.0 | 55.5 | 48.2 |
| sxf | −4.35 [−7.79, −1.00] | UNDECIDED | −8.83 | 26, 1 degenerate | 21.5 | 38.5 | 77.0 | 55.7 | 47.4 |

- **The FFN on recall:** sxf − sx19 −1.25 [−2.83, +0.33], **NOT SHOWN**. On all panels −1.87 [−5.23, +1.43]. With the FFN organs exact, recall gains at most 0.33 points.
- **Reported:** sxf is UNDECIDED, with the chat gate passing. Its recall is −7.25 against top4dm16, and its desk cost is 50.89 ms, which is not a recipe row.
- **Predictions:**

  | Prediction | Outcome |
  | --- | --- |
  | THE FFN HOLDS RECALL 55%, gain +1 to +4 | miss: NOT SHOWN (40%), −1.25 |
  | sxf UNDECIDED 60% | hit |

**What it says.** The registered row is NOT SHOWN.
- **Where the recall sits.** rowx_org put about 4.75 recall points on the organs. They are not in the FFN, so by elimination they sit in attention. No arm read attention directly.
- **What exact FFN does.** Exact FFN values under scales trained beside IQ2_KS ones lean worse, mostly on gsm8k (85 to 77). That fits a trained form that has adapted to its neighbours. The CI does not separate it from zero.
- **The stage's position on mini.** sx19 is its best point: −2.48, with chat in the gate, at 46.97 ms on the desk. It is UNDECIDED by about 1.1 points on the CI's lower end.
- **What closing on mini would cost.** Attention's stored form costs 16.17 ms, so closing the quality line there would take sx19 to about 63 ms, far from the 40.49 line.

As registered, the next step is a decision on the stage's targets, not another cell.

**Recommendation: close stage 2 on mini with sx19 as its quality point, declared UNDECIDED, and move to speed.**
- **Speed:** find sx19's 6.48 ms over the line.
- **Quality:** attention's recall is carried as named debt and read on flash with the same panels and labels, where the thesis is judged.

**What would refute it:** sx19's form on flash with the CI's upper end below −5, or failing the chat gate. Attention's form would then come back as the lever, costed at flash's ms.
