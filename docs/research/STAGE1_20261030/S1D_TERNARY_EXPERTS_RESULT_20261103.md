# Stage 1 (b): Ling-mini's routed experts in ternary, retrained block-wise. RESULT

2026-11-03. Protocol: `S1D_TERNARY_EXPERTS_PROTOCOL_20261102.md` (pushed `277b821`, before the lr pick, any training
or any panel). Records in `benchmarks/stage1/results/`:
- `s1b_lrpick.json`, `s1b_train.json`, `s1b_l10.json`: the lr pick and the trainings, per layer;
- `s1b_export_ph.json`, `s1b_export_tern.json`, `s1b_export_l10.json`: the exports, with the TQ2_0 exactness check;
- `s1b_run.json`: commands, CONFIG and MOE lines, binary and model hashes;
- `s1b_verdict.json`: from `s1b_verdict.py`, selftest PASS;
- `s1b_scores_manifest.json`: sha256 of the item-level scores kept on E:.

`s1b_posthoc.py` builds the tern_ph weights (initial scales, no data).

## Outcome in one line

**Undecided.** With all 19 MoE layers ternary and trained, tern lands at **−8.52 [−12.24, −4.64]**, UNDECIDED, and chat
20/40 fails the gate. The CI95 upper end sits 0.36 points above the OVER BUDGET line. **One layer alone costs almost
nothing:** tern_l10 has document top-1 0.938, KL 0.034, dBPB −0.0007 [−0.0026, +0.0010]. The loss comes from stacking
19 layers.

## The run

- **lr pick:** layer 1, held-out error **0.1103 at 1e-3**, 0.1111 at 3e-3, 0.1391 at 1e-2. lr = 1e-3.
- **Training (tern):** layers 1-19, 1.5 M tokens, 75 k held out, 29,962 s on the RTX 3060 (8.3 GPU-h).
  - Outlier groups at their max: 5,588.
  - Layer 1's super weight is exact after training (2.515625).
- **Training (tern_l10):** layer 10 alone on the original model's inputs, 1,351 s.
- **Total GPU:** 9.4 h with the lr pick, against the ~6.2 h the protocol estimated.
- **Exports:**

  | Arm | TQ2_0 tensors exact | Q4 sha256 |
  | --- | --- | --- |
  | tern | 57 of 57 | `f4501403…` |
  | tern_l10 | 3 of 3 | `10e58be7…` |
  | tern_ph | 57 of 57 | `4afe7d15…` |

- **Panels:** tern 7 of 7 COMPLETE in 2.0 h; tern_ph and tern_l10 documents COMPLETE.
- No VOID, and stderr is empty in every launch.

## Arms (Δ = points − ref's 55.19; the same 10,000 paired draws as part 1)

| Arm | Δ points [CI95] | Label | Chat /40 | Degen. | Top-1 | KL top-32 | dBPB [CI95] |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| tern | −8.52 [−12.24, −4.64] | UNDECIDED | 20 | 0 | 0.721 | 0.867 | +0.147 [+0.104, +0.197] |
| tern_ph | (documents only) | | | | 0.679 | 1.046 | +0.106 [+0.076, +0.140] |
| tern_l10 | (documents only) | | | | 0.938 | 0.034 | −0.0007 [−0.0026, +0.0010] |
| top6, part 1 (scale) | −0.96 [−2.81, +0.97] | WITHIN BUDGET | 29 | | 0.898 | 0.094 | +0.019 |

The chat gate needs 24/40 (ref's 28 minus 4).

**Per-panel accuracy (%):**

| | nq | tqa | gsm8k | arc | mmlu | chat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ref | 29.0 | 48.7 | 81 | 53.2 | 49.3 | 70.0 |
| tern | 18.0 | 30.5 | 87 | 51.2 | 43.3 | 50.0 |

## Predictions against outcome

| Prediction | Outcome |
| --- | --- |
| tern Δ −8 to −30, OVER BUDGET more likely than not; chat fails the gate | **hit**: −8.52 and chat fails; **miss, better**: UNDECIDED, not OVER |
| tern_ph document top-1 under 0.5 | **miss, better**: 0.679 |
| tern_l10 dBPB +0.003 to +0.03 | **miss, better**: −0.0007, inside the noise |

## Read here, not registered

- **One layer passes the stage's literal test on the documents.**
  - tern_l10 moves the documents less than part 1's top6 does: top-1 0.938 against 0.898, KL 0.034 against 0.094. top6
    fits the budget at −0.96.
  - tern_l10 dBPB, −0.0007, is under tern's dBPB / 19 (+0.0078).
  - The panel cannot see one layer, so this is a reading, not a verdict.
- **Stacking costs more than the sum, and the training buys little of it back.**
  - Post-hoc, all 19 layers give KL 1.046. Training brings that to 0.867, and top-1 from 0.679 to 0.721.
  - The training's gain falls with depth. Held-out error before → after:

    | Layer | Before | After |
    | ---: | ---: | ---: |
    | 1 | 0.254 | 0.111 |
    | 10 | 0.738 | 0.670 |
    | 19 | 0.832 | 0.732 |

  - Alone, layer 10 starts at 0.352 and ends at 0.227. In the stack it starts at 0.738, because most of its target
    error is the error carried in.
- **The carried error jumps at layer 6.**
  - The mean square of the error carried into each layer:
    - 7.5 at layers 2-5;
    - 33.5 at layer 6;
    - 66.5 at layer 19.
  - Not diagnosed. Layer 1's super weight is intact, so it is not that one.
- **dBPB disagrees with top-1 and KL.** Trained tern is closer to ref on top-1 and KL than tern_ph, but its dBPB is worse:
  +0.147 against +0.106. Rule 3 asks for both; here they point in opposite directions.
- **Knowledge recall loses most.**
  - Losses: nq −11.0, tqa −18.2, mmlu −6.0, arc −2.0, chat −8 passes.
  - gsm8k gains 6 points (81 → 87).
  - As in part 2, deep layers overfit: at layer 19, 0.651 on training rows against 0.732 held out.

## Decision (the registered rule)

- **Undecided:** stage 1 (c) measures the kernel at ternary, since the kernel question is independent of this result.
- **What a longer training would have to buy:**
  - **Points:** a fit needs the CI95 lower end at −5 or above. With this CI's half-width (about 3.8), that is Δ near
    −1.2: about 7 of the 8.5 points lost.
  - **Chat:** 4 more passes.
- **Where the loss is:** the measured loss is in the composition, not in one layer. Sequential block-wise training at
  1.5 M tokens recovers 17% of the post-hoc KL (1.046 → 0.867).

## Thesis line

The k = 4 ceiling of 25.1 tok/s assumes ternary experts. With them still undecided, the measured levers give:
- 22.9 tok/s with experts at Q2_K;
- 17.5 tok/s at Q4_K.

Ternary experts at this quality would need a training that sees the stacked model: end-to-end on the logits, or
block-wise with more data and with the carried error as part of the input distribution. That is beyond a few GPU hours
on this GPU.

## Next

Stage 1 (c): the ternary kernel at Ling-flash's shapes, starting with the existing kernels (bitnet.cpp, T-MAC). It is a
timing, so it runs on an idle machine.
