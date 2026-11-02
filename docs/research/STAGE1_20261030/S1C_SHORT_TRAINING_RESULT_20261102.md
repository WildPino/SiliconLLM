# Stage 1 (a), part 2: decode-only k = 2 with k₂ = 16, untrained and after a short block-wise training. RESULT

2026-11-02. Protocol: `S1C_SHORT_TRAINING_PROTOCOL_20261101.md` (pushed `65d578c`, addendum 1 `875b881`, before any
panel or training result). Records in `benchmarks/stage1/results/`:
- `s1t_run.json`: commands, CONFIG, MOE and MASS lines, binary and model hashes;
- `s1t_lrpick.json`, `s1t_train.json`: the lr pick and the training, per layer;
- `s1t_export_trained.json`: the export of the trained weights;
- `s1t_verdict.json`: from `s1t_verdict.py`, selftest PASS;
- `s1t_scores_manifest.json`: sha256 of the item-level scores kept on E:.

## Outcome in one line

**k = 2 does not fit.** Untrained, top2dm16 lands at −3.33 [−6.31, −0.36], UNDECIDED. The short training makes it worse,
not better: −5.63 [−9.11, −2.10], chat 20/40, which fails the gate. The rule keeps **decode-only k = 4 with k₂ = 16**
for stage 2. Lever (a) does not fail: four experts fit at decode.

## The run

- **Step 1:** the first launch (01-11) was stopped by the session's memory guard before any panel completed (addendum
  1). Relaunched 02-11 09:20, alone: 7 of 7 panels COMPLETE in 43 min (generation 2,546 s; arc, mmlu and documents
  copied from ref).
- **lr pick:** layer 1, held-out loss 0.2421 at 1e-3, **0.2417 at 3e-3**, 0.3169 at 1e-2. lr = 3e-3.
- **Training:** layers 1-19, 1.5 M tokens, 75 k held out, 16,588 s on the RTX 3060 (4.6 GPU-h, against the ~2 the
  protocol estimated). 76 tensors changed (router, gate, up and down experts of 19 layers).
- **Export:** Q4_K_M with P0's imatrix, sha256 `28f3747f…`. The trained arm's panels: 7 of 7 COMPLETE in 2.4 h.
- No VOID, stderr empty in every launch.

## Arms (Δ = points − ref's 55.19; the same 10,000 paired draws as part 1)

| Arm | Δ points [CI95] | Label | Chat /40 | Degen. | Fits | Top-1 | KL top-32 | dBPB [CI95] |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| top2dm16 | −3.33 [−6.31, −0.36] | UNDECIDED | 25 | 0 | no | (copied) | (copied) | (copied) |
| trained | −5.63 [−9.11, −2.10] | UNDECIDED | 20 | 2 | no (chat) | 0.833 | 0.320 | +0.092 [+0.072, +0.115] |

- **TRAINING HELPS:** no. Δ(trained) − Δ(top2dm16) = −2.29 [−5.83, +1.36].
- **The trained arm's documents and score panels run at k = 8** on the trained weights: they measure the prompt path,
  and it moved (top-1 0.833, KL 0.320).

**Per-panel accuracy (%)**:

| | nq | tqa | gsm8k | arc | mmlu | chat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ref | 29.0 | 48.7 | 81 | 53.2 | 49.3 | 70.0 |
| top2dm16 | 23.7 | 45.5 | 77 | 53.2 | 49.3 | 62.5 |
| trained | 25.0 | 43.3 | 79 | 52.7 | 47.4 | 50.0 |

## Predictions against outcome

| Prediction | Outcome |
| --- | --- |
| top2dm16 −6 to −20, not WITHIN SHARE; its chat may fail the gate | **miss, better**: −3.33; **hit**: not WITHIN SHARE; chat passes (25/40) |
| trained recovers 20-50% of top2dm16's loss on generation | **miss, worse**: generation moves nq +1.3, tqa −2.2, gsm8k +2.0, chat −12.5 |
| trained: its k = 8 score panels lose 0-2 points | **hit**: arc −0.5, mmlu −1.9 |
| trained Δ −4 to −14 | **hit**: −5.63 |

## Read here, not registered

- **The training damaged the k = 8 path.** The documents, run at k = 8, give top-1 0.833 and KL 0.320, close to
  part 1b's top4m16 (0.797, 0.417). The prompt of every generation item also runs at k = 8 on these weights, so the
  damage reaches the generation panels too.
- **The training log could not see that damage.** Its k = 8 error was measured on inputs from the k = 2 stream, and
  there it went down at every layer (layer 6: 0.547 → 0.520). The rule-1 table of the protocol listed this as an
  approximation; the documents show it was not a small one.
- **Gains fall with depth:** the held-out error at k = 2 goes 0.341 → 0.199 at layer 1, 0.690 → 0.621 at layer 6, and
  0.818 → 0.791 at layer 19.
- **Deep layers overfit:** at layer 19 the error is 0.650 on training rows and 0.791 held out.
- **Untrained k = 2 is closer than predicted.** Its CI95 upper end is −0.36 and its lower end −6.31: 1.3 points more would
  move it to WITHIN BUDGET. Its loss is on nq (−5.3) and chat (−7.5), with gsm8k at −4.

## Decision (the registered rule)

- **Stage 2 carries decode-only k = 4 with k₂ = 16:** a ceiling of 25.1 tok/s on the desk model, a 1.6% margin over
  24.7.
- **k = 2 does not fit with this training at this scale:** 4.6 GPU-h of block-wise distillation on 1.5 M tokens.
- **Lever (a) does not fail:** four experts fit at decode, without training.

## Thesis line

The thin margin of k = 4 stays: 25.1 against 24.7. Stage 1 (b) and (c) read fewer bytes per expert, which widens the
margin without touching k. A trained k = 2 would need a training that keeps the k = 8 path: an objective measured on
the k = 8 stream too, or end-to-end on the logits. That is not a few GPU hours on this GPU.

## Next

Stage 1 (b): the ternary layer.
