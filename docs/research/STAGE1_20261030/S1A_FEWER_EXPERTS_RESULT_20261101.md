# Stage 1 (a), part 1: fewer routed experts per token on Ling-mini, no training. RESULT

2026-11-01. Protocol: `S1A_FEWER_EXPERTS_PROTOCOL_20261030.md` (pushed `44dd2b7` before the run; addendum 1, the
verdict's reader of `docs.top.bin`, pushed `14ae9ee` before the verdict). Records:
- `benchmarks/stage1/results/s1a_run.json`: commands, CONFIG, MOE lines, binary hashes;
- `s1a_verdict.json`: from `s1_verdict.py`, selftest PASS;
- `s1a_scores_manifest.json`: sha256 of the item-level scores kept on E:.

## Outcome in one line

Without training, k = 6 fits the budget but not the lever's 2.5-point share. Every k = 4 form misses: top4 −7.5
UNDECIDED, topw4 and Top-P OVER BUDGET, decode-only fails the chat gate. The rule therefore fixes **training_k = 4**
for part 2, ranking by the router's biased score, and carries no k = 4 form into stage 2.

## The run

- **Panels:** 84 of 84 COMPLETE, in one session (31-10 08:34 to 01-11 12:58); no VOID, stderr empty.
- **ref:** reproduces C50-S2's recorded T0 scores item for item on all six panels (0 differing items).
- **replay** (ubatch 256): identical to ref: Δ 0.00, document top-1 1.0000, KL 0. The noise floor of an equivalent run
  is zero, so "within replay" below means identical to ref.

## Arms (Δ = points − ref's 55.19; CI95 paired over items)

| Arm | Experts kept | Δ points [CI95] | Label | Chat /40 | Degen. | Fits | vs control, margin | Top-1 | KL top-32 | dBPB |
| --- | ---: | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| ref | 8 | 0 | — | 28 | 0 | — | — | 1 | 0 | 0 |
| top6 | 6 | −0.96 [−2.81, +0.97] | WITHIN BUDGET | 29 | 0 | yes, not the share | +11.9, beats | 0.898 | 0.094 | +0.019 |
| rnd6 | 6 | −12.83 [−16.25, −9.37] | OVER BUDGET | 24 | 0 | no | — | 0.764 | 0.598 | +0.093 |
| top4d | 8 prompt, 4 decode | −4.83 [−7.53, −2.17] | UNDECIDED | **22** | 0 | no (chat) | — | (copied) | (copied) | (copied) |
| top4 | 4 | −7.49 [−10.63, −4.38] | UNDECIDED | 25 | 0 | no | +29.2, beats | 0.794 | 0.408 | +0.079 |
| topw4 | 4 (by weight) | −9.97 [−13.69, −6.26] | OVER BUDGET | **21** | 0 | no | +26.7, beats | 0.797 | 0.398 | +0.075 |
| topp | 4.03 mean | −8.97 [−12.08, −5.89] | OVER BUDGET | 24 | 0 | no | +30.7, beats | 0.804 | 0.378 | +0.082 |
| rnd4 | 4 | −36.65 [−40.31, −32.91] | OVER BUDGET | 8 | 5 | no | — | 0.612 | 1.544 | +0.348 |
| rndp | 4.04 mean | −39.68 [−43.42, −35.76] | OVER BUDGET | 7 | 24 | no | — | 0.580 | 1.926 | +0.504 |
| top2 | 2 | −40.11 [−43.52, −36.44] | OVER BUDGET | 4 | 0 | no | +7.4, beats | 0.495 | 2.334 | +0.670 |
| rnd2 | 2 | −47.46 [−50.52, −44.30] | OVER BUDGET | 0 | 10 | no | — | 0.323 | 4.117 | +1.287 |

- **Chat gate:** an arm passes with at least 24 of 40 chat items and at most 3 degenerate rollouts.
- **top4d:** its score panels (arc, mmlu, documents) are ref's outputs by construction, because they hold only prompt
  positions. Its Δ therefore comes from generation alone: nq −2.8, tqa −2.2, gsm8k −9.0, chat −15.0.
- **Units:** KL is in nats on the reference's top-32 partition plus a rest bin, which carries 1.5% of ref's mass on
  average. It is a lower bound of the full KL. dBPB is on the 72 documents; its CI95 lies above 0 for every arm except
  ref, replay and top4d.

**Per-panel accuracy (%)**:

| | nq | tqa | gsm8k | arc | mmlu | chat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ref | 29.0 | 48.7 | 81 | 53.2 | 49.3 | 70.0 |
| top6 | 27.7 | 48.0 | 78 | 51.3 | 47.9 | 72.5 |
| top4 | 22.7 | 40.0 | 66 | 48.2 | 46.8 | 62.5 |
| top4d | 26.2 | 46.5 | 72 | 53.2 | 49.3 | 55.0 |

## Predictions against outcome

| Prediction | Outcome |
| --- | --- |
| top6 −0.5 to −3 | **hit** (−0.96) |
| top4 −4 to −12 | **hit** (−7.49, UNDECIDED) |
| top4d at least 2 points better than top4 | **hit** (+2.66), but it fails the chat gate |
| topw4 within 1 point of top4 | **miss**: −2.47 [−5.73, +0.80], worse, CI spans 0 |
| topp at least 2 points better than top4 | **miss**: −1.48 [−5.17, +2.17] at 4.03 mean experts |
| top2 below −15 | **hit** (−40.1) |
| random controls lose more than their top arms | **hit** at every k |
| top-1 falls and KL rises monotonically with fewer experts | **hit** (top-1 0.898 / 0.794 / 0.495; KL 0.09 / 0.41 / 2.33) |

## What the registered rule fixes

- **training_k = 4.** None of top4, topw4 and top4d fits the share, so part 2 trains at k = 4.
- **k4_form = none.** Stage 2 does not assume any untrained k = 4 form.
- **Ranking rule for part 2: the router's biased score.**
  - Top-P did not beat top4 at matched compute, so its rule is not adopted.
  - topw4 did not beat top4, so ranking by weight is not adopted.
- **The router is better than chance at every k.** Every top arm beats its random control, with a CI95 above 0. The
  case "train the router first" therefore does not apply.

Not registered, read here:
- On these panels the load-balancing bias does not cost quality: ranking by weight is no better at k = 4.
- Decode-only k = 4 is not free. It keeps the scored panels, but chat falls 15 points and gsm8k 9 points.

## Thesis line (desk model of stage 0, context 2048, all-ternary Ling-flash at the RAM wall)

The model is 32 GB/s plus 4.24 ms per token, which reproduces stage 0's 19.2 and 25.0 tok/s.

| k | MB read per token | Ceiling, tok/s | Measured on mini |
| --- | ---: | ---: | --- |
| 8 | 1,531 | 19.2 | — |
| 6 | 1,336 | 21.7 | below 24.7; the only k that fits the budget |
| 4 | 1,141 | 25.1 | above 24.7; fits the share only if training brings the CI95's lower end to ≥ −2.5 |

24.7 tok/s needs about 1,160 MB per token, that is at most ~4.2 experts. The lever reaches the success threshold only
through part 2. The figures are measured at Ling-mini's scale; the transfer to flash is a stage 2 check.

## Next

Part 2 of (a) is a short training at k = 4 (router and/or experts), with its own protocol. The failure criterion of
(a) is k = 4 not fitting the budget even with training. Its target is the 2.5-point share, on the CI95's lower end.
top4's CI here is about ±3 points, so a trained k = 4 has to land near −1: it must recover roughly 6 of top4's 7.5
points. top6's −0.96 is the reference for that.
The GPU is the local RTX 3060 (12 GB) or Kaggle. torch has to be reinstalled after the C: format.
