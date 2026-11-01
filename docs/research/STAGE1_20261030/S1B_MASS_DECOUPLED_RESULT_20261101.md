# Stage 1 (a), part 1b: k = 4 with the routing mass decoupled from the kept experts, no training. RESULT

2026-11-01. Protocol: `S1B_MASS_DECOUPLED_PROTOCOL_20261101.md` (pushed `6f9e5a9` before the run). Records:
- `benchmarks/stage1/results/s1m_run.json`: commands, CONFIG, MOE and MASS lines, binary hashes;
- `s1m_verdict.json`: from `s1m_verdict.py`, selftest PASS;
- `s1m_scores_manifest.json`: sha256 of the item-level scores kept on E:.

## Outcome in one line

Decoupling the mass recovers most of k = 4's loss on Ling-mini's sigmoid router. top4m16 goes from −7.49 to −1.74
(WITHIN BUDGET), and decode-only top4dm16 lands at +0.97 [−0.44, +2.58], WITHIN SHARE with chat at 30/40. The rule
fixes **k4_form = decode-only, k₂ = 16** and **training_k = 2**: k = 4 at decode is free, and part 2 trains at k = 2.

## The run

- **Panels:** 28 of 28 COMPLETE, in one launch through `s1_autoresume.ps1` (01-11 13:46 to 21:47); no VOID, stderr
  empty.
- **MASS lines:**
  - at K = 8 the mean factor is 1.000000 on every panel;
  - at K = 16 it is 0.70-0.72 for top4m16 and 0.66-0.69 for rnd4m16;
  - top4dm16's means include the prompt rows, whose factor is 1 by construction: 0.78-0.97.

## Arms (Δ = points − ref's 55.19; the same 10,000 paired draws as part 1)

| Arm | Δ points [CI95] | Label | Chat /40 | Degen. | Fits | Mass: Δ − Δ(base) [CI95] | Top-1 | KL top-32 | dBPB [CI95] |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| top4 (part 1) | −7.49 [−10.63, −4.38] | UNDECIDED | 25 | 0 | no | — | 0.794 | 0.408 | +0.079 |
| top4m8 | −2.91 [−5.45, −0.46] | UNDECIDED | 26 | 0 | no | +4.58 [+1.29, +7.86], helps | 0.862 | 0.185 | +0.007 [−0.002, +0.015] |
| top4m16 | −1.74 [−4.64, +1.22] | WITHIN BUDGET | 30 | 0 | budget, not share | +5.76 [+1.95, +9.49], helps | 0.797 | 0.417 | −0.012 [−0.030, +0.005] |
| rnd4m16 | −39.74 [−43.87, −35.40] | OVER BUDGET | 10 | 28 | no | — | 0.550 | 2.044 | +0.448 |
| top4d (part 1) | −4.83 [−7.53, −2.17] | UNDECIDED | 22 | 0 | no (chat) | — | (copied) | (copied) | (copied) |
| top4dm16 | **+0.97 [−0.44, +2.58]** | **WITHIN SHARE** | 30 | 0 | **share** | +5.81 [+2.86, +8.81] vs top4d, helps | (copied) | (copied) | (copied) |

- **Control:** top4m16 beats rnd4m16 by 38.0 points [+33.9, +41.8]: the selection is not chance.
- **Base of the mass comparisons:** top4 for top4m16 and top4m8, top4d for top4dm16.
- **The decode-only arm's score panels** (arc, mmlu, documents) are ref's outputs by construction, as in part 1. Its Δ
  comes from generation alone: nq −1.5, tqa −1.7, gsm8k +4.0, chat +5.0 points.

**Per-panel accuracy (%)**:

| | nq | tqa | gsm8k | arc | mmlu | chat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ref | 29.0 | 48.7 | 81 | 53.2 | 49.3 | 70.0 |
| top4 | 22.7 | 40.0 | 66 | 48.2 | 46.8 | 62.5 |
| top4m8 | 24.8 | 44.7 | 80 | 52.2 | 47.0 | 65.0 |
| top4m16 | 22.8 | 41.8 | 80 | 53.3 | 47.7 | 75.0 |
| top4dm16 | 27.5 | 47.0 | 85 | 53.2 | 49.3 | 75.0 |

## Predictions against outcome

| Prediction | Outcome |
| --- | --- |
| top4m8 −4.5 to −6.5 | **miss, better**: −2.91 |
| top4m16 −1.5 to −4.5, WITHIN BUDGET or UNDECIDED, not WITHIN SHARE | **hit**: −1.74, WITHIN BUDGET |
| rnd4m16 below −20 | **hit**: −39.74 |
| top4dm16 −1 to −3, passes the chat gate | **miss, better**: +0.97; **hit** on chat (30/40) |

Read here, not registered:
- **Likelihood and accuracy disagree on which k₂ is better.** top4m8 is closer to ref on the documents (top-1 0.862,
  KL 0.185), but top4m16 scores more points (−1.74 against −2.91). Its document top-1 (0.797) and KL (0.417) barely
  move from top4's. The paper reports the same (perplexity picks a worse k₂ than MMLU), so likelihood does not choose
  k₂ here.
- **What k₂ = 16 recovers sits in generation:** gsm8k from 66 to 80, chat from 62.5 to 75. nq and tqa barely move
  (22.7 to 22.8, 40.0 to 41.8).

## What the registered rule fixes

- **k4_form = decode-only, k₂ = 16 (top4dm16).** It is part 1's branch "only top4d fits the share": k = 4 is free at
  decode, and stage 2 carries decode-only k = 4 with k₂ = 16.
- **training_k = 2, at decode.** Part 2 tries the lowest k of the series (6, 4, 2) that breaks the budget without
  training.
- **start = top4m16.** The mass helps, and top4m16 is the mass arm with the higher Δ.
- **The failure criterion of (a) is not met.** Four experts fit, at decode, without training.

## Thesis line (desk model of stage 0: 32 GB/s plus 4.24 ms per token, context 2048, all-ternary flash at the RAM wall)

| k at decode | MB read per token | Ceiling, tok/s | On mini |
| --- | ---: | ---: | --- |
| 8 | 1,531 | 19.2 | ref |
| 4 | 1,141 | 25.1 | fits the share untrained (top4dm16) |
| 2 | 946 | 29.6 | part 2's question (untrained top2, k₂ = k₁: −40.1) |

- **Decode-only is what the success threshold measures.** The threshold is decode tok/s at context 2048. The prompt
  stays at k = 8 and costs prefill time, not decode speed.
- **The margin at k = 4 is thin:** 25.1 against 24.7, which is 1.6%. k = 2 would leave 20%.
- **Scale:** Ling-mini. The transfer to flash (same router: 256 experts, top-8, 8 groups, sigmoid, scaling 2.5) is a
  stage 2 check.

## Next

Part 2 (`S1C_…`) trains k = 2 at decode, starting from k₂ = 16. First it measures top2dm16 untrained: the mass was
never tried at k = 2, and it recovered ~6 points at k = 4.
