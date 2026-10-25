# C50-T2 stitching cell result: FAIL; no v0 layer stitches linearly into F0's experts

26 October 2026; the cell ran on 25 October, 22:24-23:59. Protocol: [C50_T2_STITCH_PROTOCOL_20261025](C50_T2_STITCH_PROTOCOL_20261025.md) (`aff1fbb`,
pushed before the cell's dumps; date corrected in `c4651d9`).
- **Measure record:** `results/c50_t2_stitch.json` (sha256 `6ec221fbe923472f…`), from `c50_t2_stitch.py`.
- **Verdict record:** `results/c50_t2_stitch_verdict.json` (`dcf25d0263881c0f…`), from `c50_t2_stitch_verdict.py`,
  which reads the measure record only.
- **Post-hoc, not registered:** `results/c50_t2_stitch_posthoc.json` (`430481f35b273313…`).

## Verdict

**FAIL: none of v0's 19 MoE layers stitches.** Every check and planted control passed, so the verdict stands.

**What the rule does with it:** linear stitching does not support option X. The choice returns with the numbers.
- **What is closed:** a ridge map from v0's MoE input onto F0's.
- **What is not closed:** option X as such. A stitch trained on what F0's experts output is a different, untested
  map.

| v0 layer l | F0 layer j (chosen on select R²) | test R² | top-8 overlap / bar | routed-output cosine / bar |
| ---: | ---: | ---: | --- | --- |
| 1 | 1 | 0.470 | 0.670 / 0.740 | 0.794 / 0.926 |
| 2 | 2 | 0.352 | 0.650 / 0.732 | 0.743 / 0.948 |
| 3 | 2 | 0.318 | 0.601 / 0.732 | 0.610 / 0.948 |
| 4 | 2 | 0.284 | 0.572 / 0.732 | 0.563 / 0.948 |
| 6 | 2 | 0.245 | 0.524 / 0.732 | 0.481 / 0.948 |
| 10 | 2 | 0.228 | 0.498 / 0.732 | 0.446 / 0.948 |
| 14 | 2 | 0.217 | 0.517 / 0.732 | 0.464 / 0.948 |
| 18 | 2 | 0.180 | 0.483 / 0.732 | 0.398 / 0.948 |
| 19 | 31 | 0.187 | 0.612 / 0.729 | 0.557 / 0.810 |
| all 19 | 17 on F0 layer 2 | median 0.227 | median 0.513; range 0.483-0.670 | range 0.398-0.794 |

The bar is F0's own state at noise R² 0.5, read by the same F0 block: 0.729-0.740 on overlap, 0.810-0.948 on cosine.
The full per-layer table is in the record.

**What the numbers say:**
- **v0's state explains little of F0's linearly:** R² 0.47 at the first layer, about 0.2 in the middle, never
  above 0.47 at any of the 589 pairs.
- **The routing carries real information:**
  - F0's router, given the mapped state, picks about half of the experts it picks on its own state: overlap 0.48-0.67;
  - the shuffled-pair maps reach 0.05-0.08, which is chance level;
  - a learned map that recovers half the routing is not nothing.
- **The routed output does not follow:** the cosine is 0.40-0.79 against bars of 0.81-0.95.
- **F0's experts are robust to isotropic noise and not to this error.** At noise R² 0.3 they still read an overlap of
  0.68-0.71 and a cosine of 0.75-0.93. The map's error, at a similar or lower R², costs much more. It sits in the
  directions F0's router and experts use.

## Checks and planted controls

| Check | Result |
| --- | --- |
| Tool: numpy F0 block vs llama.cpp on F0's own states, on every chosen j (1, 2, 31) | top-8 overlap 1.000; median cosine 0.99995, 0.99993, 0.99989 (needs ≥ 0.97, ≥ 0.99) |
| Planted positive: F0 → F0 through the same ridge pipeline | R² 0.99999+, overlap 0.999-1.000, cosine 1.000: stitches on all three j |
| Planted negative: maps fitted on shuffled pairs | overlap 0.046-0.076, cosine −0.13..0.25: stitches on no layer |
| Verdict self-test | 8 of 8 |
| Inputs | `main.ids` sha256 equal to the manifest; both dumps from the same ids (FNV-1a `0602933d…`, 36,864 positions) |

## The run, with its two incidents

| Step | Wall | Note |
| --- | --- | --- |
| v0 dump, 128 documents | 22:24-22:30 | `c50_llama_dump_avx2.exe` registered build `b961d7b2…` |
| F0 dump, first | 22:30-22:56 | complete, but its `run.json` was not valid JSON (below) |
| Measure, first | 22:57-22:58 | **VOID before any number** (§12): the tool wrote the model path into `run.json` without escaping, and F0's path came with backslashes |
| Fix | | `c50_llama_dump.cpp` escapes its JSON strings; rebuilt sha256 `9ae59cba…`. Only the JSON writer changed: on the dev documents the rebuilt tool's `ffn_norm-1` is byte-identical to the old build's |
| F0 dump, redone | 23:04-23:29 | rebuilt tool, folder `t2/main/f0r`; `ffn_norm-16` byte-identical to the first dump |
| Measure | 23:30-23:46 | 909 s |
| Post-hoc | 23:48-23:59 | 627 s |

**Before the protocol:**
- the first F0 dev dump hung on a stalled read of the USB drive E:;
- the second was stopped by the session host for low memory.

F0 then ran from a staged SSD copy (sha256 `5348c73f…`, checked), and the session was restarted with the host's
memory reaper off.

## Registered predictions, scored

| Prediction | Measured | Score |
| --- | --- | --- |
| Verdict PARTIAL | FAIL | **WRONG** |
| Chosen j within ±4 of l · 31/19 on at least 14 layers | 4 of 19 (l = 1, 2, 3, 19) | **WRONG** |
| Test R², median 0.50-0.80 | 0.227 | **WRONG** |
| Mapped top-8 overlap, median 0.40-0.65 | 0.513 | **RIGHT** |
| Stitching layers mostly at l ≤ 13 | none stitch | **WRONG** |
| Shuffled maps, overlap ≤ 0.15 on every layer | 0.046-0.076 | **RIGHT** |
| Walls: v0 ≤ 30 min, F0 ≤ 1.5 h, measure ≤ 1.5 h | 6 min; 26 and 25 min; 15 min | **RIGHT** |

**Three right, four wrong.** The wrong ones all expected a linear map to carry much more of F0's state than it does.

## Post-hoc, not registered: depth-matched pairs

**Why it was run:** the registered pairing put 17 of 19 layers on F0 layer 2. Select R² favours the most predictable
target, F0's early states, over the matching depth. So the registered numbers at l = 3..18 read F0 layer 2's experts.
This reads each l at j = round(l · 31/19) instead, with that pair's best λ, the same ridge, rows and block, and its
own noise-0.5 bar.

| v0 layer l → F0 j | test R² | overlap / bar | cosine / bar |
| --- | ---: | --- | --- |
| 1 → 2 | 0.406 | 0.691 / 0.729 | 0.793 / 0.946 |
| 2 → 3 | 0.297 | 0.543 / 0.703 | 0.476 / 0.858 |
| 4 → 7 | 0.242 | 0.530 / 0.704 | 0.242 / 0.780 |
| 7 → 11 | 0.206 | 0.479 / 0.687 | 0.209 / 0.754 |
| 10 → 16 | 0.185 | 0.463 / 0.640 | 0.213 / 0.683 |
| 13 → 21 | 0.175 | 0.473 / 0.629 | 0.201 / 0.658 |
| 16 → 26 | 0.148 | 0.438 / 0.627 | 0.145 / 0.651 |
| 18 → 29 | 0.139 | 0.481 / 0.643 | 0.210 / 0.670 |
| 19 → 31 | 0.187 | 0.612 / 0.731 | 0.557 / 0.814 |

**Matched depth is worse, not better.** None of the 19 stitches. In the middle layers the routed-output cosine is
0.15-0.28, against bars of 0.65-0.78, while the overlap stays at 0.43-0.53. F0's middle experts are the ones a linear
map from v0 reaches least.

## What this means for the thesis

- **The thesis is capacity from the 100B, read at nearly v0's bytes.** Option X keeps the 100B's knowledge in its
  own weights, and that rests on v0 being able to address those weights.
- **A linear map addresses the routing partly (about half the experts) and the computation poorly**, worst at
  matched depth.
- **What stays open, in cost order:**
  1. **A stitch trained on the experts' output, locally.** Train P_in to make F0's block on the mapped state
     match F0's block on its own state (the cosine above as the loss, with the routing straight through), on the
     dumps already made.
     - One F0 layer's experts at FP16 are about 6.4 GB and fit the local 12 GB GPU, so no T4 time is needed.
     - It answers directly whether any learned map lets F0's experts read v0, before the gate and P_out are built.
  2. **Option N:** new experts trained from scratch, with F0 as teacher. The capacity is small (~0.3-0.5 B
     parameters by training memory), and the knowledge comes only through the KL.
  3. **Option A′:** arm (a) made additive. It is a LoRA on the experts again.

## What this cannot say

- **A linear map, one draw of documents.** A trained or non-linear stitch can do better; how much better is what
  item 1 above would measure.
- **The pairing criterion was a weak choice.** Select R² prefers the easiest target over the matching depth. The
  post-hoc covers matched depth, and it is worse, so the verdict does not hinge on it.
- **The noise bar is strict** (recorded in the protocol). Even at noise R² 0.3, F0's reading is better than the map's
  everywhere. A looser bar would not have passed the middle layers either: at matched depth their cosine is below
  0.3.
