# C50-T2 trained stitch cell result: STOP, no verdict; the planted positive missed the bar at two deep layers

26 October 2026. Protocol: [C50_T2_TSTITCH_PROTOCOL_20261026](C50_T2_TSTITCH_PROTOCOL_20261026.md) (`71dbd11`, pushed
before the run).
- **Measure record:** `results/c50_t2_tstitch.json` (sha256 `8b184c099d5dbf8b…`), from `c50_t2_tstitch.py`. It ran
  in 4.46 h on the local GPU, with exit 0 and no divergence.
- **Verdict record:** `results/c50_t2_tstitch_verdict.json` (`5b91537eeb706693…`), from
  `c50_t2_tstitch_verdict.py`, which reads the measure record only.

## Verdict

**STOP: no verdict.** The planted positive must stitch on every layer, and it missed the bar at two:

| v0 layer → F0 layer | Positive, overlap / cosine | Bar (noise 0.5) |
| --- | --- | --- |
| 15 → 24 | 0.676 / 0.661 | 0.632 / 0.675 |
| 16 → 26 | 0.674 / 0.633 | 0.628 / 0.652 |

- **On the other 17 layers the positive stitched,** but its cosine margin thins with depth: 0.009-0.014 at F0
  layers 23, 28 and 29.
- **The calibration covered F0 layers 2, 16 and 31 only.** There the margins were 0.02-0.04, and in the cell they
  were 0.026, 0.064 and 0.049.
- **The positive's select cosine was still rising at its last epoch on all 19 layers.** It is a budget-limited
  control in the deepest layers.
- **Per the rule, the cell's verdict does not stand.** The line stops for review. What follows reads the numbers
  without a verdict.

## The numbers (no verdict)

All other checks passed:

| Check | Result |
| --- | --- |
| Tool: bf16 block vs llama.cpp on F0's own test states, 19 layers | top-8 overlap ≥ 0.99998; median cosine ≥ 0.99978 |
| Ridge arm vs the linear post-hoc record | same λ on all 19; overlap and cosine within 0.0001 |
| Planted negative: shuffled maps, trained and selected like the measured ones | stitch on no layer; overlap ≤ 0.089, cosine ≤ 0.365 |
| Verdict self-test | 11 of 11 |

**Measured maps on the test rows, against the bar:**

| v0 l → F0 j | Arm | Ridge overlap / cosine | Trained overlap / cosine | Bar | Cosine short of the bar |
| ---: | --- | --- | --- | --- | ---: |
| 1 → 2 | linear | 0.691 / 0.793 | 0.670 / 0.851 | 0.734 / 0.948 | 0.097 |
| 2 → 3 | linear | 0.543 / 0.476 | 0.574 / 0.678 | 0.702 / 0.860 | 0.182 |
| 3 → 5 | mlp | 0.514 / 0.346 | 0.535 / 0.542 | 0.686 / 0.790 | 0.248 |
| 4 → 7 | mlp | 0.530 / 0.242 | 0.537 / 0.455 | 0.707 / 0.778 | 0.323 |
| 7 → 11 | mlp | 0.479 / 0.209 | 0.487 / 0.347 | 0.683 / 0.749 | 0.402 |
| 10 → 16 | linear | 0.463 / 0.213 | 0.479 / 0.327 | 0.643 / 0.685 | 0.358 |
| 13 → 21 | mlp | 0.473 / 0.201 | 0.486 / 0.321 | 0.634 / 0.664 | 0.344 |
| 15 → 24 | linear | 0.445 / 0.196 | 0.469 / 0.291 | 0.632 / 0.675 | 0.384 |
| 16 → 26 | mlp | 0.438 / 0.145 | 0.455 / 0.246 | 0.628 / 0.652 | 0.406 |
| 18 → 29 | linear | 0.481 / 0.210 | 0.505 / 0.329 | 0.644 / 0.671 | 0.342 |
| 19 → 31 | linear | 0.612 / 0.557 | 0.644 / 0.665 | 0.725 / 0.810 | 0.145 |

All 19 layers are in the record.

**What the numbers say:**
- **Training on the experts' output helps, a lot less than needed.**
  - The test cosine rises by 0.06-0.21 over the ridge (median 0.12). That closes 20-53% of the gap to the bar.
  - The cosine stays 0.10-0.41 short of the bar, and in the middle layers (l = 4..16) it is ≤ 0.48 on every one.
- **The routing does not move.** The top-8 overlap changes by −0.02..+0.03 over the ridge. Training buys output
  direction from the experts already chosen, and not the right experts.
- **The affine and MLP arms are the same.** Their select cosines differ by ≤ 0.005 on every layer, and the MLP is
  chosen on 11 of 19.
- **The measured maps overfit early:**
  - their best epoch is 2-8 on 18 layers (15 on one);
  - the fit-minus-test cosine is 0.11-0.47.
- **More data helps, slowly.** Doubling the fit documents (half → full) raised the linear arm's test cosine by
  0.033-0.078 (median 0.062). That is above the protocol's 0.05 threshold for DATA-LIMITED.
  - At that rate, the middle layers' 0.30-0.41 shortfall needs five to seven more doublings, i.e. 32-128× the
    documents.
  - That is a linear extrapolation on two points, and gains per doubling usually shrink.

**What the STOP leaves unread:**
- The two layers where the positive missed (l = 15, 16) are also two of the measured maps' worst: cosine 0.291 and
  0.246 against the positive's 0.661 and 0.633.
- This reading is post-hoc and not registered: no verdict comes from it. With the two layers left out, none of the
  other 17 stitches either.

## Registered predictions, scored

| Prediction | Measured | Score |
| --- | --- | --- |
| Verdict FAIL, not data-limited | STOP; the data gain is 0.062, which would have read DATA-LIMITED | **WRONG** |
| MLP chosen on ≥ 12 of 19 | 11 | **WRONG** |
| Test cosine gain over the ridge, median 0.05-0.20 | 0.120 | **RIGHT** |
| Middle layers (l = 4..16): test cosine ≤ 0.50 on every one | max 0.479 | **RIGHT** |
| Best candidates l = 1 and 19 | the two smallest shortfalls (0.097, 0.145); none stitches | **RIGHT** |
| Fit-minus-test ≥ 0.10 on most layers | 0.11-0.47 on all 19 | **RIGHT** |
| Data gain, median 0.02-0.05 | 0.062 | **WRONG** |
| The positive stitches on all 19; no shuffled map does, cosine ≤ 0.55 | positive on 17; shuffled on none, ≤ 0.365 | **WRONG** (half right) |
| Wall ≤ 6 h | 4.46 h | **RIGHT** |

**Five right, four wrong.**
- The wrong ones are the control's budget, the data gain and the arm count.
- None of them is the size of the effect.

## What this means for the thesis

**The thesis is capacity from the 100B, read at nearly v0's bytes.** Option X keeps the knowledge in F0's own
weights, and that rests on v0 addressing them.
- **Two cells tried two kinds of map.** Under linear stitching (FAIL) and trained stitching (STOP, with the same
  picture), F0's experts read about half the right routing from v0's state, and a computation far from their own.
- **Training moves the output, not the routing.** The overlap stays at 0.45-0.67 while F0's own state at noise 0.5
  keeps 0.62-0.73.
- **The rule on frozen organs applies:** after two cells on one local lever without a pass, it is frozen with its
  state recorded, and the pipeline moves on.
- **The state recorded for X:**
  - no stitch on any layer;
  - the trained map closes 20-53% of the cosine gap;
  - the shortfall is 0.10-0.41;
  - the cell is data-limited, at a rate that needs 32-128× the documents for the middle layers;
  - the positive is budget-limited in F0's deepest layers.

**The choice returns:**
1. **Freeze X, go to option N:** new experts trained with F0 as teacher. The knowledge then travels through the
   distillation, not through F0's weights.
2. **Another cell for X with more data:** a new F0 dump on several times the documents, on a few layers only, with
   the positive given a longer budget. It costs local machine time only. The data-gain rate predicts a FAIL.
3. **Option A′.**

## What this cannot say

- **No verdict.** The planted positive failed on two layers, so the rule gives none. Everything above is a reading
  of the numbers.
- **Two families of map, one budget, one seed, one draw of documents.** A much larger dataset or a different family
  of map (a deep non-linear one, or one fed by several of v0's layers) is untested.
- **The bar is F0's own state at noise R² 0.5, as in the linear cell.** It asks for F0's experts to compute what
  they compute in F0. Whether a branch below that bar still adds knowledge to v0 is a different question, and only
  the training protocol (v0's CE and fact panels) measures it.
