# E55 — THE DEMONSTRATION TABLE, WITH AN INTERVAL THAT SURVIVES

**Pre-registered 2026-09-13, pushed before a single cell runs.**

---

## 1. What this is for

The programme's goal is a **10B model at 50 tok/s (good) / 100 tok/s (excellent)**. After
fifty-four experiments the record still does not contain **one table** that says *"the 10B shape
runs at X tok/s at context C, with an interval, measured this way, and here is exactly what is
and is not real about it."* It contains a headline (`110.62`), a slope, a `C50`, and a large
number of caveats scattered across briefs.

Worse, E54 §B.5 showed that **every `±` in this programme is a lower bound**: they are all
small-`k` min–max ranges, and a range grows with sample size without bound. E44's headline
`spread 2.3%` is a 5-rep range; on the one cell type where 58 reps exist, a 5-rep range reads
3.85% where the truth is 11.57%.

So the old intervals cannot simply be re-quoted, and the demonstration has to be re-measured
with a statistic that holds still.

> **E55 produces the table, with intervals that do not move when you take more samples, using
> every design repair the last three experiments paid for.**

## 2. What is real and what is not — stated once, at the top, and repeated in the output

| | |
|---|---|
| **shape** | **REAL.** `e40_r128.bin`, 5.11 GB, the 10B target shape: `E = 256` experts, `k = 3` active = **1.17% of the FFN**, GQA, the arm every speed number in this programme is quoted on. |
| **weights** | **SYNTHETIC.** These are not a trained model's parameters. E45 measured that rate does not depend on weight *values* (bounded under 5%, no reproducible direction) — *supported, not certified*. |
| **quality at this shape** | **NOT DEMONSTRATED.** E37's verdict on post-hoc conversion of real weights was **NO**. The trained-carve attempt is H1 and is running on the user's T4. |
| **therefore the claim** | **a speed claim about the shape, not a working-model claim.** Anyone reading the table must be able to see that from the table itself. |

**This brief does not close the goal and does not claim to.** It closes the speed half of it, to
a standard the quality half can later be attached to.

## 3. Every repair the last three experiments bought, applied

1. **A discarded warm-up cell before every recorded one** — E54 B.4: `W0` read 13.65% foreign on
   a 2.10% box because the run's own launch (`nohup`, interpreter, shell) sits inside its first
   cells.
2. **Window order interleaved, not ascending** — E54 §2: E53's ascending ladder made "time
   elapsed" and "heavy work just finished" the same observation. Here every repetition visits the
   windows in a rotated order, so no window sits systematically behind heavier work.
3. **`k ≥ 15` per window and a QUANTILE interval, never min–max** — E54 B.5: the interquartile
   width converges (2.10 / 2.25 / 2.29% at `k` = 20 / 40 / 58) where the range never does.
4. **Per-cell occupancy, timestamp and CPU clock** — E53 C.0 and E54 §3.
5. **The attention kernel and the exponential arm printed in the output** — E50's law, after the
   fast kernel hid on a default for twenty-two experiments.

## 4. The design

`donor_engine_e53.exe --weights e40_r128.bin --carve-k 3 --threads 6 --fexp libm`, the control
arm. (`--fexp poly` is faster by `G-E53f` but is not promoted, so the demonstration is quoted on
the incumbent — the conservative choice.)

| window `N` | mean position | reps |
|---|---|---|
| 160 | 80 | 40 |
| 640 | 320 | 25 |
| 1280 | 640 | 25 |
| 2560 | 1280 | 15 |

Reps are allocated by cost, not uniformly: `k` is chosen so each window takes roughly the same
wall time, and every `k` is at or above where E54's resampling shows the interquartile width has
converged. Estimated total **~25 minutes**, derived from measured per-cell wall times
(`3.1 s` at N=160, `17.8 s` at N=1280) and the fitted slope for N=2560.

## 5. The gates

### `G-E55a` — the interval must be STABLE, or it is not an interval
For each window, compute the interquartile width on **all `k` reps** and on a **random half**.

> **PASS** if for every window the two agree within **20% of the full-sample width**.
> **UNSTABLE** otherwise, and that window's interval is reported as a lower bound and excluded
> from `G-E55b`.

This is the gate E44 never had, and it is the one E54 proved was missing. 20% is not a tolerance
I invented for convenience: E54's resampling puts the `k`=20 → `k`=40 IQR change at **7%**, so
20% is loose enough that a converged statistic passes comfortably and tight enough that the
min–max statistic, which changes by **30%** over the same step, would **fail**.

### `G-E55b` — the demonstration table
For each window report **median** and the **25th–75th percentile band** in tok/s. Then:

> **`TARGET HELD`** at a window if the **25th percentile** — not the median — is at or above
> 50 tok/s. The pessimistic end of the band carries the claim, so the claim is not an average
> that half the runs miss.
>
> **`EXCELLENT HELD`** at a window if the 25th percentile is at or above 100 tok/s.

### `G-E55c` — how far does it hold?
Fit `ms/tok = a + b·pos` to the **medians**, and separately to the **25th percentiles**, and
report the context at which each crosses 20 ms/tok (= 50 tok/s).

> Reported as a **range between the two fits**, never as a single `C50`. E53 C.5 found the
> control arm's own slope moving 10.6% between sessions; a four-figure `C50` was never
> defensible and this replaces it with a band.
>
> **REFUSED** if any window feeding the fit failed `G-E55a`, or if fewer than three windows
> survive.

### `G-E55d` — occupancy, unchanged
`OCC_BAR = 4.39` per cell, recorded, breaches counted and reported by window. Cells above the
bar are **kept in the record and excluded from the quantiles**, and both versions are printed —
E54 B.6 observed heavy cells reporting double the foreign occupancy of light ones on a quiet
box, so silently dropping them could bias exactly the long-context end.

## 6. Predictions, before the data

| | prediction |
|---|---|
| `G-E55a` | **PASS on all four windows.** `k` ≥ 15 is past where E54 showed IQR converging. |
| interquartile width | **2–4%** at every window — E54's 58-cell IQR was 2.29% |
| `G-E55b` at N=160 | `EXCELLENT HELD` — tonight's cells read ~117 |
| `G-E55b` at N=1280 | `TARGET HELD`, not excellent — tonight's read 76.7–82.6 |
| `G-E55b` at N=2560 | **`TARGET HELD`**, 25th percentile between 52 and 60 tok/s |
| `G-E55c` | the two crossings land **1300–1900**, and the band between them is **wider than 15%** |

The last one is the one I most expect to be wrong, and it is the one worth being wrong about:
if the median fit and the 25th-percentile fit disagree by less than 15%, then `C50` was more
robust than E53 C.5 made it look.

## 7. What this does not do

It does not use real weights, does not demonstrate quality, does not promote `--fexp poly`, and
does not close the goal. It makes the speed half of the goal **quotable without a footnote**,
which nothing in the record currently is.
