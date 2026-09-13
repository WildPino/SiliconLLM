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

---

# ADDENDUM A — THE TABLE EXISTS; MY STABILITY GATE WAS A COIN FLIP; THE OCCUPANCY BAR COSTS 40% OF THE SAMPLES AND BUYS NOTHING

Run: **105 cells + 1 discarded warm-up, 25.4 minutes**, box measured at **1.20% foreign**
immediately before launch. `CONFIG attn=avx4 attnr=none fexp=libm mvacc=4 threads=6
quant=ternary`, printed by the engine on every cell.

## A.1 The table

**SHAPE REAL — WEIGHTS SYNTHETIC — QUALITY NOT DEMONSTRATED AT THIS SHAPE.** A speed claim about
the shape, not a working-model claim.

Clean cells (those at or under `OCC_BAR`), which is the registered analysis:

| `N` | mean pos | cells | **p25** | median | p75 | IQR | min–max |
|---|---|---|---|---|---|---|---|
| 160 | 80 | 25 | **118.47** | 119.57 | 121.27 | 2.34% | 9.10% |
| 640 | 320 | 16 | **98.77** | 99.34 | 100.20 | 1.43% | 4.66% |
| 1280 | 640 | 17 | **79.14** | 80.89 | 81.35 | 2.73% | 7.50% |
| 2560 | 1280 | 8 | **49.57** | 51.98 | 54.43 | 9.34% | 11.68% |

All cells, as the counterfactual (E52's law: a refusal must not launder a number):

| `N` | mean pos | cells | p25 | median | IQR |
|---|---|---|---|---|---|
| 160 | 80 | 40 | 118.09 | 119.09 | 2.43% |
| 640 | 320 | 25 | 98.37 | 99.52 | 2.01% |
| 1280 | 640 | 25 | 79.40 | 80.53 | 2.38% |
| 2560 | 1280 | 15 | 50.71 | 53.32 | 7.91% |

**The two analyses agree to better than 2.5% everywhere**, so the numbers themselves are not in
question. What is in question is every PASS/FAIL label attached to them, for the reason in A.2.

**Read against the goal, and this is the part that was missing from the record:**

* **100 tok/s (excellent) holds to roughly mean context 320.** p25 is 98.4–98.8 there — within
  1.5% of the line, on both analyses.
* **50 tok/s (good) holds to roughly mean context 1280.** p25 is 49.6–50.7 there — straddling
  the line.

That is more informative than `C50 = 1443` ever was, and it comes from 105 cells with quantiles
rather than 5 cells with a range.

## A.2 `G-E55a` is a coin flip, and that is this run's real finding

`G-E55a` compares the full-sample interquartile width against the width of **one random half**.
Re-drawing that half 2000 times per window, on exactly the cells the registered analysis used:

| `N` | `k` | P(the gate says PASS) | IQR(full) | median \|half−full\|/full |
|---|---|---|---|---|
| 160 | 25 | **60%** | 2.34% | 15% |
| 640 | 16 | **48%** | 1.43% | 21% |
| 1280 | 17 | **41%** | 2.73% | 27% |
| 2560 | 8 | **38%** | 9.34% | 46% |

**A gate whose verdict is a coin flip at the `k` it runs at is not a gate.** The proof is in the
run itself: swapping only the random seed — the clean-cell analysis versus the all-cell one —
flips *which* windows pass. Registered analysis: 160 and 640 pass. Counterfactual: 160 and 1280
pass. Same engine, same cells, different draw.

**I built an instrument to check whether an interval was stable, and the instrument was less
stable than the interval.** E44's `G-E44b` measured the engine as part of the load it was
supposed to exclude; this is the same family — *the gate's own sampling noise was never
measured, and it is larger than the quantity it gates*.

### What this does and does not invalidate

* **The medians, quantiles and IQRs stand.** They are 105 measured cells and two independent
  analyses agree within 2.5%.
* **`G-E55b`'s `EXCELLENT HELD` / `TARGET HELD` labels do not stand**, because my runner gates
  them on `G-E55a`. They are withdrawn as *verdicts* and the underlying p25 values are reported
  as data instead. As it happens the substance survives: p25 at mean pos 80 clears 100 by 18%
  under either analysis, and p25 at mean pos 320 clears 50 by 97%.
* **`G-E55c` REFUSED** and stays refused. No `C50`, no band, and §6's 1300–1900 prediction is
  not scored.
* **Run 1 is NOT restated with a repaired gate** (E36 run-2 rule).

### `G-E55a2`, registered for run 2

> Replace the single half-split with the **median of `|half − full| / full` over 200 random
> splits**, against the same 20% bar.

This is a **variance reduction on the same statistic**, not a relaxation — and it is not
automatically conservative, since one draw produces false PASSes as readily as false failures.
It is therefore registered as post-hoc, and it may only judge **fresh** cells.

## A.3 `OCC_BAR = 4.39` costs 40% of the samples and buys nothing measurable

On a box measured at **1.20% foreign before launch**, foreign occupancy *during* cells ran:

    N=160    median 4.08%    N=640   median 3.90%
    N=1280   median 3.96%    N=2560  median 4.10%      breaches: 38% / 36% / 32% / 47%

**The median cell sits within 0.3 points of the bar.** The bar is not separating a quiet box
from a busy one; it is sitting on top of the idle floor *as measured while the engine runs*.

And the excluded cells are not slower:

    N=160   breached vs clean median   -0.82%
    N=640                              +0.18%
    N=1280                             -0.95%
    N=2560                             +3.95%

**Contention would make them slower. They are indistinguishable, and at the longest window they
are faster.** So the exclusion removes ~40% of the samples at random with respect to rate: it is
**pure power loss with no bias correction**, and it is why `N=2560` had only 8 usable cells and
could never have passed a stability check.

**The defect is in how the bar's zero point was derived.** E52 set `OCC_BAR = 4.39` as *the
largest foreign occupancy whose predicted rate cost is at most half the zero-load dispersion* —
treating 0 as the idle value. But `foreign = system − child` also counts kernel work done **on
behalf of** the engine (page faults, zeroing, TLB shootdowns), which `GetProcessTimes` does not
attribute to the child. With the engine running, the idle floor of that quantity is **~4.0%, not
0**. The bar was therefore set essentially *at* the floor.

E54 B.6 raised this as a hypothesis from two blocks. Four windows and 105 cells now support it,
and the "breached cells are not slower" check is the part that turns it from suspicion into a
measurement. **It is registered as E56 and is not repaired here**, because relaxing a bar after
seeing which cells it rejected is the move this programme exists to prevent.

## A.4 Design errors of mine, recorded

1. **I allocated reps by cost without accounting for the exclusion rate.** `N=2560` was planned
   at 15 and delivered 8 usable. Reps must be planned for the **post-exclusion** count.
2. **`G-E55a`'s own sampling noise was never measured before it was registered** — the whole
   subject of the experiment, applied to every statistic except the gate's.

## A.5 Scorecard

| §6 prediction | measured | |
|---|---|---|
| `G-E55a` PASS on all four windows | 2 of 4, and the 2 are seed-dependent | **WRONG** |
| IQR 2–4% at every window | 2.34 / 1.43 / 2.73 / 9.34% | **WRONG** |
| `N=160` EXCELLENT | p25 118.47, clears 100 by 18% | **right** |
| `N=1280` TARGET not excellent | p25 79.14 | consistent, **not scored** (label withdrawn) |
| `N=2560` p25 between 52 and 60 | **49.57** (clean) / 50.71 (all) | **WRONG**, and below 50 |
| `G-E55c` crossings 1300–1900, band > 15% | REFUSED | **not scored** |

**1 right, 3 wrong, 2 unscored.** The pattern from E53 and E54 repeats exactly: every direction
right, every magnitude wrong. The `N=2560` miss is the useful one — I predicted the target would
hold comfortably at mean context 1280 and it lands **on** the line.

## A.6 What is owed

1. **E56: re-derive `OCC_BAR`'s zero point** against the foreign-occupancy floor measured *with
   the engine running*, and decompose `system − child` into contention and
   kernel-on-behalf-of-the-child. Until then the bar rejects 40% of every run for nothing.
2. **E55 run 2**: `N=1280` and `N=2560` only, reps planned post-exclusion, judged by `G-E55a2`.
   The two short windows are **not** re-measured.
3. `G-E55c` needs three stable windows and currently has none it can trust.
4. **Unchanged and still the real blocker**: the weights are synthetic and quality at this shape
   is undemonstrated. Speed is not what is standing between this programme and its goal.
