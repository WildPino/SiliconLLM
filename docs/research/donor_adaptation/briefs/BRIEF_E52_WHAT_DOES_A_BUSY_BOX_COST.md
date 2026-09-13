# BRIEF E52 — WHAT DOES A BUSY BOX ACTUALLY COST, AND WHERE SHOULD THE BAR BE?

**Registered before any E52 cell exists.** This experiment measures the instrument, not the
engine. Nothing in it may be quoted as a speed result for any arm.

## 1. Why this, and why now

E44 addendum C closed with a gate that works and a bar that does not. Under `G-E44b2`:

| run | guard | foreign during reps | median rate |
|---|---|---|---|
| 1 | 10.4% | ~12.6% (implied) | 102.90 tok/s |
| 2 | 2.2% | 3.8% | **110.62 tok/s** |

**Both pass the 15.0% bar and they are 7.5% apart.** So the bar does not deliver what a bar is
for — that two readings which pass are comparable. `OCC_BAR = 15.0` was chosen from the range
E43 happened to observe; it has never been measured against what it buys.

That matters now rather than in general, because the next experiment on the critical path needs
it. E51's `G-E51d` measured the softmax's exponential at **1.689 ms of a 13.887 ms token at mean
position 640**. As a slope that is `1.689/640 = 0.00264 ms/position`, against E49's total
context term `b(avx4) = 0.008144` — **about 32% of the whole context slope**, and the context
slope is what sets `C50`. A vectorised `exp2` softmax is therefore the largest identified lever
on the goal. **E51 also showed I cannot presently measure a change of that size**: its speed
phase resolved one window in five.

Calibrating the instrument is cheaper than the experiment it protects. This one is minutes.

## 2. What is measured

Rate on the standing arm (`e40_r128.bin --carve-k 3`, `--bench 300`, `--threads 6`,
`donor_engine.exe`) against **foreign occupancy** as `G-E44b2` defines it — system busy minus
the engine process's own CPU, from `GetProcessTimes` on the child handle.

Foreign load is produced by `L` busy single-core processes, `L ∈ {0, 1, 2, 3, 5, 8}` on a
6c/12t part. `L` is the knob; **the measured foreign occupancy, not `L`, is the x-axis** — the
knob is not assumed to deliver its nominal level.

## 3. The confound this design exists to defeat

E51 addendum A.4 established that this machine's *level* drifts downward under sustained load
and stays down: at n=1280 both arms stepped 84.3 → 75.9 tok/s at the same repetition. A naive
sweep that walks `L` from 0 upward would produce a falling rate **whether or not foreign load
costs anything**, and would look exactly like a clean dose-response.

Two defences, both registered:

1. **The level order is rotated.** Rep 1 runs the levels in the order given, rep 2 rotated by
   one, rep 3 by two. Every level therefore appears early, middle and late.
2. **`L = 0` is measured first and last, outside the rotation.** These two bracket the whole
   run and are the drift witness.

## 4. The gates

### `G-E52a` — the knob must work, and the meter must see it (planted control)

Foreign occupancy must rise monotonically in the median across `L = 0, 1, 2, 3, 5, 8`, and
`L = 0` must read **below 5%**. If the meter cannot see load it is deliberately given, no null
from it counts. **This is the control; nothing below is read if it fails.**

### `G-E52b` — the drift witness

The opening and closing `L = 0` cells are compared. Let `d` be their relative difference.

* `d` at or below **2.3%** — the rep-to-rep dispersion E44 run 2 measured at 3.8% foreign — and
  the run is **CLEAN**: what follows is about load.
* `d` above 2.3% and the run is **DRIFT-CONTAMINATED**. The cells are still printed, the fit is
  still shown, and it is labelled as not separating load from time. **No bar is set from a
  contaminated run.**

### `G-E52c` — the cost of foreign load

Fit `rate = r0 · (1 − k · f)` over all cells, `f` = measured foreign occupancy in percent.
Report `k` with the spread of the per-level medians, and apply E49 addendum C's rule: **a level
whose three reps disperse by more than 6% takes no verdict and is excluded from the fit**, and
if fewer than three levels survive, `k` is **NOT COMPUTABLE** and no bar is set.

### `G-E52d` — the bar, by a rule fixed before the data

**The bar is the largest foreign occupancy whose predicted rate cost is at most half the
instrument's own dispersion at zero load.** With E44 run 2's 2.3%, that is a cost of **1.15%**,
so `bar = 1.15 / k` with `k` in percent-rate per percent-occupancy.

The rule is registered, not the number. Whatever it returns is the new `OCC_BAR`, including a
number *larger* than 15.0 — which is the outcome that would tell me the current bar was fine and
that run 1 and run 2 differ for some other reason.

**A bar may only be lowered by this rule, never by my picking a level after seeing the rates.**
That is `G-E45c` and `G-E49c` and it is not happening a fourth time.

## 5. My predictions, registered

| quantity | prediction |
|---|---|
| `G-E52a` | fires; `L = 0` reads 1–4%, `L = 8` reads 55–70% |
| `G-E52b` | **CLEAN** — the whole run is ~10 minutes, far short of the 35 that drifted E51 |
| `k` | **0.6–1.2% of rate per point of foreign occupancy**, bracketing the 0.85 that E44's two runs imply |
| `G-E52d` | bar lands at **1–2% foreign**, i.e. **far below the present 15.0** |
| shape | the fit will be poor at high `L`: past 6 busy processes the engine's own 6 threads are being descheduled, which is not the same mechanism as light contention. I expect the linear form to hold to ~20% and bend after |

If `G-E52d` returns a bar near 1%, the honest consequence is uncomfortable and is accepted in
advance: **no timing this programme has ever published was taken under it**, and the corrected
statement is that absolute rates carry their ±5% for a measured reason rather than a
conventional one.

## 6. Scope

E52 sets a bar on an instrument. It does not re-time any arm, does not revisit `C50`, does not
touch quality, and produces no number that may be quoted as this engine's speed. The one arm it
runs is the standing one, purely so the y-axis is in familiar units.
