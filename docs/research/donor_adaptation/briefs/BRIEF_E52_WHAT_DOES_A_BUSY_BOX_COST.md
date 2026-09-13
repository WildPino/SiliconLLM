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

---

# ADDENDUM A — `G-E52a` FAILED, ONE OF ITS TWO CLAUSES SHOULD NEVER HAVE BEEN WRITTEN, AND RUN 1 IS NOT SCORED

**Registered before run 2 exists. Run 1's cells are kept and NOT scored** — the control stopped
the run, so run 1 produced no measurement, and nothing below is a re-reading of data whose
answer I have already seen being made readable.

## A.1 What run 1 did

    open   L=0    110.76 tok/s   system 38.3%   foreign  4.9%
    rep 1/3  order [0,1,2,3,5,8]
       L=0  107.79   foreign  9.0%      L=3   97.47   foreign 35.7%
       L=1  105.28   foreign 16.1%      L=5   82.36   foreign 50.3%
       L=2  103.07   foreign 24.0%      L=8   30.02   foreign 72.2%
    rep 2/3  order [1,2,3,5,8,0]
       L=1  107.23   foreign 11.1%      L=5   92.77   foreign 48.5%
       L=2  104.59   foreign 23.4%      L=8   48.02   foreign 71.0%
       L=3   95.81   foreign 33.9%      L=0  109.09   foreign  5.1%
    rep 3/3  order [2,3,5,8,0,1]
       L=2  105.29   foreign 20.8%      L=0  109.38   foreign  6.1%
       L=3  101.59   foreign 32.2%      L=1  107.46   foreign 11.9%
       L=5   96.62   foreign 47.2%
       L=8   51.81   foreign 69.6%
    close  L=0    108.78 tok/s   system 40.1%   foreign  6.7%

    G-E52a (control) : FAILS
       L=0 reads 6.1% foreign, above the 5.0% this control requires

**The control did its job and stopped everything after it.** No fit was computed, no bar was
derived, and the results file holds the cells and the failed control and nothing else.

## A.2 Two causes, and the second is the one worth recording

**Cause 1, mine and avoidable.** Rep 1's `L=0` read `9.0%` while I was polling the log from
another shell. The opening witness, taken before I started, read `4.9%`. *I was the foreign
load in the experiment measuring foreign load.* Run 2 is run without polling.

**Cause 2, and it is the same defect this experiment exists to fix.** Reps 2 and 3 read `5.1%`
and `6.1%` with nothing of mine running, and the closing witness read `6.7%`. **This box's idle
foreign floor is 5–7%.** I registered the clause `L = 0 must read below 5%` without ever
measuring that floor — so, exactly like `OCC_BAR = 15.0`, it is a constant chosen against a
quantity nobody had measured, and on this machine **it could not have passed whatever the data
did.**

That is `G-E44b`'s defect, committed by me, inside the brief written to correct `G-E44b`. It is
recorded here in those words rather than softened.

## A.3 But the clause was also conceptually unnecessary, which is why it is removed and not retuned

§2 registered that **the x-axis is the measured foreign occupancy, not `L`** — "the knob is not
assumed to deliver its nominal level". Once that is true, the experiment does not need a quiet
baseline at all. It needs a *range* of foreign levels and a meter that reads them. A baseline
sitting at 6% rather than 2% moves where the leftmost point sits on the x-axis and changes
nothing else; the fit is over what was measured.

The clause was a leftover from the older question — *"is the box quiet?"* — which is the very
question E52 exists to replace with a measured one.

**So it is removed, not loosened.** No threshold is raised to let data through: a clause that
tested the wrong thing is deleted, and what remains is structural.

## A.4 `G-E52a2`, registered, with no constants in it

The control keeps the half that carries the content — **the knob must work and the meter must
see it** — expressed without any number I could have chosen:

1. The median foreign occupancy must be **monotone non-decreasing in `L`**.
2. `L = 0` must be the **minimum** of the levels.
3. The span from `L = 0` to the largest `L` must be at least **10×** — the meter must not merely
   order the levels, it must resolve them. (This is a ratio, not an absolute, and it is the one
   number here; it is set so that a meter returning noise around a constant cannot pass, which
   is the `C2` requirement from E44 addendum B in its general form.)

`G-E52b`, `G-E52c` and `G-E52d` are **unchanged in every constant and every rule.**

## A.5 One more thing run 1 exposed in my own brief, and it is fixed the same way

§5 predicted: *"the fit will be poor at high `L`: past 6 busy processes the engine's own 6
threads are being descheduled, which is not the same mechanism as light contention."*
`L = 8` read `30.02 / 48.02 / 51.81` tok/s — a **58% spread**, and a collapse to a third of the
baseline.

**And §4 gave `G-E52c` no way to act on that prediction**: it fits every surviving level, so
the collapse would have dragged the slope and produced a `k` describing oversubscription where
the bar needs one describing contention. The dispersion rule would probably have dropped `L = 8`
by accident — 58% is well past 6% — but "probably, by accident" is not a specification.

**Registered here, on a structural ground decided by the hardware and not by the data:** the
model `rate = r0·(1 − k·f)` describes **contention**, and contention becomes oversubscription
once the foreign processes outnumber the logical CPUs the engine is not using. The engine runs
`--threads 6` on a 6c/12t part, so **only levels with `L ≤ 6` enter the fit.** On this machine
that excludes `L = 8` and keeps `L = 5`, and it would have done so before any cell was measured.

## A.6 What run 2 is, and what it may produce

Same arm, same levels, same reps, same rotation, same drift witness, same `G-E52b/c/d`. Run 1's
cells stay in `results/e52_occupancy_cost_run1.json`, **unscored**, as the record of a control
that fired.

**Predictions from §5 are not re-registered and not re-scored** — they were made before run 1
and stand as they were. One is added, because A.5 changed the fit's domain:

| quantity | prediction |
|---|---|
| `G-E52a2` | fires; the span from `L=0` to `L=8` is ~11× (6% → 70%) |
| `k`, now fitted on `L ≤ 6` only | **0.3–0.7% of rate per point** — below §5's 0.6–1.2, because that guess was anchored on E44's two runs, whose difference I now suspect was part thermal |
| `G-E52d` | bar at **2–4% foreign** — and on this box's 5–7% floor, **that is a bar the machine cannot meet**, which would be the finding rather than a failure |
