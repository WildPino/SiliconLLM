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

---

# ADDENDUM B — RUN 2: ALL FOUR GATES FIRE, THE BAR IS **4.39%**, AND MOST OF MY PREDICTIONS WERE WRONG

Run 2 completed on an idle box with no polling. Every gate returned a verdict.

## B.1 The run

    open   L=0    107.05 tok/s   system 41.9%   foreign  8.9%      <- the witness, before anything
    rep 1/3  order [0,1,2,3,5,8]
       L=0  108.24  foreign  6.2%      L=3  102.85  foreign 30.9%
       L=1  104.15  foreign 16.6%      L=5   97.05  foreign 46.1%
       L=2  102.85  foreign 26.1%      L=8   46.97  foreign 71.3%
    rep 2/3  order [1,2,3,5,8,0]
       L=1  105.64  foreign 14.9%      L=8   53.30  foreign 69.4%
       L=2  105.69  foreign 21.0%      L=0  110.35  foreign  4.4%
       L=3  103.84  foreign 29.8%
       L=5   97.90  foreign 44.9%
    rep 3/3  order [2,3,5,8,0,1]
       L=2  105.56  foreign 20.2%      L=0  109.99  foreign  4.4%
       L=3  104.41  foreign 29.1%      L=1  108.06  foreign 12.0%
       L=5   96.46  foreign 47.6%
       L=8   31.46  foreign 71.7%
    close  L=0    109.29 tok/s   system 37.4%   foreign  4.0%

| gate | verdict | number |
|---|---|---|
| `G-E52a2` (control) | **FIRES** | monotone; `L=0` is the minimum at 4.4%; span **16.1×** to 71.3% |
| `G-E52b` (drift) | **CLEAN** | bracketing `L=0` cells agree to **2.09%**, inside the registered 2.3% |
| `G-E52c` (cost) | **FIT** | `rate = 111.19 · (1 − 0.2622/100 · f)` → **k = 0.262% of rate per point** |
| `G-E52d` (the bar) | **4.39% foreign** | `1.15 / 0.262`; previous `OCC_BAR` was 15.0 |

Per-level medians entering the fit, with their three-rep spread:

| L | tok/s | foreign | spread | in the fit? |
|---|---|---|---|---|
| 0 | 109.99 | 4.4% | 1.9% | yes |
| 1 | 105.64 | 14.9% | 3.7% | yes |
| 2 | 105.56 | 21.0% | 2.7% | yes |
| 3 | 103.84 | 29.8% | 1.5% | yes |
| 5 | 97.05 | 46.1% | 1.5% | yes |
| 8 | 46.97 | 71.3% | **46.5%** | **NO** — `L > 6`, excluded structurally by A.5, and it would also have failed the 6% dispersion rule |

## B.2 The scorecard, and it is mostly **wrong**

| quantity | §5 predicted | A.6 revised | measured | verdict |
|---|---|---|---|---|
| `k` | 0.6–1.2 %/pt | 0.3–0.7 %/pt | **0.262 %/pt** | **WRONG in both** — below even the revised floor |
| the bar | 1–2% | 2–4% | **4.39%** | **WRONG in both** — above both ranges |
| shape | "linear to ~20% foreign, bend after" | — | linear and tight to **46.1%** | **WRONG** |
| span `L=0 → L=8` | — | ~11× | **16.1×** | **WRONG**, in the safe direction |
| "a bar the machine cannot meet" | — | asserted | `L=0` reads 4.4%, close witness 4.0% | **scored WRONG here; CORRECTED to right in B.7 below** |
| `G-E52a2` | fires | fires | fires | right |
| `G-E52b` | CLEAN | CLEAN | CLEAN, 2.09% | right |
| `L=8` collapses | yes (§5) | yes | 31–53 tok/s, 46.5% spread | right |

**3 right, 5 wrong.** The pattern in the misses is one error, made twice: I assumed light foreign
load is expensive. It is not. Six free hyperthreads absorb five busy processes with a **3%**
total loss of rate, and the linear form stays inside ±1.4% of the fit across the whole fitted
domain:

    f= 4.4%   measured 109.99   fit 109.91   residual +0.08%
    f=14.9%   measured 105.64   fit 106.85   residual -1.13%
    f=21.0%   measured 105.56   fit 105.07   residual +0.47%
    f=29.8%   measured 103.84   fit 102.50   residual +1.31%
    f=46.1%   measured  97.05   fit  97.75   residual -0.72%

The bend I predicted at 20% is not there. What *is* there is a cliff, and it is not a bend in the
same curve: at `L = 8` the engine's six threads are being descheduled, the rate falls to a third
and the dispersion goes to 46.5%.

**Two independent rules exclude that level** — A.5's structural `L ≤ 6` domain, and §4's
pre-existing 6% dispersion rule, which 46.5% fails by a factor of eight. Either alone suffices,
so the fit is not resting on the rule I added after seeing run 1. But it is worth stating what
the fit would have said with **neither**: `k = 0.724 %/pt`, `r0 = 122.6`, bar **1.59%**. That is
**inside §5's predicted `k` of 0.6–1.2 and inside §5's predicted bar of 1–2%.** My original
predictions would have been scored as *correct* by a fit that had oversubscription in it. The
scorecard above is worse for me than the one I would have written without the exclusion rules,
which is the whole point of registering them first.

## B.3 Consequence 1 — `OCC_BAR` becomes 4.39

Registered in §4: *"Whatever it returns is the new `OCC_BAR` ... The rule is registered, not the
number."* Applied: `e44_interval.py:74` changes `OCC_BAR = 15.0` → `OCC_BAR = 4.39`.

The bar now means something checkable: **at 4.39% foreign the predicted rate cost is 1.15%, half
the 2.3% dispersion the instrument shows at zero load** — i.e. load is not permitted to move the
reading by more than a fraction of what the reading wobbles anyway. At the old 15.0 the permitted
cost was **3.93%**, nearly double the whole dispersion budget.

**One asymmetry, recorded because the same constant now guards two different quantities.**
`G-E44b2` judges *during-rep* foreign occupancy, which is what `k` was fitted against — that is
the matching use. The pre-flight guard compares the same 4.39 against *idle* system busy with no
engine running, and those are not the same number: during-rep foreign still contains OS work the
engine induces but that `GetProcessTimes` does not charge to it (page faults, I/O completion,
scheduling). Idle reads lower — E44 run 2's guard read **2.2%** where E52's `L = 0` cells read
**4.4%** — so applying the bar to the guard is the conservative direction and is left as is. It
does mean the guard now has little headroom on this box; a refusal there is the bar working, not
a fault, and it is not to be raised to get a run through.

## B.4 Consequence 2 — E44 addendum C's open question is answered, and only half in my favour

| E44 run | foreign during reps | new bar | status |
|---|---|---|---|
| 1 | ~12.6% | 4.39% | **not citable** — 3.3% of rate lost to load |
| 2 | **3.8%** | 4.39% | **CITABLE** — 1.0% of rate lost, under budget |

So **110.62 tok/s stands** and 102.90 does not, which is the outcome E44 addendum C suspected but
could not justify. That is the half in my favour.

**The other half is not.** The two runs differ by 7.50% (102.90 → 110.62). The fitted cost of the
load difference, 12.6% → 3.8% foreign, is only **2.33 points**. **5.2 points of the gap are not
explained by foreign occupancy at all** — consistent with E44 addendum C's thermal-soak diagnosis,
which this experiment does not measure and cannot close.

**Therefore the bar is necessary and is not sufficient.** A reading that passes `OCC_BAR = 4.39`
is not thereby comparable to another that passes it; it is only not *load*-contaminated. The ±5%
on absolute rates stays, and E44's stated reason for it stays with it. Anyone reading this as
"the instrument is now calibrated" is reading it wrong — one confound of at least two is priced.

## B.5 What E52 is not

No rate in this brief is a speed result for any arm. `109.99 tok/s` at `L=0` is the standing arm
at `--bench 300`, mean position 150, on synthetic weights; it is here as a y-axis unit and E44
run 2 remains the registered reading at that shape.

## B.6 Owed

1. **A thermal witness.** The residual in B.4 is the largest uncontrolled term in every absolute
   timing this programme publishes, and nothing measures it. A first cut costs nothing: record
   wall-clock-since-boot and the per-rep sequence position alongside each cell, and look for the
   soak in data already being collected.
2. **The bar is derived from one box on one day.** `k` is a property of this Zen 2 part with
   `--threads 6`; the *rule* ports, the *4.39* does not.
3. The vectorised `exp2` softmax brief, which E52 exists to protect, is now unblocked: at
   `k = 0.262` the instrument can resolve a change of 12.2% of the token provided the box is kept
   under the new bar.

---

## B.7 A correction to B.2, written the same day, after E53 ran under this bar

Addendum A.6 predicted the derived bar would be **"a bar the machine cannot meet"**. B.2 scored
that **WRONG at the margin**, because E52's own `L = 0` cells read 4.4% and the closing witness
4.0%.

**E53's speed phase read foreign occupancy between 4.9% and 23.8% on all fifty of its cells,
every one above the bar**, and a per-process sample taken afterwards with no engine running put
the box at **5.6%**, of which **Chrome alone is about 4.2%**. A.6 was right and my scoring of it
was wrong: E52's quiet `L = 0` cells were a quiet *moment*, not the machine's normal state, and I
read one session's floor as the floor. That is the same error as `OCC_BAR = 15.0` itself -- a
constant taken from the range one session happened to observe -- committed while scoring the
experiment that fixed it.

**Corrected, run 2's scorecard is 3 right / 4 wrong, and the prediction I got right is the
uncomfortable one.** The bar is not re-opened: it comes from a rule fixed before the data, it is
doing exactly what it was derived to do, and a bar that refuses most sessions is information
about the sessions.
