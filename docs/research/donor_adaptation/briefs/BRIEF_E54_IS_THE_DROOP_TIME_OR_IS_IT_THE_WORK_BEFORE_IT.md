# E54 — IS THE DROOP TIME, OR IS IT THE WORK THAT RAN BEFORE IT?

**Pre-registered 2026-09-13, pushed before a single cell runs.** No measurement in this brief has
been taken except the two marked `ALREADY MEASURED` in §3, which are the instrument's own planted
controls and were run before the instrument was trusted.

---

## 1. Why this is the blocking experiment and not a side quest

The programme's goal is a 10B-shaped model at 50 tok/s (good) / 100 tok/s (excellent). The arm
that carries that claim already reads **110.62 tok/s at mean context 150** (E44 run 2) and holds
50 tok/s out to **`C50 ≈ 1443`** (E49). What it does not have is a way to measure a *change* to
those numbers, because:

- `G-E53e` refused E53's entire speed phase on a **4.64% drift** between two identical cells.
- E53 addendum C.5 found the control arm's own slope reads **−10.6%** against E49's published
  value across sessions.
- So **`G-E53f` could establish that the new exponential is faster and not by how much**, and
  the next speed experiment — whatever it attacks — will be refused the same way.

**Every remaining attack point on speed is gated behind this.** That is the justification for
spending a box window on an experiment that improves no rate.

## 2. The question, and the confound that made the old answer premature

E53 addendum C.3 concluded the droop "is a function of time-in-run". **The design cannot support
that reading**, and I did not notice until after publishing it:

    opening witness   <-  preceded by an IDLE box
    [ n=40, 160, 320, 640, 1280 -- ASCENDING ]
    closing witness   <-  preceded by TEN n=1280 cells, the heaviest KV traffic in the run

The ladder ascends, so **"9.7 minutes elapsed" and "ten heavy cells just finished" are the same
observation**. C.3 read it as the first without excluding the second.

There is a third reading, and the corrected duration (§C.0: 9.7 minutes, not the ~48 I published)
makes it the least likely rather than the most: **CPU thermal soak**. Tonight's planted control
shows this part's clock settling *within 25 seconds* of sustained load, which is nowhere near a
nine-minute timescale.

> **E54's question, exactly.** Take an identical witness cell and place it after blocks of
> **equal wall time** but different weight. Does the witness fall **monotonically with elapsed
> time**, or does it fall after **heavy** blocks and **recover** after light ones?

## 3. The instrument, and its planted controls — `ALREADY MEASURED`

`e54_clock.py` reads `\Processor Information(_Total)\% Processor Performance` through
`PdhAddEnglishCounterW`, so the **English** counter path resolves on this Italian-locale box
rather than hardcoding one spelling. `e44_interval.Split` now stamps every cell with `seq`,
`since_start_s`, `wall_s` and `clock_pct`.

**Control 1 — the witness must fire on a known-positive** (`e54_burn.py`, 12 processes):

    idle           105.44 .. 107.23 %      4008 .. 4069 MHz
    burn    5 s    102.47 %                3887 MHz
    burn   60 s    100.73 %                3821 MHz     <- still falling
    recovery 5 s   106.20 %                4029 MHz

Complete separation, recovery inside five seconds. It also established that **two of the four
candidate counters are DEAD on this part** — `% Performance Limit` pinned at 100.0 and
`Performance Limit Flags` at 0 throughout — which is known only because the known-positive ran
first. A second burn minutes later decayed **0.49** points instead of 1.70 *because the part
started warm*, which is the soak showing its hand on a short timescale.

**Control 2 — it must discriminate at the ENGINE's duty cycle**, not only under a 12-process
burn, because that is the regime it will actually be read in:

    n=8    cells    108.57 .. 109.13 %          (near-idle)
    n=1280 cells    104.67 .. 105.12 %          (loaded)      COMPLETELY DISJOINT

Three consecutive `n=1280` cells read `105.12 → 104.83 → 104.67` across 53 s **while the rate
stayed flat at 80.5 tok/s** — so at this timescale the clock moves and the rate does not, and
`G-E54c` below is the gate that says whether that holds at the timescale that matters.

## 4. The design

One arm, one binary (`donor_engine_e53.exe --fexp libm`), one machine, one session. Arm
`R128 --carve-k 3 --threads 6`, the arm every speed number in this programme is quoted on.

    W0   |   HEAVY (180 s)   |   W1   |   LIGHT (180 s)   |   W2   |   HEAVY (180 s)   |   W3

* **`W`** = **3 reps** of `--bench 160`, identical to `G-E53e`'s witness cell. Three reps so each
  `W` carries its **own dispersion**, which is what the decision is compared against — the
  standing law that a gate's tolerance may not be tighter than the dispersion on the axis it
  watches.
* **`HEAVY`** = `--bench 1280` cells until 180 s of wall time has elapsed.
* **`LIGHT`** = `--bench 160` cells until 180 s of wall time has elapsed.
* **Blocks are bounded by TIME, not by cell count.** If the light block were shorter, elapsed
  time and block weight would be confounded again, which is the exact defect being repaired.

Estimated cost: **~11 minutes**, of which ~9 is the three blocks. *E53's run was 9.7 minutes and I
asked the user for 50; this estimate is derived from measured per-cell wall times, not guessed.*

## 5. The gates, registered now

### `G-E54a` — the witness fires
Already satisfied by §3's two controls. Recorded here so that the nulls below count.

### `G-E54b` — TIME or WORK (the discriminator, a RANK question)
Compare the four witness levels, each a median of 3 reps, with each `W`'s own min–max spread as
the yardstick:

> * **`WORK`** if `W2 > W1` by more than the larger of the two spreads — the witness *recovered*
>   across a light block of the same duration that a heavy block costs.
> * **`TIME`** if `W0 > W1 > W2 > W3` monotonically **and** `W2 − W1` is within the spreads —
>   the level fell with the clock and not with the workload.
> * **`NEITHER`** if the sequence is non-monotone in a way neither rule covers.
> * **`UNRESOLVABLE`** if any `W`'s spread exceeds the total `W0 − W3` range: the instrument
>   cannot see an effect smaller than its own noise, and says so instead of guessing.

**Planted controls required before the data is read**: a synthetic monotone series must return
`TIME`; a synthetic sawtooth must return `WORK`; a flat series with wide spreads must return
`UNRESOLVABLE`; a series where `W2 > W1` by exactly the spread must not tip.

### `G-E54c` — does the clock EXPLAIN the rate? (a SCORE, with `G-E54b` as its RANK partner)
For every cell, regress rate on `clock_pct`. Report the fraction of the witness droop predicted
by the clock droop alone.

> If the clock accounts for the droop, **every future timing gets a clock covariate and the drift
> gate can be replaced by a correction** — which is the payoff that justifies the box window.
> If it does not, the droop has a second mechanism and E54 says so without naming it.

`G-E54c` is refused if the clock readings themselves disperse more than the rate readings do —
a predictor noisier than its target explains nothing.

## 6. Predictions, before the data

Recorded so the scorecard can be kept, and deliberately specific enough to be wrong:

| | prediction |
|---|---|
| `G-E54b` | **`WORK`**. E53's closing witness sat 2.90% below the minimum of the very window it repeats, immediately after ten `n=1280` cells, and the clock control settles in 25 s — nine minutes of *time* has no mechanism behind it, ten heavy cells does. |
| size of the recovery `W2 − W1` | 2–4% |
| `G-E54c` | the clock explains **less than half** the droop — the `n=1280` triple already showed the clock moving 0.45 points with the rate flat |
| `W` spread within a block | 1.5–3%, matching E53's `n=160` libm spread of 3.9% or tighter |

**If `G-E54b` returns `WORK`, E53 addendum C.3 is wrong and gets a correction**, and the fix for
every future speed run is cheap and structural: **interleave window order** so no arm sits
systematically behind heavier work — which is the same repair `G-E53f` already applies to arms
and which the *window* axis never got.

## 7. What this does not do

It does not produce a rate, improve one, or touch quality. It does not promote `--fexp poly`. It
decides whether the drift refusal can be replaced by a correction, and that is all.

---

# ADDENDUM A — `G-E54b`'s `TIME` BRANCH WAS MALFORMED, REPAIRED BEFORE ANY CELL RAN

**Written before the run. No measurement existed when this was changed.**

§5 registered `TIME` as *"`W0 > W1 > W2 > W3` monotonically **and** `W2 − W1` is within the
spreads"*. Those two clauses contradict each other. Under a monotone decline `W2 − W1` is
negative and as large as the decline itself, so **no genuinely monotone series could ever
satisfy the conjunction** — a 120/117/114/111 series with 0.2% spreads returns `NEITHER`.

Planted control **B-1** caught it on the runner's first execution, which is what planted
controls are for. **MALFORMED under the E4 precedent, not failed.**

### The repair

> **`TIME` if `W0 > W1 > W2 > W3` monotonically.** Nothing else.

Monotonicity already excludes a recovery, so the second clause was redundant at best and
contradictory at worst.

**The direction of the change is the part that needs stating.** This makes `TIME` *strictly
easier* to reach and leaves `WORK`, `UNRESOLVABLE` and `NEITHER` untouched. §6 predicts
**`WORK`**. So the repair relaxes the branch that would prove me **wrong** and tightens nothing
in my favour — which is the only kind of pre-data repair that should be allowed to stand without
a fresh registration.

### The controls, all firing

    B-1  a monotone series reads TIME
    B-2  a sawtooth reads WORK
    B-3  noise wider than the effect is UNRESOLVABLE
    B-4  a recovery of exactly the spread does not tip
    B-5  a rise at the end is NEITHER, not TIME
    B-6  a flat series is UNRESOLVABLE, not TIME
    C-1  a planted slope is recovered
    C-2  and it is scored as fully explained
    C-3  a predictor noisier than its target is REFUSED
    C-4  too few cells is NOT COMPUTABLE
    C-5  partial explanation reports a fraction below 1

    11 of 11 fire.

---

# ADDENDUM B — THE DROOP DID NOT REPRODUCE, `G-E54b` REFUSED, AND THE RUN FOUND SOMETHING WORSE THAN THE THING IT WAS LOOKING FOR

Run: 91 cells, **9.9 minutes**, box measured at **2.10% foreign** immediately before launch.
*This run reports its own duration because every cell now carries a timestamp — the repair from
E53 addendum C.0, working on the first run after it.*

## B.1 The registered verdicts

| gate | verdict |
|---|---|
| `G-E54a` witness fires | **satisfied** (§3's two controls, measured before the run) |
| `G-E54b` TIME or WORK | **UNRESOLVABLE** |
| `G-E54c` does the clock explain it | **FIT, and it explains −7%** — i.e. nothing, with the sign backwards |

    W0  116.29 tok/s  spread 2.18%   foreign 13.65%   <- CONTAMINATED, see B.4
    W1  117.52 tok/s  spread 1.03%   (after HEAVY 193 s)
    W2  120.10 tok/s  spread 2.26%   (after LIGHT 181 s)
    W3  116.85 tok/s  spread 0.57%   (after HEAVY 180 s)

**`G-E54b` refused because the widest witness spread (2.72 tok/s) exceeds the entire `W0 − W3`
range (0.56 tok/s).** That is the refusal doing exactly what it was written to do.

## B.2 The headline: **E53's droop did not reproduce**

`W0 = 116.29` and `W3 = 116.85`. Over 9.9 minutes containing **two** heavy blocks, the witness
level did not fall at all — it ended 0.5% *higher* than it started. E53's 4.64% droop is
therefore **not a stable feature of a run of this length on this box**, and E53 addendum C.3's
"it is a function of time-in-run" does not survive its first replication attempt.

**What that does to E53.** `G-E53e` still fired and E53's speed fit is still refused — a gate
that fired is not un-fired by a later run. But the *explanation* offered for it in C.3 is
withdrawn. The 4.64% remains a measured disagreement between two cells with **no established
mechanism**: not load (C.3), not elapsed time (this run), and not the CPU clock (B.3).

## B.3 The clock tracks the workload cleanly and does **not** explain the rate

    HEAVY-1   clock 103.81 .. 104.48   median 104.37
    LIGHT     clock 105.37 .. 105.83   median 105.62
    HEAVY-2   clock 104.16 .. 104.58   median 104.44

A clean **1.2-point** separation between heavy and light blocks, and the witnesses inherit it:
after heavy `W1 105.48` and `W3 105.19`, after light `W2 105.67`. **The instrument works and the
WORK mechanism is visible in it.**

It just does not drive the rate. The decisive cell is `HEAVY-2` read in order:

    rate   81.8  82.0  79.9  76.9  76.7  76.8  77.7  79.7  79.4  80.3     range 6.9%
    clock 104.58 104.47 104.38 104.47 104.44 104.44 104.47 104.37 104.16 104.38   range 0.40%

**A 6.9% swing in rate against a 0.40% move in clock.** The clock cannot be the cause, and
`G-E54c`'s pooled fit says the same thing with the wrong sign. **Prediction scored: right** — §6
said the clock would explain less than half; it explains none.

## B.4 What contaminated `W0`, and it is my own fault by design

    W0 foreign occupancy, in order:  13.9%   13.6%   12.4%

The box was at 2.10% before launch. **The run's own launch — `nohup`, the Python interpreter,
the shell — is inside its first cells**, and `W0` is the only witness placed there. That is a
design defect in every runner here, not a property of this box: **the first cells of a run
measure the run starting up.**

Fix, cheap and structural: **a discarded warm-up cell before the first recorded one.** Without
it, `W0` is the very comparison point the whole design rests on, and it is the one cell
guaranteed to be dirty.

*Excluding the contaminated `W0`, the remaining three are the sawtooth §6 predicted:*
`W1 117.52 → W2 120.10 → W3 116.85`, a recovery of **+2.20%** — inside §6's predicted 2–4% band
and **below the 2.72% noise bar by 5%**. It is not promoted, it is not scored, and `G-E54b`
stays `UNRESOLVABLE`: a result that misses its bar by 5% is a miss.

## B.5 The finding that outranks the question: **a min–max spread is not a dispersion**

The `LIGHT` block ran **58 identical `--bench 160` cells back to back**. That is by far the
largest single-cell sample this programme has ever taken, and it was a by-product.

    58 identical cells:  min 114.92   median 119.31   max 128.73   FULL RANGE 11.57%

Resampling from those 58 real cells, 4000 draws per `k`:

| reps `k` | median min–max spread | median interquartile width |
|---|---|---|
| 3 | **2.70%** | — |
| 5 | **3.85%** | 0.58% |
| 10 | 5.53% | 1.69% |
| 20 | 7.91% | 2.10% |
| 40 | 10.31% | 2.25% |
| 58 | **11.57%** | **2.29%** |

**A 3-rep min–max reports about a quarter of what the same cell actually does.** This is not an
empirical quirk — the min–max of a sample is a *range statistic*, and a range grows with sample
size without bound. The interquartile width converges (2.10 → 2.25 → 2.29); the range never
does.

**Every dispersion this programme quotes is a small-`k` min–max**, and the bias is in the
dangerous direction — it makes gates calibrated on it **too permissive**, which is the opposite
of the failure catalogued in `feedback_gate_vs_measured_dispersion`:

* **E44's headline `110.62 tok/s, spread 2.3%` is a 5-rep range.** On the one cell type where 58
  reps exist, a 5-rep range reads 3.85% where the truth is 11.57%.
* `G-E52b`'s `2.09%` CLEAN, `G-E53e`'s `2.3%` drift rule, `G-E53d`'s `6%` dispersion refusal, and
  `G-E54b`'s own witness spreads are all the same statistic.

**What may NOT be concluded from this.** These 58 cells are `--bench 160` in this session; E44's
were `--bench 300` in another. **The 11.57% does not transfer to E44's number** and I am not
restating the headline from it. What transfers is the arithmetic: *a 5-rep range understates
dispersion, here by about 3×, and the amount is unknown for any cell where 5 reps is all that
was taken.*

**This is post-hoc and is NOT promoted to a gate** (E14 §6). It is registered as E55's question.

## B.6 A second instrument defect, observed and not yet explained

Breaches of `OCC_BAR = 4.39`, by block:

    W0       3 of  3   foreign median 13.65%      <- the launch, B.4
    HEAVY-1  8 of 11   foreign median  7.14%
    LIGHT   12 of 58   foreign median  3.57%
    HEAVY-2  8 of 10   foreign median  6.07%

On a box measured at 2.10% foreign, the **heavy** cells report roughly **double** the foreign
occupancy of the light ones. Nothing else was running. The standing hypothesis is that kernel
work done *on behalf of* the engine — page faults, TLB shootdowns, zeroing — is charged to the
system but not to the child process, so `system − child` counts it as foreign; and long-context
cells generate more of it.

If that is right, **`OCC_BAR` systematically penalises exactly the long-context cells that `C50`
is built from**, which would be a bias in the gate rather than noise. It is stated as a
hypothesis with a measurement attached and nothing is concluded from it here.

## B.7 Scorecard

| §6 prediction | measured | |
|---|---|---|
| `G-E54b` returns `WORK` | **`UNRESOLVABLE`** | **WRONG** |
| recovery `W2 − W1` of 2–4% | +2.20% | inside the band, **not scorable** — the gate refused |
| the clock explains less than half | **−7%** | **right** |
| `W` spread 1.5–3% | 2.18 / 1.03 / 2.26 / 0.57% | two inside, two below — and B.5 says the statistic itself was the wrong one |

## B.8 What is owed

1. **E55: measure a dispersion properly, once, on the arm the headline is quoted on.** ≥40 reps
   of E44's own cell, reported as a **quantile interval**, which B.5 shows is stable in `k`.
   Until then, **every `±` in this programme is a lower bound**, and the ledger must say so.
2. **A discarded warm-up cell** at the head of every runner (B.4).
3. **Decompose foreign occupancy** into contention and kernel-on-behalf-of-the-child (B.6),
   because if `OCC_BAR` is biased against long-context cells it is biased against the target.
4. The 4.64% of E53 still has **no mechanism**: not load, not elapsed time, not the clock.
