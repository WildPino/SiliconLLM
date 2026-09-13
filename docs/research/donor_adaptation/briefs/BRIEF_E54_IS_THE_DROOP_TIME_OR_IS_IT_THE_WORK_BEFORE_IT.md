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
