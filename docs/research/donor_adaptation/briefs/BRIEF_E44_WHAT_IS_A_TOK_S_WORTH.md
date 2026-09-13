# E44 — what is a tok/s worth? Page-cache residency, and the interval every rate in this programme is missing

**PRE-REGISTERED. Pushed before any measurement.** INDEX §4 item **0-A**, which outranks
everything below it because it prices every number in §2.

## 0. The one sentence

Every absolute tok/s this programme has published is a **point estimate on a quantity that
disperses 9–22% on an idle box**, and the single named candidate for that dispersion —
**page-cache residency of the 5.49 GB weight file** — has never been controlled in E39, E40 or
E43. E44 controls it, and returns rates **with intervals**.

## 1. Why this is not a re-run of E43

E43's speed half is **VOID, permanently** (its addendum D). E44 does not revisit E43's
question, does not reuse its cells, and may not resurrect its number. What E43 produced by
accident is the *observation* that motivates E44 (ledger §53.9.4):

| | |
|---|---|
| box occupancy across both E43 sessions | **1.0–13.3%** — the quietest measured here |
| cells | 40 |
| median within-cell spread | **10.9%** and **9.2%** |
| cells over the 13.5% bar | **4 of 20 in each session** |
| same magnitude on a *busier* box (E40) | yes |

**Emptying the machine did not shrink the dispersion.** That is the finding: it is not
contention, so the standing rule *"a contended timing is not a timing"* is correct and
**insufficient**. An uncontended timing is not a point estimate either.

Consequence already applied: `112.73 tok/s` is really **≈113–130** (quiet readings 128.54,
123.33, 130.30). Consequence **not** applied anywhere: no absolute rate in this programme
carries a measured interval, and several decisions were priced off point estimates.

## 2. The hypothesis, and the honest state of it

**H-CACHE: the engine's rate depends on how much of the weight file is resident in the OS page
cache when the run starts.** The arm reads a **5.49 GB** file, repeatedly, on an **80 GB** box.
A warm file is served from RAM; a cold one from D:. Nothing in E39/E40/E43 set that state, and
the runs were launched in whatever order the session happened to take.

**This is a HYPOTHESIS and §53.9.3 already weakened it** as an explanation of E43's single
outlier. It is **untouched** as an explanation of the *level*. E44 must be able to return
"page cache explains none of it", and that outcome is as publishable as the other.

## 3. The arms — residency is SET, not observed

Three arms per configuration, each a deliberate cache state:

| arm | how | what it is |
|---|---|---|
| **COLD** | evict the file from the standby/page cache immediately before the run | the state a first run after boot sees |
| **WARM** | read the file end to end once, then run | the state a repeated run sees |
| **OVER** | a weight footprint deliberately larger than free RAM | the state a real 10 B deployment sees |

`OVER` is the one that matters for the goal and the one nobody has measured: **the target shape
is 10 B, and a 10 B ternary arm is not guaranteed to fit the page cache alongside everything
else.** If `OVER` is materially slower than `WARM`, then every rate in §2 was measured in a
regime the product will not be in.

**Eviction must be verified, not assumed.** Windows has no `drop_caches`; the arm is whatever
mechanism is used *plus a measurement that it worked* (resident-bytes before and after). An
eviction that silently no-ops turns `COLD` into a second `WARM` and the whole experiment reads
"no effect" — the exact plausible artefact the planted-control law exists for.

## 4. The gates, registered now

**`G-E44a` — PLANTED CONTROL, and nothing below counts until it fires.**
The instrument must show it can **tell the two states apart at all**: resident bytes for the
weight file after `COLD` must be a small fraction of the file, and after `WARM` essentially all
of it, measured and printed. *If eviction cannot be demonstrated, E44 reports that it could not
be run and spends nothing further.* No null from an unverified evictor.

**`G-E44b` — every rate is reported with dispersion or not at all.**
≥ 5 reps per cell, idle box, and the cell reports median **and** min–max spread. A single
number may not appear in the output. This is the gate E43 taught us to write.

**`G-E44c` — ORDINAL, no tolerance: `COLD` ≥ `WARM` in time per token.**
A cold cache cannot be *faster* than a warm one. If it reads faster, the arms are mislabelled
or the eviction ran at the wrong moment, and the experiment is void rather than interesting.

**`G-E44d` — the decision gate, and it is stated so it can fail:**
does residency explain the 9–22%? It does **iff** the within-arm spread, with residency held
fixed, is **materially smaller** than the 9–22% seen across uncontrolled runs. Registered
threshold: **within-arm median spread ≤ 5%** in `WARM` counts as "explained"; **> 9%** counts
as "not explained, the dispersion is something else". Between the two is **INCONCLUSIVE** and
will be reported as such, not rounded to the nearer story.

## 5. What E44 may and may not conclude

**MAY:** re-quote §2's rates as intervals; state whether residency is a controllable lever;
state whether the `OVER` regime — the one a real 10 B lands in — is slower than what was
published.

**MAY NOT:** mint a new headline tok/s. **E43's cells stay void.** If `WARM` reads higher than
`112.73`, that is not a faster engine, it is the same engine measured in a state the old number
did not record. Any re-quote replaces a point with a band and says which arm it came from.

**MAY NOT** be used to reopen E40's or E43's verdicts. Those were decided on **ratios**, and
the standing convention is that ratios do not carry the ±5% that absolutes do.

## 6. Cost and dependencies

CPU only, no GPU, no training. **Needs an idle box** — it is a speed measurement, so the
standing rule applies in full and then some, since dispersion is the *subject*. It does **not**
need the user to do anything beyond not using the machine during the reps, which E43 has
already shown they are willing to arrange.

**It does not block H1.** H1 is a quality experiment on a GPU and takes no timing at all
(brief §9). E44 and H1 do not contend.

## 7. The prediction, written before the run

I expect `WARM` − `COLD` to be **real but smaller than the dispersion it is supposed to
explain**, i.e. `G-E44d` returns *not explained* or *inconclusive*, and the residual points at
DVFS/boost behaviour rather than I/O. I am recording that so that a clean "residency explains
it" result counts for something, and so that the more likely messy outcome cannot be
retro-fitted into a tidier story.

---

# ADDENDUM A — H-CACHE IS DEAD, AND THE PROGRAM TEXT KILLED IT BEFORE A SINGLE REP WAS SPENT

**Written and pushed before any E44 measurement.** The brief's §2 premise is **wrong, and it was
mine**: *"The arm reads a 5.49 GB file, repeatedly."* It does not. It reads it **once, into
private memory, before the clock starts.**

## A.1 Three independent facts, all in `donor_engine.c`

**1. There is no mapping.** `load()` is `xmalloc(sz)` followed by a chunked `fread`
(`donor_engine.c:800-820`). No `mmap`, no `CreateFileMapping`, no `MapViewOfFile` anywhere in
the file. After `load()` returns, every weight lives in the process's **anonymous private
heap**, which the OS page cache does not serve.

**2. There is nothing left to fault.** `fread` *writes* every byte of the blob, so every page is
committed and resident by the time `load()` returns. There is no lazy first-touch cost waiting
inside the decode loop — not even for a carved run, where a single forward touches only 6.25% of
the expert weights and would otherwise leave most pages cold.

**3. The clock excludes all of it.** In `bench` mode (`donor_engine.c:1459-1477`):

```c
state_init(&s,&M,arg3+2);
forward(&M,&s,1,0);                 // warm
double t0=now_s();                  // <- the clock starts HERE
for(long i=0;i<arg3;i++) forward(...);
double dt=now_s()-t0;
```

`t0` is taken after the file read, after `state_init`, and after a warm forward. **No file is
read inside the timed window.** A mechanism that is absent from the window cannot explain
variance measured in it.

## A.2 What that retires, and what it does not

**RETIRED:**

* the **COLD** and **WARM** arms as explanations of the 9–22% dispersion;
* **`G-E44a`**, the eviction planted control — doubly. It is **moot** (nothing to evict in the
  window) and it is **unsatisfiable on this box**: the session is not elevated and neither
  RAMMap nor EmptyStandbyList is present, so eviction could not have been *demonstrated*. Its
  own clause then applies verbatim — *"If eviction cannot be demonstrated, E44 reports that it
  could not be run and spends nothing further."* It is recorded as **not run**, and no null is
  claimed from it;
* **`G-E44c`** (`COLD ≥ WARM`) — moot with its arms;
* **`G-E44d`**, the decision gate — it **cannot fire in either direction** and is withdrawn. It
  could only ever have returned "not explained", and returning that from a live experiment
  would have dressed a structural fact as an empirical one.

**NOT RETIRED — this is the part that matters:**

* **The dispersion is real and still unexplained.** 40 cells, 9–22% on the quietest box ever
  measured here (1.0–13.3% occupancy), same magnitude on a busier one. Item 0-A's actual content
  stands untouched: **no absolute tok/s in this programme carries a measured interval**, and
  `112.73` is really ≈113–130.
* **`G-E44b` survives unchanged** — every rate reported with dispersion or not at all, ≥5 reps,
  idle box. It is now the *whole* of E44.
* **§7's registered prediction is confirmed, and by the wrong method.** I wrote *"the residual
  points at DVFS/boost behaviour rather than I/O"* before knowing any of this. It is right, but
  it is now established **structurally** rather than measured, so it counts as a prediction that
  survived, not as a result E44 produced.

## A.3 `OVER` is not retired. It is re-aimed, and it found a design constraint

`OVER` asked what happens when the weight footprint exceeds free RAM. The answer is not
"slower". **The loader `xmalloc`s the entire blob, so this engine cannot run a model that does
not fit in RAM at all — it dies rather than degrades.**

For the target shape that is fine with margin: the 10 B ternary blob is **5,485,888,696 bytes
(5.49 GB)** against **~59 GB available**. But `SCALEUP_ARCHITECTURE`'s knowing-half is
explicitly *"experts streamed from DRAM"*, and a streamed design needs a loader this engine does
not have. **That is a constraint discovered now, on a 5 GB artifact, instead of at 10× the size.**

## A.4 A fact about this box that the programme had never recorded

`D:` — where **every** weight artifact lives, including `e40_r128.bin` — is a **Seagate Basic
external hard disk on USB 3.1**, not internal storage. (`C:` is a Kingston NVMe SSD with only
**40.2 GB** free.)

This does **not** touch any published tok/s, for exactly the reason in §A.1: storage is outside
the timed window. It matters in two other places, and both are worth having on the record:

* **startup**, which is real time the user waits and which has never been reported beside a rate;
* **any future streaming design**, which would be reading at USB-HDD rates (~100–140 MB/s for
  this class of drive), not at NVMe rates. A design that assumes it can stream experts from
  storage on this machine is assuming a bandwidth this machine does not have on that path.

## A.5 What E44 is now, and what it needs

E44 collapses from a three-arm residency experiment to one question — **the interval question** —
and one gate, `G-E44b`. Reporting load time beside decode rate is added, because it is free and
because §A.4 makes it non-trivial.

It is a **speed** measurement, so the standing rule applies in full and then some: the box must
be idle, ≥5 reps, median and spread, no single number. That is the one thing here I cannot do
alone, and it is recorded in `COMMUNICATION.md` rather than left in chat.

**No cell of E44 has been measured at the time this addendum is pushed.**

---

# ADDENDUM B — `G-E44b` IS MALFORMED: THE OCCUPANCY METER COUNTS THE ENGINE'S OWN THREADS

**Registered before run 2 exists.** Run 1 (2026-09-13, the first time the box was ever quiet
enough for the guard to pass) is reported in full below, and nothing in it is re-run.

## B.1 What run 1 did

The guard's **planted control fired** — 12 deliberately busy processes read `occupancy median
100.0%` against the `15.0%` bar and the guard refused, as it must. Then the real guard:

    occupancy over 20s: median 10.4%  (min 6.2, max 21.1)  bar 15.0%
    box is quiet enough -- proceeding

    rep    tok/s     engine dt     wall      outside window    occupancy during
      1    103.36      2.902 s      4.88 s       1.98 s         41.9%
      2    102.90      2.915 s      4.88 s       1.96 s         43.2%
      3    101.35      2.960 s      4.90 s       1.94 s         49.5%
      4     98.32      3.051 s      5.01 s       1.95 s         49.5%
      5    104.16      2.880 s      4.74 s       1.86 s         46.5%

    RATE      median 102.90 tok/s   min 98.32   max 104.16   spread 5.7% of median
    G-E44b  >= 5 reps, quiet box, median AND spread reported : *** FAILS ***
       reps above the bar during the run: [1, 2, 3, 4, 5]

## B.2 Why that is a defect in the gate and not a reading of the box

`Occupancy.sample()` is `1 - idle/(kernel+user)` from `GetSystemTimes` — **system-wide, and the
engine is part of the system.** The engine runs `--threads 6` on a 6c/12t part, so while it is
decoding it *is* roughly 50% of the machine by construction. The guard reads 10.4% with nothing
running and 42-50% with the engine running, and the difference is the engine.

`OCC_BAR = 15.0` is documented in the source as the bar for *"a box quieter than the quietest
conditions this programme has ever achieved"* — a statement about **foreign** load. Comparing it
to a number that structurally includes the treatment means:

* **No possible state of the machine can pass it.** Not an idle box, not a box with the user
  logged out. The gate could never have fired, on any run, ever.
* It is therefore **MALFORMED, not FAILED** — the E4 precedent, where a gate that could not be
  answered was recorded as `MALFORMED` rather than scored.

Run 1's own verdict line stands as printed and the band it produced is **not citable**, exactly
as the runner says. What is withdrawn is the *interpretation* that the box moved under the reps.
It did not; the engine did.

The same meter produced the `occ 18-52%` column in E51's speed phase. Nothing in E51 changes —
no E51 gate consumed occupancy — but that column should be read the same way.

## B.3 `G-E44b2`, registered here, with NOTHING relaxed

Same arm, same `--bench 300`, same `--threads 6`, **same `OCC_BAR = 15.0`**, same
`MIN_REPS = 5`. One thing changes, and it is *what is measured*:

> **Foreign occupancy** over an interval = system busy time minus the engine process's own CPU
> time (`GetProcessTimes` on the child handle, kernel + user), divided by the interval's total
> CPU time. Clamped at zero.

With no engine running this is identical to the old quantity, so the pre-run guard and its
planted control are unchanged.

**`G-E44b2` FIRES iff** there are at least 5 reps and **no rep's FOREIGN occupancy exceeds the
bar**. The band, median and spread are reported either way.

## B.4 Two planted controls, and the second one is the one that matters

**C1 — the meter must still refuse a genuinely busy box.** 12 burn processes, no engine: foreign
occupancy must exceed the bar. This is run 1's existing control, re-run against the new
quantity.

**C2 — the meter must DISCRIMINATE, not merely return a small number.** A metric that always
returned zero would pass every box and look exactly like a correct one. So, during a real engine
rep, the run must show **system busy above the bar AND foreign below it, separated by at least
25 percentage points.** If foreign tracks system, the subtraction is not happening and the run
**STOPs without producing a rate**.

Without C2 this addendum would be indistinguishable from lowering the bar until the gate passes,
which is what `feedback_gate_vs_measured_dispersion` and the `G-E45c` precedent forbid.

## B.5 What is and is not allowed to come out of run 2

* Run 1's rates are **kept** in the results file and not overwritten. Run 2 writes its own.
* If `G-E44b2` fires, **its** band is the citable interval for this arm. That is not run 2
  promoting run 1: run 1's gate is malformed, so there is no registered measurement being
  overturned, and the E36 run-2 rule has nothing to protect here.
* The band may still **not** be quoted as a new headline — addendum A's restriction is
  untouched. It replaces a point with an interval on an arm that is named.

## B.6 Prediction, registered

| quantity | prediction |
|---|---|
| C1 | fires, foreign ≈ 100% on the loaded box |
| C2 | fires, system ≈ 45% and foreign ≈ 1-8% during a rep, separated by ~40 points |
| `G-E44b2` | **FIRES** |
| the band | within a few percent of run 1's 98-104 tok/s — the box really was quiet, so the corrected meter should not move the rates at all, only the verdict about them |
| spread | 4-8%, i.e. the same order as run 1's 5.7% and well below E43's 9-22% |

If the band moves a lot, something other than the meter is wrong and this prediction is the
thing that says so.
