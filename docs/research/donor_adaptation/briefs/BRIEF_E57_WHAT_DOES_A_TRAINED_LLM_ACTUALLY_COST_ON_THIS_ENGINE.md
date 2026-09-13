# E57 — WHAT DOES A **TRAINED** LLM ACTUALLY COST ON THIS ENGINE, AND AT WHAT QUALITY?

**Pre-registered 2026-09-13, pushed before the measured runs.** The six single-cell probes in §2
were taken first and are marked as such; everything gated below is unmeasured.

---

## 1. The gap this closes, stated plainly

The goal is *"far funzionare un LLM **già addestrato** sulla nostra architettura"* at 50 tok/s
(good) / 100 tok/s (excellent). Fifty-six experiments in, **every tok/s in this programme is
measured on `e40_r128.bin` — synthetic weights.** E55 produced a careful demonstration table and
had to stamp `WEIGHTS SYNTHETIC` on it.

Meanwhile a **real trained model has been running on this engine since E6**, with token-for-token
greedy parity against HuggingFace (`G-E53c2`: A1 160/160), and it has been used only as a
*correctness control*. **Its speed has never been measured, never tabulated, and never compared
to the synthetic number the programme quotes.**

That is not a small omission. It is the difference between "the architecture can be fast" and
"an already-trained LLM is fast on it", and only the second is the goal.

## 2. Six probes, `ALREADY MEASURED`, one cell each at `--bench 160`

| weights | quant | head | GB | tok/s | `ffn~` ms/tok |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | fp32 | fp32 | 1.98 | 18.98 | 33.81 |
| Qwen2.5-0.5B | packed | fp32 | 0.72 | 42.83 | 7.64 |
| Qwen2.5-0.5B | packed | **ternary** | 0.79 | **81.28** | 7.70 |
| Qwen2.5-1.5B | fp32 | fp32 | 6.17 | 6.08 | 121.82 |
| Qwen2.5-1.5B | packed | fp32 | 1.59 | 18.32 | 25.85 |
| Qwen2.5-1.5B | packed | **ternary** | 1.71 | **29.14** | 25.80 |
| *`e40_r128` `--carve-k 3`* | *ternary* | — | *5.11* | *119.57 (E55)* | — |

**Two things fall out of these numbers before any gate runs.**

**(a) The head is 47% of the token at 0.5B and 37% at 1.5B.** Ternarising it is worth **1.90×**
and **1.59×** while `ffn~` does not move (7.64 → 7.70, 25.85 → 25.80). Qwen2.5 ties its embedding
to its output head, so an fp32 head is `151936 × 1536 × 4 B = 933 MB` streamed **per token** — at
this box's measured ~30 GB/s (Probe-3) that is ~31 ms of a 54.3 ms token, and the arithmetic
agrees with the measurement.

**(b) The synthetic 10B shape is 6.5× faster than a real model 6.7× smaller.** That is not an
error and not cheating — it is `--carve-k 3` activating **1.17% of the FFN**. But it means the
programme's headline rate is produced *entirely by sparsity that no trained model has yet been
shown to survive* (E37's verdict on post-hoc conversion was **NO**).

## 3. The thing that makes this experiment necessary

Cross-referencing §2 against E6's stored quality scores:

| arm | tok/s | HF greedy match |
|---|---|---|
| 0.5B fp32 | 18.98 | **160/160** — E6 A1, the known-positive |
| 0.5B packed, fp32 head (`tq`) | **42.83** | **NEVER MEASURED** |
| 0.5B packed + ternary head (`tqh`) | 81.28 | **3/160** — E6 A2, a *planted control* |
| 1.5B fp32 | 6.08 | NEVER MEASURED |
| 1.5B packed, fp32 head (`tq`) | **18.32** | **NEVER MEASURED** |
| 1.5B packed + ternary head (`tqh`) | 29.14 | **10/160** — E6 A3, a *planted control* |

**The two fastest real arms are E6's deliberately-degraded controls.** They are fast and they
emit almost entirely different tokens from HuggingFace. The only arm with proven parity is the
*slowest* one at 0.5B.

**And the `tq` arms — ternary body, fp32 head — have never been scored.** They sit at 42.83 and
18.32 tok/s, between a proven-correct 18.98 and a known-broken 81.28. *Whether a trained LLM runs
correctly on this engine above 20 tok/s is an open question, and it has been open since E6
without anyone noticing.*

## 4. Design

**Quality** reuses E51's `score_against_ref` unchanged, against E6's stored HuggingFace
references — 5 prompts × 32 new tokens, greedy. The references are keyed by *model name*, not by
quantisation, so `Qwen/Qwen2.5-0.5B`'s reference scores every 0.5B arm. **No new scoring code is
written**, so E57 cannot drift away from E6/E50/E51/E53.

**Speed** reuses E55's method: discarded warm-up cell, rotated arm order, quantile intervals,
per-cell occupancy + timestamp + clock, `k = 15` per arm at `--bench 160`.

## 5. Gates

### `G-E57a` — the scorer must fire before any null counts
> A1 (`05b_f32`) must score **160/160**, A2 (`05b_tqh`) **3/160** and A3 (`15b_tqh`) **10/160**,
> reproducing E6 exactly. Any deviation stops the experiment.

Three planted controls: a known-positive and two known-negatives at different degradation levels,
all with numbers fixed by an experiment run seven days ago. This is the strongest control set in
the programme and it costs nothing because it already exists.

### `G-E57b` — the unscored arms
> For `05b_tq`, `15b_tq` and `15b_f32`: matched tokens out of 160, **and the position of the
> first divergence per prompt**. Greedy divergence compounds, so a match *count* alone hides
> whether an arm failed at token 1 or token 31.
>
> **`FAITHFUL`** only at **160/160**, the same bar `G-E53c2` had to clear. Anything else is
> reported as its count and is **not** called faithful.

### `G-E57c` — the speed table
Median, p25, p75 and interquartile width per arm, judged by **`G-E55a2`** — the median of
`|half − full| / full` over **200** random splits against a 20% bar, registered in E55 A.2 as a
variance reduction after `G-E55a`'s single split was shown to be a coin flip (P(PASS) 38–60%).
These are fresh cells, which is the only condition under which E55 A.2 permits it.

### `G-E57d` — the joint claim, which is the whole point
> An arm **DEMONSTRATES THE TARGET** only if it is `FAITHFUL` **and** its p25 is ≥ 50 tok/s.
> **DEMONSTRATES EXCELLENT** at ≥ 100 tok/s.
>
> Speed from a non-faithful arm and quality from a slow one may **not** be combined into a claim.

That last sentence is the entire reason this brief exists.

## 6. Predictions

| | prediction |
|---|---|
| `G-E57a` | all three reproduce exactly — they are stored constants |
| `05b_tq` quality | **160/160 FAITHFUL.** The body is ternary in both A1-adjacent arms and E6 showed ternarising the *body* is survivable; it is the *head* that breaks A2. |
| `15b_tq` quality | **160/160 FAITHFUL**, same reason |
| `15b_f32` quality | 160/160 |
| `G-E57d` at `05b_tq` | **DEMONSTRATES THE TARGET** — 42.83 is below 50 on one cell, but that cell was unreplicated; I expect p25 in **44–50** and therefore a **narrow miss** |
| `G-E57d` at `15b_tq` | below target — p25 near 18 |
| best faithful arm overall | **`05b_tq`, 42–48 tok/s** |

**I am predicting that nothing demonstrates the target.** If `05b_tq` is faithful at ~43 tok/s,
the honest headline becomes *"a trained LLM runs correctly on this engine at 43 tok/s at 0.5B,
14% short of the good bar, and the 50 tok/s claim rests on sparsity no trained model has yet
survived."* That is a worse headline than the programme has been carrying and it is the one the
evidence would support.

## 7. What this does not do

It does not test a 10B trained model — none exists locally, and acquiring one is a separate
decision with a real download cost. It does not test the carve on trained weights (that is H1).
It does not change the engine. **It measures what is already on this disk and has never been
read.**

---

# ADDENDUM A — THE THREE CONTROLS REPRODUCE, THE PREDICTION IS WRONG IN THE OPPOSITE DIRECTION, AND NO TRAINED MODEL RUNS FAITHFULLY ABOVE 19 tok/s

Run: **30 greedy generations + 90 timed cells + 1 discarded warm-up, 16.1 minutes measured**
(runner start 20:30:25, last timed cell at 966.2 s; quality phase 1.7 min, speed phase 14.4 min).
`CONFIG attn=avx4 attnr=none fexp=libm mvacc=4 threads=6 quant=ternary`, printed per cell.
Weight files were warm in the page cache: a 6.17 GB load costs ~2 s of a 27.65 s cell, not the
~50 s a cold read from the USB HDD would cost.

## A.1 `G-E57a` FIRES — all three stored controls reproduce E6 exactly, seven days apart

| arm | matched | E6 | |
|---|---|---|---|
| A1 `05b_f32` | **160/160** | 160 | REPRODUCED |
| A2 `05b_tqh` | **3/160** | 3 | REPRODUCED |
| A3 `15b_tqh` | **10/160** | 10 | REPRODUCED |

The scorer fires on a known-positive and on two known-negatives at different degradation
levels. The nulls below therefore count.

## A.2 `G-E57b` — and the prediction was wrong in the *opposite* direction

| arm | matched | first divergence, per prompt | verdict |
|---|---|---|---|
| `05b_f32` | 160/160 | never | **FAITHFUL** |
| `05b_tq` | **3/160** | 0, 0, 1, 0, 0 | NOT FAITHFUL |
| `05b_tqh` | 3/160 | 0, 0, 1, 0, 0 | NOT FAITHFUL |
| `15b_f32` | **160/160** | never | **FAITHFUL** |
| `15b_tq` | **12/160** | 0, 0, 0, 0, 0 | NOT FAITHFUL |
| `15b_tqh` | 10/160 | 0, 0, 0, 0, 0 | NOT FAITHFUL |

§6 predicted `05b_tq` and `15b_tq` at **160/160 FAITHFUL**, on the reasoning that "the body is
ternary in both A1-adjacent arms and E6 showed ternarising the *body* is survivable; it is the
*head* that breaks A2". **That is backwards.** The ternary **body alone** diverges at **token 0**
in four of five prompts at 0.5B and in **five of five** at 1.5B. Ternarising the head on top of
it changes the count by 0 at 0.5B and by 2 at 1.5B.

**The check that number needed.** `05b_tq` and `05b_tqh` returned the *same* count *and* the
same divergence positions, which is the shape a code-path defect makes. Re-generating both and
comparing the two arms **to each other**:

    05b_tq   vs 05b_tqh  : 0 of 5 prompt sequences IDENTICAL,  28 of 160 tokens equal
    15b_tq   vs 15b_tqh  : 1 of 5 prompt sequences IDENTICAL,  81 of 160 tokens equal

So the coincidence was in the *counts*, not in the *sequences* — the two arms are genuinely
different models. The second row is the interesting one: at 1.5B the two ternary arms agree with
**each other** on 81/160 tokens while agreeing with HuggingFace on 12 and 10. **Ternarising the
body moves the model to a common different place, and the head is a second-order perturbation
on top of it.**

## A.3 `G-E57c` — the speed table

All-cell analysis (see A.6 for why this is the honest column), with bootstrap 95% intervals from
2000 resamples, reported as **data, not as a gate**:

| arm | cells | **p25** | 95% CI | median | 95% CI | IQR |
|---|---|---|---|---|---|---|
| `05b_f32` | 15 | **19.09** | [18.55, 19.47] | 19.47 | [19.12, 19.54] | 2.29% |
| `05b_tq` | 15 | **43.69** | [42.44, 44.13] | 44.13 | [43.86, 44.18] | 1.20% |
| `05b_tqh` | 15 | **83.41** | [82.37, 84.91] | 84.88 | [83.31, 85.76] | 2.62% |
| `15b_f32` | 15 | **6.25** | [6.02, 6.28] | 6.27 | [6.24, 6.28] | 0.56% |
| `15b_tq` | 15 | **19.15** | [18.75, 19.23] | 19.19 | [19.16, 19.45] | 1.35% |
| `15b_tqh` | 15 | **29.18** | [28.85, 30.03] | 30.03 | [29.39, 30.37] | 3.58% |

The single-cell probes in §2 were 18.98 / 42.83 / 81.28 / 6.08 / 18.32 / 29.14. **Fifteen
replicated cells reproduce every one of them to within 2.7%**, so §2's probes were not flukes —
they were just uninterval'd.

## A.4 `G-E57d` — the joint claim

> **NO ARM DEMONSTRATES THE TARGET.**
> The fastest **FAITHFUL** arm is `05b_f32` at p25 **19.09 tok/s** (160/160) — **38% of the
> 50 tok/s bar** and 19% of the 100 tok/s bar.
> The fastest arm overall is `05b_tqh` at p25 **83.41 tok/s** and it is **NOT FAITHFUL** (3/160).

That is the sentence §5 was written to force, and the numbers arrived in exactly the
configuration it was written to prevent being combined: **the speed is on one arm and the
correctness is on another, and they are 4.4× apart.**

§6 predicted `05b_tq` FAITHFUL at p25 44–50 and therefore a narrow miss. The **rate** was right
(43.69, inside the predicted band) and the **quality** was wrong by 157 tokens. The registered
sentence — *"I am predicting that nothing demonstrates the target"* — holds, but for a
completely different reason than the one given.

## A.5 `G-E55a2` REFUSED all six arms, and it is MALFORMED — its verdict depends on the rep count and on nothing else

Every arm came back `UNSTABLE`, on IQRs of **0.56% to 2.29%** — tighter than any window E55
measured, and E55 *passed* two windows at 1.43% and 2.34%.

    A1_05b_f32   IQR 2.29% on 15 reps; median |half-full|/full over 200 splits = 35%
    05b_tq       IQR 1.20% on 15 reps;                                          75%
    A2_05b_tqh   IQR 1.31% on  8 reps;                                          23%
    15b_f32      IQR 0.56% on 15 reps;                                          29%
    15b_tq       IQR 1.35% on 15 reps;                                          27%
    A3_15b_tqh   IQR 1.21% on 10 reps;                                          37%

A gate that refuses data four times tighter than data it previously passed is making a
statement about something other than the data. **P(`G-E55a2` = PASS) over 300 independent
draws from a clean Gaussian population, at four true dispersions spanning 30×:**

| true σ | k=8 | **k=15** | k=25 | k=40 | k=80 |
|---|---|---|---|---|---|
| 0.1% | 2% | **24%** | 69% | 94% | 100% |
| 0.5% | 3% | **22%** | 70% | 93% | 100% |
| 1.0% | 1% | **28%** | 71% | 94% | 100% |
| 3.0% | 3% | **22%** | 67% | 93% | 100% |

**The columns are flat and the rows are identical.** The verdict is a function of `k` alone and
is completely blind to the dispersion it claims to gate — because `|IQR(half) − IQR(full)| /
IQR(full)` is the *relative sampling error of the IQR estimator*, and that is **scale-free**: it
depends on how many points you have, never on how tight they are. At the `k = 15` this
experiment registered, the gate says PASS **22–28% of the time no matter how good the
measurement is**.

**This is the same defect as `G-E44b`, one level up.** E55 A.2 found that `G-E55a`'s single
split was a coin flip and registered `G-E55a2` as the repair — *"a variance reduction on the
same statistic"*. The repair is real: it removed the **split** noise. But **the statistic was
the wrong one**, so the **sample** noise walked straight through it. I measured the noise of the
gate I was replacing and did not measure the noise of the gate I replaced it with, in the very
addendum that established the principle.

**Handling, and what is NOT done:**

* `G-E55a2` is recorded **MALFORMED**, not failed (E4 precedent). It cannot be answered at the
  `k` it was registered at by any measurement of any quality.
* **E57 run 1 is not re-labelled** (E36 run-2 rule). The `NO CLAIM (interval unstable)` verdicts
  stand as printed, and A.3's quantiles are reported as data with bootstrap intervals beside
  them — which is what a refused gate leaves behind.
* **The conclusion does not depend on it.** `G-E57d` turns on quality, and on 19.09 against a
  bar of 50 — a factor of **2.6**, against measured IQRs under 2.3%. No interval gate changes
  that verdict in either direction.
* **The replacement is not chosen here**, because choosing it after seeing which arms it would
  pass is the move this programme exists to prevent. What is registered is the *requirement*:
  the gate must watch **the quantity actually quoted** (the p25 / median), not the IQR, and it
  must have a bar that a tight sample can clear at its own `k`. The bootstrap interval in A.3 is
  a candidate and is reported here only as data.

## A.6 The occupancy bar rejected the memory-bound arms almost entirely, and that is E56's answer arriving early

| arm | cells clearing `OCC_BAR = 4.39` | median foreign | median clock | rate |
|---|---|---|---|---|
| `15b_f32` | **0 of 15** | 7.20% | 107.0% | 6.27 |
| `05b_f32` | **1 of 15** | 6.19% | 107.0% | 19.47 |
| `15b_tq` | 3 of 15 | 4.85% | 104.9% | 19.19 |
| `05b_tq` | 4 of 15 | 5.14% | 105.6% | 44.13 |
| `05b_tqh` | 8 of 15 | 4.17% | 104.2% | 84.88 |
| `15b_tqh` | 10 of 15 | 3.96% | 103.7% | 30.03 |

**Pearson r(rate, foreign) = −0.651 across the six arms.** The slower the arm, the more
"foreign" occupancy it is charged — and the two fp32 arms, which stream the most bytes per
token, are charged the most and are rejected almost completely.

This is not contention. Nothing else was running. It is exactly the mechanism E55 A.3 proposed
and registered as E56: **`foreign = system − child` counts kernel work done *on behalf of* the
engine** — page faults, zeroing, TLB shootdowns — which `GetProcessTimes` does not attribute to
the child. E55 inferred it from a floor; **E57 measures it as a gradient**, because here the
memory traffic differs by a factor of 8.6 between arms while the box is equally idle.

**Consequence, stated plainly: `OCC_BAR` is biased against the treatment.** Applying it here
would discard 15 of 15 cells of the slowest arm and 8 of 15 of the fastest, which is a
selection on the very axis being measured. Four of six arms fell below the six-cell minimum and
fell back to the all-cell set automatically; A.3 therefore reports the all-cell analysis for
every arm, and the clean-cell column is in the JSON for anyone who wants it. **The clock witness
agrees and is doing its job**: 107.0% on both fp32 arms against 103.7% on the fastest ternary
one — the memory-bound arms leave the cores idle enough to boost harder.

## A.7 A display defect of mine, found and fixed

The first speed table printed a `cells` column with no indication of **which set** the count came
from. Four of six arms had silently fallen back to the all-cell set, so the column showed 15 for
rows computed one way and 8 or 10 for rows computed the other. The verdicts were unaffected —
both analyses are in the JSON and they agree to 2.8% at worst — but a column labelled one thing
and holding another is the same defect class as `G-E53f`'s `p = nan`. The runner now names the
analysis per row and says what `ALL` means.
## A.8 Scorecard

| §6 prediction | measured | |
|---|---|---|
| `G-E57a`: all three controls reproduce | 160/160, 3/160, 10/160 — exact | **right** |
| `05b_tq` quality 160/160 FAITHFUL | **3/160**, diverges at token 0 | **WRONG, and backwards** |
| `15b_tq` quality 160/160 FAITHFUL | **12/160**, diverges at token 0 | **WRONG, and backwards** |
| `15b_f32` quality 160/160 | **160/160** | **right** |
| `05b_tq` p25 in 44–50 | **43.69** [42.44, 44.13] | **just outside**, 0.7% low |
| `15b_tq` p25 near 18 | **19.15** | **right** |
| best faithful arm `05b_tq` at 42–48 | best faithful arm is `05b_f32` at **19.09** | **WRONG** |
| *"nothing demonstrates the target"* | nothing demonstrates the target | **right, wrong reason** |

**4 right, 4 wrong.** The rates were predicted well (two of three inside or within 1% of the
band); **every quality prediction about a ternary arm was wrong, and wrong in the direction
that flattered the programme.** That is the fourth experiment in a row with the same signature —
directions right, magnitudes wrong — except that here the error is not a magnitude. It is a
mechanism: I had the wrong organ.

## A.9 What this changes, and what is now the actual blocker

**The good half, and it is newly demonstrated, not assumed:** a trained LLM runs on this engine
**token-for-token identically to HuggingFace, at two different model sizes** — `05b_f32`
160/160 and `15b_f32` 160/160, the second scored here for the first time. The engine is not
wrong. That part of the goal — *far funzionare un LLM già addestrato* — is met, in fp32.

**The blocking half:** it is met only at **19.09 tok/s** (0.5B) and **6.25 tok/s** (1.5B), and
**every configuration that is fast is a configuration that is broken.**

    FAITHFUL  and slow :  05b_f32  19.09    15b_f32   6.25
    FAST      and broken:  05b_tqh 83.41    05b_tq   43.69    15b_tqh 29.18    15b_tq 19.15

**The programme's headline rate has, all along, been standing on the broken side of this
table.** E55's 118.47 / 98.77 / 79.14 / 49.57 are ternary **and** `--carve-k 3` — ternary
quantisation plus 1.17% FFN activation — measured on synthetic weights. E57 is the first
measurement of what those two treatments do to a **trained** model's output, and the answer for
the first of them alone is: **it diverges at the first token.**

**That is not an engine defect and it should not be reported as one.** It is the known cost of
**post-hoc** ternarisation, and E37 already returned `NO` on post-hoc conversion. What E57 adds
is the number and the *organ*: the body, not the head, and at token 0, not by drift.

**So the route to the goal narrows to one thing, and it is a training question, not an engine
question.** A trained model that is fast on this engine must be **ternary by training**
(quantisation-aware), not ternary by conversion. Everything the engine can do has now been
measured on both sides:

| | trained weights | synthetic weights |
|---|---|---|
| **fp32** | faithful, 19.09 tok/s | — |
| **ternary, post-hoc** | 3/160, 43.69–83.41 tok/s | — |
| **ternary + carve, post-hoc** | never scored | 49.57–118.47 tok/s (E55) |
| **ternary by training** | **never built at donor scale** | — |

The empty cell in the bottom-left is the goal.

## A.10 What is owed

1. **`G-E55a2` is malformed and the programme currently has no working interval gate.** The
   requirement is registered in A.5; the replacement is not chosen here and must be registered
   before the run that uses it.
2. **E56 is now over-determined** — A.6 turns E55's floor argument into a gradient with
   `r = −0.651`. `OCC_BAR` must be re-derived against the foreign floor measured *with the
   engine running*, and `system − child` decomposed into contention versus
   kernel-on-behalf-of-the-child. Until then the bar is biased against memory-bound arms.
3. **A `tq` arm has never been given a continuous quality metric.** `3/160` is a discrete count
   at the top of a ladder; it cannot distinguish "lossy but functioning" from "broken". A BPB
   reading on the same ids file, across `f32` / `tq` / `tqh`, separates those two and is owed
   before anyone argues about how much QAT would have to recover.
4. **The 10B question is untouched and is now sharper, not vaguer**: no trained 10B model exists
   locally, and E57 has just measured that the cheap route to making one fast — convert it —
   destroys it.

---

# ADDENDUM B — CORRECTION: §3'S PREMISE WAS FALSE. E17 SCORED THESE ARMS SIX DAYS AGO, AND E57'S QUALITY PHASE IS A REPLICATION, NOT A DISCOVERY

**This addendum retracts a claim in §3 and in addendum A of this brief, in the INDEX headline,
in `SPEED_LEDGER` §56 and in the commit message that carried them.**

## B.1 What I wrote, and what was already in the repository

§3 of this brief says, in bold, of `05b_tq` / `15b_tq` / `15b_f32`:

> *"**And the `tq` arms — ternary body, fp32 head — have never been scored.** … Whether a
> trained LLM runs correctly on this engine above 20 tok/s is an open question, and it has been
> open since E6 without anyone noticing."*

**It was not open, and someone had noticed: me, on 2026-09-07.**
`BRIEF_E17_DOES_THE_HEAD_RANK.md` registered those exact three artifacts as arms `H0c`, `H1`
and `H2` and `results/e17_head_rank.json` holds their scores:

| E17 arm | artifact | E17, 2026-09-07 | E57, 2026-09-13 |
|---|---|---|---|
| `H0a` | `qwen25-05b_f32.bin` | 160/160 | **160/160** |
| `H0b` | `qwen25-15b_tqh.bin` | 10/160 | **10/160** |
| `H0c` | `qwen25-15b_f32.bin` | 160/160 | **160/160** |
| `H1` | `qwen25-15b_tq.bin` | 12/160 | **12/160** |
| `H2` | `qwen25-05b_tq.bin` | 3/160 | **3/160** |

And `results/e17_head_vs_twin.json` holds the arm-versus-twin comparison that addendum A.2
presents as "the check that number needed":

| | E17 | E57 |
|---|---|---|
| `15b_tq` vs `15b_tqh` | 81/160, first diff [0, 1] | **81/160** |
| `05b_tq` vs `05b_tqh` | 28/160, first diff [0, 0] | **28/160** |

**Every single number matches.** I ran, as a new experiment, a measurement that was already in
`results/`, and wrote a pre-registration whose motivating claim contradicted a brief in the same
directory.

## B.2 What this costs, itemised

**Retracted:**

* §3's *"never been scored"* / *"open since E6 without anyone noticing"* — **false**.
* The INDEX headline's *"`15b_f32` is scored for the first time"* — **false**; E17 `H0c`, and
  E17's own brief calls it a "NEW cell", so the phrase was even taken from the right place and
  attached to the wrong experiment.
* Addendum A.2's framing of the twin comparison as a check *I* devised in response to a
  suspicious coincidence. I did devise it in response to the coincidence — I simply did not know
  it already existed, which is worse, not better.
* The commit message's implication that the quality half of E57 is new.

**What stands, and is not diminished:**

* **The speed table is new and E17 explicitly could not produce it.** E17 §7: *"No speed number
  moves. Nothing here is timed and nothing may be quoted from it… Decode rates are recorded by
  the engine and are **explicitly not quotable** — the machine's idleness is not being controlled
  for and a contended timing is not a timing."* E57's 90 replicated cells, occupancy, timestamps,
  clock witness and bootstrap intervals are the thing E17 refused to claim.
* **`G-E57d` — the joint claim — is new**, because it is the first time quality and rate are
  measured in one run and a rule forbids crossing them.
* **The replication itself is worth having, and is stronger evidence than a single reading.**
  E17 ran on `donor_engine.exe`/`donor_engine_e13.exe`; E57 ran on `donor_engine_e53.exe`, which
  carries E49's `avx4` attention kernel and E53's `vexpf8` exponential. **Five quality readings
  and two twin comparisons are bit-stable across two engine builds and six days.** That is a
  cross-version parity result nobody registered and it should be recorded as one.
* The instrument findings (A.5 `G-E55a2` malformed, A.6 `OCC_BAR` biased) are untouched.

**And the prediction is worse than A.8 recorded.** §6 predicted `05b_tq` and `15b_tq` at
**160/160**. They were **published at 3/160 and 12/160 in this repository when I wrote the
prediction.** A.8 scored that as "WRONG, and backwards". It is not merely wrong: **it was
contradicted by a file in `results/` at the moment of writing.** A pre-registration that
contradicts the record is not a prediction, it is a failure to read.

## B.3 Cause, and the rule that follows

Not a slip. **I asserted the absence of a measurement without searching for it.** The search that
would have caught it is `grep -rn "05b_tq" docs/ benchmarks/*/results/` and it takes four
seconds; I ran it *after* the experiment, while looking for a BPB baseline, and it returned E17
immediately.

This is the same shape as the fabricated duration of E53: **a statement about the CONDUCT of the
programme — what has been done, how long something took — rather than about its RESULTS, and
those statements pass through every gate here untouched.** Gates check numbers against
predictions. Nothing checks a sentence that says "this has never been measured".

**Registered rule, and it is cheap enough to be unconditional:**

> **A brief may not claim that something is unmeasured until the claim has been searched for by
> artifact name across `docs/` and `*/results/`, and the search is named in the brief's
> "Checked, not assumed" section.** E17 has such a section; E57 does not, and that is the
> difference between the two briefs.

## B.4 What E57's one-line finding actually is, restated honestly

> **A trained LLM runs on this engine at 19.09 tok/s with its output identical to HuggingFace,
> and at 43.69–83.41 tok/s with its output destroyed. Both halves of that sentence were known
> separately; E57 is the first time they are in the same table, with intervals, and with a rule
> forbidding them from being combined.**

The quality numbers are E17's, replicated. The rates, the intervals and the prohibition are
E57's.

---

# ADDENDUM C — THE CONTINUOUS PARTNER, READ OFF PUBLISHED RESULTS: THE BODY IS THE ORGAN, 0.5B TERNARY IS NOT A MODEL, AND THERE IS NO SCALE TREND THAT RESCUES CONVERSION

**Zero new compute.** A.10 item 3 asked for a BPB reading to separate "lossy but functioning"
from "broken", because `3/160` is a discrete count at the top of a ladder. **Applying addendum
B's own rule before running anything, I searched `*/results/` by artifact name first — and the
reading already exists**, in `density/results/e1_bpb_through_engine_*.json` (E1), on one slice
(`sha a1a48dc9…`, 24 × 512, 12264 predicted tokens), one rule (`R3`), `fold=none`, engine and
PyTorch agreeing to 1.5e-05. A run of mine that would have re-measured it was **killed 25
minutes in** when the search returned E1.

## C.1 The table that was already there

Chance on this slice is **4.069819 BPB** (E16, `V = 151936`).

| model | arm | BPB | damage vs its own fp32 | vs chance |
|---|---|---|---|---|
| Qwen2.5-0.5B | fp32 | **0.871810** | — | −3.198 |
| Qwen2.5-0.5B | `tq` (ternary body) | **4.509164** | **+3.637354** | **+0.439345 ABOVE** |
| Qwen2.5-0.5B | `tqh` (+ ternary head) | 4.531234 | +3.659424 | **+0.461415 ABOVE** |
| Qwen2.5-1.5B | fp32 | **0.702319** | — | −3.368 |
| Qwen2.5-1.5B | `tq` | **3.484253** | **+2.781934** | −0.585566 |
| Qwen2.5-1.5B | `tqh` | 3.475707 | +2.773388 | −0.594112 |

## C.2 It answers A.10 item 3, and the answer is "broken", not "lossy"

**At 0.5B the ternary model is ABOVE chance — by 0.439 BPB with the fp32 head and 0.461 with
the ternary one.** A model that scores worse than the uniform distribution over its own
vocabulary is not a degraded model; **there is nothing left to match**, and that is why `3/160`
and why the divergence is at token 0. E14 §3 asks for a continuous partner to every discrete
score, and here the partner does not merely agree, it **reclassifies the finding**: `3/160` is
not "very lossy", it is "not a model".

At 1.5B the same conversion lands **0.586 below chance**. It predicts — badly, at 4.96× its
fp32 BPB — but it predicts. **So the same treatment produces a non-model at 0.5B and a weak
model at 1.5B**, which is the first thing in this programme to give the discrete `3/160` vs
`12/160` gap a mechanism instead of a magnitude.

## C.3 The body is the organ, confirmed on the continuous metric — and at 1.5B the head is NEGATIVE

| | `tq` → `tqh` |
|---|---|
| 0.5B | **+0.022070 BPB** (0.6% of the body's damage) |
| 1.5B | **−0.008546 BPB** — ternarising the head **IMPROVES** BPB |

A.2 concluded from greedy counts that the body is what breaks the model and the head is a
second-order perturbation. The continuous metric says the same thing and then says something
the counts could not: **at 1.5B the head's ternarisation is not even a perturbation in the
damaging direction.** The whole of the damage, on both models, is the body.

This also retires a live suspicion. §2 attributed `05b_tqh`'s 1.90× speed advantage to
ternarising a 933 MB tied embedding/head — that is a **bandwidth** statement and it stands
(7.64 → 7.70 ms `ffn~`, the head is 47% of the token). What does **not** follow, and what I had
half-assumed, is that the head is where the quality went. It is not, at either size.

## C.4 And there is no scale trend that rescues conversion — the only bigger model on the disk is above chance

The obvious hope after C.1 is that damage shrinks with scale (+3.637 at 0.5B → +2.782 at 1.5B,
**−23.5%** for 3×), and that at 10B post-hoc conversion would simply work. **The third point
kills it.** E16's `B3` is the fold-matched, rule-matched 7B:

| model | params | damage vs own fp32 | vs chance |
|---|---|---|---|
| Qwen2.5-**0.5B** | 0.49 B | +3.637354 | **+0.439 above** |
| Qwen2.5-**1.5B** | 1.54 B | **+2.781934** | −0.586 below |
| Qwen2.5-**Coder-7B** (`B3`, R3, `fold=none`) | 7.07 B | **+3.494298** | **+0.098 above** |

**Non-monotone, and the largest model is the second-worst.** `B3` is `R3` and `fold=none`,
matched to the other two on rule and on fold — E16 built it for exactly this comparison — so
the confound that remains is **family**: `Coder-7B` is a code model, not a base model, and
E16 §0 already refused to read `+0.692619` across that boundary as a scale term. **That refusal
is honoured here: no scale law is claimed and no line is fitted through three points, two of
which are the same family.**

What *is* claimed is the negative, and it is enough:

> **Nothing measured in this programme supports the expectation that post-hoc ternarisation
> becomes survivable at larger scale. Two of the three models converted this way score ABOVE
> chance, and the largest of them is one of the two.**

That is the cheapest remaining route to a fast 10B, and it is now closed on evidence rather
than on the E37 verdict alone.

## C.5 What this does to the conclusion

Nothing, and that is the point — it hardens it. A.9's table gains a column:

| | trained weights, greedy | trained weights, BPB | rate |
|---|---|---|---|
| **fp32** | 160/160 | 0.872 / 0.702 | 19.09 / 6.25 |
| **ternary body, post-hoc** | 3/160, 12/160 | **4.509 (above chance)** / 3.484 | 43.69 / 19.15 |
| **+ ternary head** | 3/160, 10/160 | 4.531 / 3.476 | 83.41 / 29.18 |
| **ternary by TRAINING** | — | — | — |

The bottom row is still empty, it is still the goal, and the only evidence that it can be
filled is **H0**, which took a ternarised factored attention from `28/160` to `115/160`
teacher-forced and removed **97.9%** of its BPB damage in 1000 steps. **That is the experiment
to scale, and it needs a GPU session.**

## C.6 Checked, not assumed — the section this brief was missing

Applying addendum B.3's rule retroactively to everything above:

* `grep -rln "qwen25-05b_tq\.bin|qwen25-15b_tq\.bin|qwen25-05b_f32\.bin" */*/results/*.json`
  → returned `e1_bpb_through_engine_qwen25-05b.json`,
  `e1_bpb_through_engine_qwen25-15b_tqtqh_s24.json`, `e17_head_rank.json`,
  `e49_attn_kernel.json`. **Run before writing this addendum; it is what killed the redundant
  BPB job and what produced C.1.**
* `ls docs/research/donor_adaptation/briefs/` → `BRIEF_E16_R3_AT_7B.md`, read before C.4 was
  written; C.4's three-point table is E16's own comparison and its refusal to cross the family
  boundary is E16 §0's, quoted.
* **`BRIEF_E12_TERNARY_COST_VS_SCALE.md` — read AFTER C.4 was drafted, which is the rule being
  broken again in the same addendum that restates it.** The listing above surfaced the filename
  and I wrote C.4 from the filename. See C.7, which is the correction; C.4's numbers stand, its
  claim to originality does not.
* Chance BPB, the slice sha, the rule and the fold were read **out of the result files**, not
  assumed from memory.

## C.7 CORRECTION to C.4, one hour later: E12 asked this exact question, ran, and its verdict is SUSPENDED

**`BRIEF_E12_TERNARY_COST_VS_SCALE.md` is titled *"does the ternarization cost shrink with donor
scale?"* and its §1 calls it "the oldest open item in this programme, and after E10 the only one
that can still move the goal".** It ran: a 2161 s sweep over **0.5 B / 1.5 B / 3 B** — a third
**same-family** point that C.4 said did not exist.

| donor | BPB fp32 | BPB ternary | **dBPB** | planted `Z` | real `F` | `Z-F` |
|---|---|---|---|---|---|---|
| 0.5 B | 0.871795 | 4.587453 | **3.715658** | 3.693528 | 3.674503 | +0.019 |
| 1.5 B | 0.767595 | 5.505834 | **4.738239** | 4.001257 | 3.309099 | +0.692 |
| 3 B | 0.724450 | 5.734699 | **5.010249** | 3.714379 | 5.203945 | **−1.490 FAIL** |

**E12's verdict is `CONTROL-FAILED`, and §9 suspends even that.** At 3 B the mis-specified rule
(`random_sign`) produced a **better** model than the real one by 1.49 BPB, and converting *more*
organs did *less* damage — two monotonicity violations in one cell — so by E12 §4's own rule
the cell does not count and `r = 1.3484` (`COST-GROWS`) **is not reported as a result**.

**What this does to C.4:**

* **C.4's numbers are untouched.** They are E1's and E16's, on rule `R3`, `fold=none`, one
  slice. E12 used a **different conversion** (`t1_ternarize.ternarize`, BitLinear158) and a
  different organ set — its 0.5 B ternary reads 4.587453 against E1's 4.509164 — so the two
  tables are not the same estimand and neither corrects the other.
* **C.4's claim to be asking a new question is wrong**, exactly as §3's was. The question is the
  programme's oldest open item.
* **And the two tables point in OPPOSITE directions.** C.4's same-family pair has damage
  *falling* 23.5% from 0.5 B to 1.5 B; E12's same-family pair has it *growing* 27.5%
  (3.716 → 4.738). Same two donors, different rule and organ set, opposite sign. **That is a
  stronger negative than either table alone**: the sign of the scale trend is not even stable
  across conversion rules, so no scale argument — in either direction — is available to this
  programme today.

**The negative in C.4 therefore survives and is restated more carefully:**

> **There is no measured, control-passing scale trend for post-hoc ternarisation damage in this
> programme.** The only experiment designed to find one returned `CONTROL-FAILED` at its largest
> cell and is suspended; the two readable pairs disagree in sign; and two of the three `R3`
> models score ABOVE chance, the largest among them. Nothing here supports the expectation that
> conversion becomes survivable at 10 B, and nothing here supports the opposite either. **The
> question is open, it has been open since E12, and it is not what E57 closed.**

**Cause, again.** I ran the search, it returned the filename, and I wrote the paragraph without
opening the file. Addendum B.3's rule says *search by artifact name and name the search*; it
does not say *read what the search returns*, because I did not think that needed saying.
**Amendment to the rule: the search is not complete until every brief it returns has been
opened, and the "Checked, not assumed" entry records what each one said** — not that it exists.

---

# ADDENDUM D — WHICH ARM IS AT THE WALL: THE FAITHFUL ONE, AND ONLY THE FAITHFUL ONE

**Derivation over measured quantities** (`e57_wall.py`, `results/e57_wall.json`). No timing is
taken; the rates are addendum A's medians and the bandwidth bounds are the ledger's
(proj-GEMV streamed floor **37.0 GB/s**, DRAM aggregate **40–44 GB/s**, ledger §1). **Nothing
here is quotable as a rate.**

## D.1 The unit trap this had to avoid first

A file size is not a moved byte. Qwen2.5 **ties** its embedding to its head, so in the `f32` and
`tq` artifacts one matrix is *gathered* once per token (4·D bytes, noise) **and streamed** once
per token as the head. But `tqh` ternarises the head, and the exporter therefore **unties** them:
an fp32 embedding that is only gathered, plus a separate ternary head that is streamed. Charging
that fp32 embedding to a `tqh` token reads **67 GB/s on a machine whose DRAM ceiling is 44** —
the byte-convention law (E31) in its purest form, and it would have been invisible as an error
because 67 is merely *implausible*, not *impossible-looking*.

**The inference is checked, not asserted.** Reconstructing each artifact from E1's stored ternary
code counts:

| arm | layout | ternary codes | reconstruction | residual |
|---|---|---|---|---|
| `05b_tq` | tied | 357,826,560 | 723.5 MB | **+0.21%** |
| `05b_tqh` | **untied** | 493,961,216 | 791.5 MB | **+0.27%** |
| `15b_tq` | tied | 1,310,195,712 | 1588.6 MB | **+0.20%** |
| `15b_tqh` | **untied** | 1,543,569,408 | 1705.3 MB | **+0.22%** |

Every residual is positive and under 0.3% — scales, norms and header, in the right direction and
the right size. And the `tqh` code counts decompose exactly: `493,961,216 − 357,826,560 =
136,134,656 = 151936 × 896`, the head. (The two `f32` rows reconstruct trivially by construction
and are **not** a check; they are in the table for completeness.)

## D.2 The table

| arm | file MB | gathered MB | **moved MB/token** | tok/s | **GB/s** | |
|---|---|---|---|---|---|---|
| `05b_f32` | 1976.1 | 0.0 | **1976.1** | 19.47 | **38.5** | **AT THE WALL** |
| `15b_f32` | 6174.9 | 0.0 | **6174.9** | 6.27 | **38.7** | **AT THE WALL** |
| `05b_tq` | 725.0 | 0.0 | 725.0 | 44.13 | 32.0 | near it |
| `15b_tq` | 1591.8 | 0.0 | 1591.8 | 19.19 | 30.5 | near it |
| `15b_tqh` | 1709.0 | 933.5 | 775.6 | 30.03 | 23.3 | **1.59× of headroom** |
| `05b_tqh` | 793.6 | 544.5 | 249.1 | 84.88 | 21.1 | **1.75× of headroom** |

**A consistency check nobody arranged.** The two fp32 arms differ **3.13× in size** and read
**38.5 and 38.7 GB/s — 0.62% apart.** A bytes-per-token model that is wrong does not agree with
itself across a threefold change of denominator, and the agreement is tighter than the ±5% every
absolute rate in this programme carries.

## D.3 What it says, and it is the sharpest thing E57 produced

**The faithful arm is bandwidth-bound and there is no engine work left on it.** 38.5 and 38.7
GB/s sit **above** the proj-GEMV streamed floor of 37 and inside the 40–44 GB/s DRAM band. A
faster kernel, a better exponential, a smarter attention — E49, E51, E53, all of it — cannot move
`05b_f32` off 19.09 tok/s, because it is not waiting on arithmetic. **The only lever on a
faithful arm is fewer bytes, fewer bytes means quantisation, and quantisation is precisely what
addendum C measured destroying it.** That is the whole of the programme's problem in one
sentence, and it is now measured rather than argued.

**And the broken arms are NOT at the wall**, which is where the remaining engine work is:

| arm | now | at the 37 GB/s floor | |
|---|---|---|---|
| `05b_tqh` | 84.88 | **148.54** | ×1.75 |
| `15b_tqh` | 30.03 | 47.71 | ×1.59 |
| `15b_tq` | 19.19 | 23.24 | ×1.21 |
| `05b_tq` | 44.13 | 51.04 | ×1.16 |

This is consistent with the ledger's own correction after E28 — *"nothing here is at the
bandwidth wall"* — which was made about ternary configurations, and is confirmed here with the
fp32 counter-example that gives it a scale.

**The consequence for the goal is a division of labour, and it is not the one the programme has
been assuming.** Engine work has ~1.6–1.75× left *on the ternary path only*. Quality work has
everything. A healed ternary 1.5B at `15b_tqh`'s byte count, running at the streamed floor,
would read **47.71 tok/s** — 95% of the good bar, from a 1.5B, with the 1.59× coming from work
that has nothing to do with training. **That is the first time the two halves of the goal have
had a common denominator.**

## D.4 Checked, not assumed

* Structural quantities (`V`, `D`, file sizes, ternary code counts) **read** from the artifacts
  and from `e1_bpb_through_engine_*.json`; none typed from memory.
* The tie/untie inference **reconstructed and closed to <0.3%** (D.1) rather than inferred from
  the size ordering alone — which is what first suggested it (`tqh` is *larger* than `tq`, which
  is impossible under a tied layout).
* Bandwidth bounds taken from `SPEED_LEDGER` §1 (DRAM 40–44, proj floor 37.0), not from the
  34.75 GB/s figure quoted in older sections, which §1's correction supersedes.
* `grep -rln "GB/s" briefs/` run first, and per C.7's amended rule the returned probes were
  **opened and read**, not listed:
  * **`probes/E30_IS_THE_ENGINE_AT_THE_WALL.md`** returns `AT-THE-WALL` for `T10-LUTBLK` at
    **33.01 GB/s against a ceiling of 36.30 GB/s measured on this box** (0.909), and carries a
    correction: the *operative* packed path reads 0.639 of that ceiling, so the engine sits
    **1.57× from the wall**, not 1.10×. E30 asks the question of the **engine at a synthetic
    shape**; it had no trained rate to ask it with, because E17 §7 refused to time.
  * **`probes/E31_WHAT_A_GATHERED_BYTE_COSTS.md`** carries two corrections of its own and its
    surviving statement is that the gathered-byte lever is **1.0432 normalised at T10, below the
    resolvable line in five jackknives**. It is about *carved* reads and does not bear on a dense
    stream, which is what every arm here is.
* **A bound disagreement that must be named rather than smoothed.** E30 measured **36.30 GB/s**
  as this box's ceiling in its session; `SPEED_LEDGER` §1 registers a DRAM aggregate of
  **40–44 GB/s** and a proj-GEMV streamed floor of **37.0**. **E57's fp32 arms read 38.5 and
  38.7 — above E30's figure and below the ledger's band.** Exceeding a measured ceiling is a
  reason to check, not to celebrate: the check is D.1's reconstruction plus the 0.62% agreement
  between two arms 3.13× apart, and a dense sequential fp32 stream is a *friendlier* access
  pattern than the packed/LUT reads E30 timed. **The honest statement is therefore "between the
  two bounds this programme has measured, and at or above both estimates of the streamed floor"
  — which is what "at the wall" means here, and the D.3 multipliers use the 37.0 floor, the
  more conservative of the two for that purpose.**
