# E41 — is the PARTITION the lever? **`VERDICT-UNRESOLVABLE`**

> ⚠ **THE FILENAME CARRIES RUN 1'S VERDICT AND THE VERDICT DID NOT SURVIVE ADDENDUM B.**
> The file is not renamed because `SPEED_LEDGER §51`, `INDEX.md` and commit `d136f98`
> already point at this path. **Read §0.0 before anything else in this document.**

**Brief:** `briefs/BRIEF_E41_IS_THE_PARTITION_THE_LEVER.md`, pushed before the runner existed;
**addendum A** (`G-E41C` diagnosed) and **addendum B** (pre-registered re-run) pushed at
`fadeb42`, addendum B before its runner existed.
**Runner:** `benchmarks/donor_adaptation/engine/e41_partition_lever.py`,
`e41_g41c_diag.py`, `e41_addendum_b.py`.
**Results:** `engine/results/e41_partition_lever.json`, `e41_g41c_diag{,_f64}.json`,
`e41_addendum_b.json`. **Logs:** `e41_run.log`, `e41_g41c_diag{,_f64}.log`, `e41_addb.log`.

**QUALITY ONLY, fp32, on a really trained donor. No speed, no synthetic weights.**

---

## 0.0 CORRECTION — the registered verdict is WITHDRAWN, by my own registered rule

**Everything below §0.0 was written and committed (`d136f98`) BEFORE addendum B had printed a
number**, deliberately, so the run-1 narrative could not be tuned to B's outcome. B then printed
its number and **it goes against the verdict.**

Addendum B, pre-registered at `fadeb42`, refit all six routers at the calibration budget the band
boundary was actually measured at (32 sequences, seed 42424) and re-ran the six verdict cells:

```
G-E41D  D0C refit at 32/42424 = 3.597108  vs E38 3.597108  diff -4.73e-07  -> FIRES
ordering run 1      : D0C < PERM < CONC < COACT < STRIPE < RAND
ordering addendum B : COACT < D0C < PERM < CONC < STRIPE < RAND    -> CHANGED
best is COACT at 3.583800  -> crosses the 3.597108 boundary by -0.013308
```

The rule I registered in the brief, verbatim: *"If B's best partition crosses **3.597**, both are
reported and **the verdict cell is declared unresolvable**; in that case E41 does not strengthen
the T4 ask and must say so."*

**So: the verdict cell is UNRESOLVABLE. `PARTITION-IS-NOT-THE-LEVER` is withdrawn. E41 does NOT
strengthen the T4 ask in `COMMUNICATION.md`, and every sentence below that counts it as a fourth
independent measurement of the closed road is WRONG and is corrected in §10.**

Three things must be said in the same breath, and none of them is a hedge:

1. **B's instrument is the known-good one.** `G-E41D` reproduces E38's published `3.597108` to
   **4.7e-07**. The calibration budget was exactly the whole `+0.207` discrepancy §5 reported.
   **B may still not promote anything** — E36's run-2 rule, registered before B existed.
2. **The margin is small and the dispersion was never measured.** `COACT` wins by **0.013308**.
   The only noise estimate on the fitted path is the label-permutation control, `PERM − D0C =
   +3.07e-04`, so the margin is **43×** it — but that bounds *relabelling* noise, not the choice
   of calibration slice, which is the axis that just reversed the ordering. **Neither run
   measured that, and that is the defect that makes the cell unresolvable rather than merely
   overturned.**
3. **Nothing about the goal changes.** `COACT` at 3.583800 is **85.3% of the way from dense to
   chance** and **2.58 BPB above** the usability bar (1.005039). A partition that "helps a
   little" does not make ~6% activation into a model. **What changed is whether E41 gets to
   be counted as evidence — it does not — not what the numbers say about the goal.**

**And the door I thought I had closed is ajar.** §3 argued that what decides the attainable column
is the oracle→fitted gap, i.e. how *predictable* a grouping is from `x`. Addendum B's own
prediction 4 said `COACT` would gain most from more calibration for exactly that reason — **it
gained most (−0.303203, against `RAND`'s −0.0048, which got WORSE) and gained enough to take
first place.** My registered mechanism predicted the thing that broke my verdict.

---

## 0. The one-line answer, and the thing that spoils it

> **Six ways of grouping the same 8,960 neurons, and the best attainable selector at the rate
> 50 tok/s affords is the label family we already had.** `D0C` reads **3.804346** BPB at
> `k = 16`; nothing beats it. Band: **`PARTITION-IS-NOT-THE-LEVER`**.

**And the gate that was supposed to protect that sentence went VOID.** `G-E41C` demanded that
relabelling the 256 group IDs move the oracle by *exactly* zero. At `k = 3` it did. At `k = 16`
it moved by `-1.22e-3`. **Addendum A traces it to two exactly-tied float32 masses out of 344,064
selections** — the oracle is not a single-valued function when two groups weigh the same — but
the gate stays VOID and prediction 1 is scored a MISS. The jitter floor it establishes is
`1.2e-3`; the tightest comparison in this probe is `9.5×` that, and the verdict cell is not an
oracle cell at all.

**A second defect is mine and is reported in addendum B:** the band boundary (`3.597108`) was
measured by E38 with a router fit on **32** calibration sequences, while my `§2` registered the
calibration slice at **8**. Every `fitted` number here is `+0.207238` off its own reference — 29%
of the whole spread the probe measures. **The verdict's direction survives it** (the best
partition *is* the control, so "nothing beats `D0C`" holds whatever the absolute), and addendum B
re-runs the six verdict cells at the reference budget to test whether the *ordering* does.

---

## 1. The gates

| gate | what it demanded | result |
|---|---|---|
| **`G-E41A`** | at `k = E` every partition bit-inert, `|BPB − dense| < 1e-6` | **FIRES**, worst **0.00e+00** across all six |
| **`G-E41B`** | unmasked fp32 = E22's `0.767595 ± 0.001` | **FIRES**, `0.767594964`, diff **−3.59e−08** |
| **`G-E41C`** | `PERM` oracle == `D0C` oracle **exactly** | **VOID** — `k=16` diff `−1.221e-03`, `k=3` diff `+0.000e+00` |

`G-E41A` at *exactly* zero says the same thing it said in E38: labels, partition construction and
the masking hook are inert when nothing is dropped, in all six label families.

### 1.1 Three of E38's published numbers reproduce, free

Nothing in E41 was tuned to make this happen and the harness refit its own routers from scratch:

| quantity | E38 published | E41 measured |
|---|---|---|
| `D0C` oracle, `k = 3` | `4.131817` | **`4.131817`** |
| `D0C` oracle, `k = 16` | `3.449466` | **`3.449466`** |
| grouping swing (oracle, `k = 3`, `D0C` → random) | `+0.0310` | **`+0.031031`** |

The instrument is measuring the published object. What it does *not* reproduce is E38's `fitted`
number — see §5.

---

## 2. The measurement

Qwen2.5-1.5B (`S15`), fp32, `D=1536 F=8960 L=28`, `E=256` groups of 35. Frozen heldout slice
`("heldout", 24, 512, 1234)`, `ids_sha a1a48dc9…`, **51,870 scored bytes**. Dense **0.767595**,
chance **4.069819**; the window between them is **3.302224** BPB wide and every percentage below
is a fraction of it. 5,558 s of CPU, 31 cells.

### 2.1 `k = 16` — 6.25%, what E40 measured 50 tok/s buying at ten billion

| partition | oracle | % of window lost | fitted | % of window lost | oracle→fitted gap |
|---|---|---|---|---|---|
| **`COACT`** | **3.095548** | **70.5%** | 3.887003 | 94.5% | **0.791455** |
| `CONC` | 3.369107 | 78.8% | 3.815911 | 92.3% | 0.446805 |
| **`D0C`** | 3.449466 | 81.2% | **3.804346** | **92.0%** | **0.354880** |
| `PERM` | 3.448246 | 81.2% | 3.804346 | 92.0% | 0.356101 |
| `STRIPE` | 3.813485 | 92.2% | 4.415238 | 110.5% | 0.601753 |
| `RAND` | 3.902486 | 94.9% | 4.530484 | 114.0% | 0.627998 |

### 2.2 `k = 3` — 1.17%, what ~110 tok/s affords

| partition | oracle | vs chance | fitted | vs chance |
|---|---|---|---|---|
| **`COACT`** | **4.125956** | **+0.0561** | **4.522115** | +0.4523 |
| `D0C` | 4.131817 | +0.0620 | 4.574331 | +0.5045 |
| `PERM` | 4.131817 | +0.0620 | 4.574331 | +0.5045 |
| `RAND` | 4.162848 | +0.0930 | 4.691421 | +0.6216 |
| `CONC` | 4.203996 | +0.1342 | 4.874205 | +0.8044 |
| `STRIPE` | 4.254170 | +0.1844 | 4.818690 | +0.7489 |

**Every one of the twelve cells at `k = 3` is above the chance line.** Perfect per-token
selection, on the best of six groupings, at 1.17% activation, is still worse than a coin. E38
said that with one partition; E41 says it with six.

---

## 3. The result that is worth more than the verdict

**The partition is a large lever under the oracle and the ordering REVERSES under the router that
can actually be built.**

| | best | 2nd | 3rd | 4th | 5th |
|---|---|---|---|---|---|
| by **oracle**, `k=16` | `COACT` | `CONC` | `D0C` | `STRIPE` | `RAND` |
| by **fitted**, `k=16` | **`D0C`** | `CONC` | **`COACT`** | `STRIPE` | `RAND` |

`COACT` has the **best ceiling of the six** — 3.095548, a full **0.354** better than `D0C`'s
oracle and 70.5% of the window against `D0C`'s 81.2% — **and the worst realization gap**,
0.791455 against `D0C`'s 0.354880. Grouping neurons that fire together makes the *perfect* choice
much better and the *attainable* choice worse. Rank correlation across the five distinct
partitions is `ρ = 0.60`: the top three are in a different order in the two columns.

**This is E14 §3 again, and E37's `ROUTER-IS-NOT-THE-CONSTRAINT` mirror again, on a third axis.**
A partition that scores better under a perfect selector is not the partition you should ship.
Anyone who prices groupings by their oracle — which is the cheap thing to measure — picks
`COACT` and ships the third-best model.

### 3.1 The grouping swing decays with `k`, like everything else on this axis

Same selector, `D0C` against a random equal partition:

| `k` | activation | oracle swing |
|---|---|---|
| 16 | 6.25% | **+0.453019** |
| 3 | 1.17% | **+0.031031** |

Activation falls `5.3×` and the value of grouping well falls `14.6×`. E38's `static` selector saw
**2.37** at `k = 64` (25%) — a different selector, so it does not belong in the same column, but
it points the same way. **The partition's value peaks where selection's value peaks, at 25%, and
the rate the box affords is far below the knee** — which is E38 §5's shape, measured again on a
parameter E38 could not move.

---

## 4. Why `PARTITION-IS-NOT-THE-LEVER`, in the terms the goal is stated in

The verdict cell — best `fitted` at `k = 16` — reads **3.804346**, against a band boundary of
**3.597108** and a usability bar of **1.005039** (E22's `QO512+V52` arm, which scored 126/160
teacher-forced). The best grouping of six is **3.79 BPB above dense** and **2.80 above the bar**.

| what 50 tok/s buys at 10 B (E40) | what that rate is worth, best of six partitions (E41) |
|---|---|
| **6.0 ± 0.3%** activation | **3.804346** BPB — 92.0% of the way from dense to chance |
| ~1.7% at 100 tok/s | **4.522115** BPB — **above the chance line** |

**The affordable rate and the useful rate still do not overlap.** ⚠ **CORRECTED (§0.0):** the
sentence that stood here — that E37, E38, E40 and E41 are *four* independent measurements of the
same closed road — **is withdrawn.** E41's verdict cell is unresolvable, so it is not one of
them; the count is **three** (E37 conversion, E38 selector, E40 attention shape). What E41 shows
at this rate is still true on both instruments and does not depend on the verdict: at 6.25% the
best of six groupings loses **85.3%** (addendum B) to **92.0%** (run 1) of the dense→chance
window, and at 1.17% every cell is above chance. **Re-grouping moves the number by hundredths
where the goal needs units.**

---

## 5. The instrument defect the brief walked into, and what it does and does not change

`D0C` fitted at `k = 16` reads **3.804346** here and **3.597108** in E38. Same donor, same slice,
same labels, same `Carve`, same `e23_router.fit_routers` — **different calibration budget**: E38's
router came from `e37_fit_routers.py`'s npz, fit on **32** sequences at seed **42424**; my `§2`
registered **8** sequences at seed **424242**, which is `e38_oracle_ceiling`'s constant for its
`static` selector and not for its router.

- **+0.207238**, which is **29%** of the entire fitted spread across partitions (0.726138). The
  instrument difference is bigger than most of the effects the instrument compares.
- **The verdict's direction is not at risk**: the winner *is* `D0C`, the control, so "no partition
  beats the label family we already had" is true on E41's own instrument at any absolute level.
  Re-expressed against E41's own control the bands read `PARTITION-HELPS-A-LITTLE` below 3.804346
  and `PARTITION-IS-NOT-THE-LEVER` at or above it — and the best is exactly 3.804346.
- **The ordering could still depend on it**, and that is what **addendum B** measures: all six
  routers refit at 32/42424, the six verdict cells re-run, `G-E41D` requiring `D0C` to reproduce
  E38's 3.597108 within ±0.001 as a known positive. **E36's run-2 rule governs it: addendum B may
  not promote the registered verdict.**

---

## 6. `G-E41C`, and the hypothesis I got wrong

Full forensics in the brief's **addendum A**. In one paragraph:

The gate went VOID at `k = 16` by `1.22e-3`. My first hypothesis — float32 accumulation order in
`mass.index_add_` — **is wrong and is recorded as wrong**; a synthetic check took two minutes and
the full-slice replay confirms it at 344,064 rows: **group mass is bit-identical under
relabelling, max abs diff 0.000e+00.** The actual cause is that **two rows of 344,064 carried
bitwise-equal float32 mass astride the `k`-th boundary** — `topk` breaks such a tie by index, and
permuting names flips it. Two flipped selections, in layers 1 and 11, cascade into the whole
`1.22e-3`. **The oracle is not single-valued when groups tie**, which is a defect in how I
specified the gate as much as in how the code computes it.

**Repair, verified:** accumulate group mass in float64 → 0 ties, 0 flips, 0 diverging layers on
the same slice. **`e38_oracle_ceiling.py` is NOT edited** — its published numbers must keep
reproducing bit-for-bit — so the repair is specified for the next probe that needs an oracle.

**The gate is not re-run to a pass** (E14 §6, E40 addendum A's precedent). What it leaves behind
is a measured jitter floor of `1.2e-3` on oracle cells: 290–660× smaller than every oracle
difference reported above, and 9.5× smaller than the tightest comparison in the probe
(`D0C` 3.804346 vs `CONC` 3.815911). **The verdict cell is `fitted`, and `PERM fitted` reproduced
`D0C fitted` to all sixteen digits at both `k`** — a planted control passing on the selector that
decides, reported and **not** promoted to a gate, because it was not registered as one.

---

## 7. Predictions — **0 clean HIT, 2 split, 3 MISS**

| # | registered | outcome |
|---|---|---|
| 1 | all three gates fire, `G-E41C` **exactly** zero | **MISS.** `G-E41A` and `G-E41B` fire; `G-E41C` VOID at `k=16` |
| 2 | best partition in `PARTITION-HELPS`, **2.90 BPB** | **MISS**, and by a lot: `3.804346`, band `PARTITION-IS-NOT-THE-LEVER`, registered number off by **+0.904** |
| 3 | `CONC` wins at `k=3`, does not win at `k=16` | **SPLIT, wrong half right.** It does not win at `k=16` ✓ — but at `k=3` it is the **worst** fitted arm (4.874205) and 5th of 6 by oracle. The mechanism I argued (concentrate mass when you may keep almost nothing) is measured **backwards** |
| 4 | `COACT` beats `D0C` at both `k` | **SPLIT, 3 of 4 cells.** Oracle: wins at `k=16` (−0.354) and `k=3` (−0.006) ✓✓; fitted: wins at `k=3` (−0.052) ✓ but **loses at the verdict cell** (+0.083) ✗ |
| 5 | `STRIPE` worst at every `k`, worse than `RAND` | **MISS at both.** `k=16`: `STRIPE` 4.415238 **beats** `RAND` 4.530484, and `RAND` is worst. `k=3`: worst is `CONC` 4.874205 |

**Three of the five misses are the same mistake**: I modelled the partition's value as a property
of how mass is *distributed across groups* (concentrate it — `CONC`; spread it — `STRIPE`;
co-activate it — `COACT`) and predicted the attainable selector would follow. It does not. The
only thing that predicted the `fitted` column would have been the *oracle→fitted gap*, which is a
property of how **predictable from `x`** the grouping is, and I had no model of it at all.

---

## 8. What E41 cannot claim

- **Nothing about ternary.** fp32 throughout, like E38, for E37's reason.
- **Nothing about training.** No gradient was taken. Whether *training into* a partition helps is
  still H0/H1's question and this probe does not touch it.
- **Nothing about 10 B.** One donor at 1.5 B; the activation *fraction* transfers, the BPB does not.
- **Nothing about criteria beyond these five,** and nothing about `E`: 256 groups throughout.
  Group count is a third axis, untouched.
- **Nothing that requires the oracle to be reproducible to the last bit** — `G-E41C` is VOID and
  every oracle number here carries `±1.2e-3`.

## 9. Owed

1. **The oracle's float64 repair, carried into whatever probe next needs one** — specified and
   verified here, deliberately not retrofitted into `e38_oracle_ceiling.py`.
2. **A criterion sweep for the *selector*** — E38 §7 item 1, still owed; E41 moved the partition,
   which was the other half.
3. **The group count `E`.** Three axes exist (partition, selector, `E`); two have now been swept
   and `E = 256` has never been varied at all.
4. **A grouping that is optimised for PREDICTABILITY, not for mass — PROMOTED to the top of
   this list by addendum B.** §3 says the quantity that decides the fitted column is the
   oracle→fitted gap, and none of the five criteria targeted it. `COACT` is a crude 1-D proxy
   for co-activation and **at the reference calibration budget it already beats the control**.
   This is the version of "the partition is the lever" E41 leaves not merely alive but ahead.
5. **The dispersion of the fitted column over the CALIBRATION SLICE.** The single axis that
   reversed the ordering was never replicated: two budgets, one seed each, no error bar. **This
   is what makes the cell unresolvable rather than decided, and it is the first thing E42 must
   measure.**

---

## 10. ADDENDUM B — the measurement that withdrew the verdict

Pre-registered at `fadeb42`, **before this runner existed**, with the outcome rule written in
advance. Runner `e41_addendum_b.py`, 2,501 s. Only the router's calibration budget changes:
partitions are rebuilt from the **same registered 8-sequence statistics** run 1 used, so
`CONC`/`STRIPE`/`COACT` are the same neuron sets; the six routers are refit on **32 sequences,
seed 42424** — `e23_router`'s own constants, the budget E38's boundary was fit at.

### 10.1 `G-E41D`, a known positive, fires to seven digits

```
G-E41D  D0C refit at 32/42424 = 3.597108  vs E38's published 3.597108  diff -4.73e-07  -> FIRES
```

**The calibration budget was the entire `+0.207238` discrepancy of §5, and nothing else was.**
Same donor, same slice, same labels, same `Carve`, same fit code: give the router the data E38's
router had and it reproduces E38's number to **4.7e-07**. That closes §5 as a diagnosis and makes
B's table the one measured on the known-good instrument.

### 10.2 The six verdict cells at the reference budget

| partition | addendum B | run 1 | Δ from 4× calibration | % of window lost | vs chance |
|---|---|---|---|---|---|
| **`COACT`** | **3.583800** | 3.887003 | **−0.303203** | **85.28%** | −0.4860 |
| `D0C` | 3.597108 | 3.804346 | −0.207239 | 85.69% | −0.4727 |
| `PERM` | 3.597415 | 3.804346 | −0.206932 | 85.69% | −0.4724 |
| `CONC` | 3.726419 | 3.815911 | −0.089492 | 89.60% | −0.3434 |
| `STRIPE` | 4.359874 | 4.415238 | −0.055364 | 108.78% | +0.2901 |
| `RAND` | 4.535312 | 4.530484 | **+0.004828** | 114.10% | +0.4655 |

**Ordering run 1:** `D0C < PERM < CONC < COACT < STRIPE < RAND`.
**Ordering addendum B:** `COACT < D0C < PERM < CONC < STRIPE < RAND` — **CHANGED.**
Best is **`COACT` 3.583800**, crossing the `3.597108` boundary by **−0.013308**, band
`PARTITION-HELPS-A-LITTLE`.

**The registered rule applies and the cell is UNRESOLVABLE** (§0.0). Both tables stand; neither
promotes the other.

### 10.3 What the Δ column says, and it is the most informative thing in the probe

**How much a partition gains from 4× more calibration is not noise and it is not uniform.**
`RAND` gains **nothing and in fact loses 0.0048** — a random grouping has no structure for a
linear router to learn, so more data buys nothing, which is exactly the behaviour a null arm
should show. `STRIPE`, built to equalise mass and therefore to destroy structure, gains **0.055**.
`CONC` gains **0.089**. The two D0c-derived arms gain **0.207** each. **`COACT` gains 0.303, the
most of the six** — and enough to move from third to first.

That ordering of *gains* is the predictability axis §3 identified, read directly: the partitions
a router can learn are the ones that benefit when the router gets more to learn from. **My own
registered prediction 4 said `COACT` would gain most for this reason. It did. I then registered
that it would not overtake `D0C`, and it did.**

### 10.4 The only dispersion estimate available, and why it is not enough

At the 8-sequence budget `PERM fitted` reproduced `D0C fitted` to all sixteen digits. **At 32 it
does not**: `PERM − D0C = +3.070e-04`. The ridge refit is not exactly permutation-equivariant
once the normal equations are better conditioned, and that number is a genuine, unregistered
**noise floor on the fitted path**: `COACT`'s winning margin of `0.013308` is **43×** it.

**That is not sufficient and the cell is not rescued by it.** A relabelling control bounds
relabelling noise. The axis that actually reversed the ordering is the **choice of calibration
slice**, and on that axis this probe has two points, one seed each, and no replication at all.
**A margin of 0.0133 measured on an axis whose dispersion is unmeasured is not a result** — it is
the reason the cell is declared unresolvable rather than simply overturned.

### 10.5 Addendum B's predictions — **1 HIT, 2 split, 2 MISS**

| # | registered | outcome |
|---|---|---|
| 1 | `G-E41D` fires, `D0C` = 3.597108 ± 0.001 | **HIT**, and by `4.7e-07` |
| 2 | ordering unchanged, `D0C < CONC < COACT < STRIPE < RAND` | **MISS.** `COACT < D0C < PERM < CONC < STRIPE < RAND` |
| 3 | nothing crosses 3.597; closest is `CONC` at `+0.012 ± 0.030` | **MISS twice.** `COACT` crosses at `−0.013308`; `CONC` lands at `+0.129311`, four times outside the registered band |
| 4 | `COACT` gains most, closes to within `+0.04`, **does not overtake** | **SPLIT.** Gains most ✓ (`−0.303203`, largest of six), closes ✓ (past `+0.04`), **overtakes ✗** |
| 5 | `RAND` stays last ✓; `STRIPE` crosses back below chance | **SPLIT.** `RAND` last at 4.535312 ✓; `STRIPE` 4.359874, still `+0.290` above chance ✗ — and I registered this as the prediction most likely to be wrong |

**Across the whole probe: 1 HIT, 4 split, 5 MISS on ten registered predictions.**

### 10.6 What E41 is, now that it is over

**Not a verdict.** Two instruments, one of them known-good, disagree on the ordering of the six
partitions at the cell that was supposed to decide, and the axis they disagree along was never
replicated.

**What survives both instruments**, because it does not depend on which partition wins:

- At 6.25% activation — what E40 measured 50 tok/s buying at ten billion — **the best of six
  groupings loses 85.3% (B) to 92.0% (run 1) of the dense→chance window**, against a usability
  bar it misses by **2.58 BPB**.
- At 1.17%, **every cell of twelve is above the chance line**, oracle included.
- **The oracle and the attainable router rank the six partitions differently** (§3), and
  addendum B adds that **the attainable ranking is itself unstable in the router's calibration
  budget** — so *both* cheap ways of pricing a grouping are misleading.
- **Predictability, not mass, is the axis that pays** (§10.3), and the only criterion that even
  gestures at it — a 1-D co-activation proxy — is the one that wins on the known-good instrument.

**What E41 does NOT get to do is count as evidence for the T4 ask.** That ask now rests on E37,
E38 and E40 exactly as it did before this probe was run, and `COMMUNICATION.md` has been
corrected to say so.
