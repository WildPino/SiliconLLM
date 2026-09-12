# E41 — is the PARTITION the lever? `PARTITION-IS-NOT-THE-LEVER`

**Brief:** `briefs/BRIEF_E41_IS_THE_PARTITION_THE_LEVER.md`, pushed before the runner existed;
**addendum A** (`G-E41C` diagnosed) and **addendum B** (pre-registered re-run) pushed at
`fadeb42`, addendum B before its runner existed.
**Runner:** `benchmarks/donor_adaptation/engine/e41_partition_lever.py`,
`e41_g41c_diag.py`, `e41_addendum_b.py`.
**Results:** `engine/results/e41_partition_lever.json`, `e41_g41c_diag{,_f64}.json`,
`e41_addendum_b.json`. **Logs:** `e41_run.log`, `e41_g41c_diag{,_f64}.log`, `e41_addb.log`.

**QUALITY ONLY, fp32, on a really trained donor. No speed, no synthetic weights.**

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

**The affordable rate and the useful rate still do not overlap, and re-grouping the neurons does
not move either one.** E37 (post-hoc conversion), E38 (any selector), E40 (any attention shape),
E41 (any of six groupings) are four independent measurements of the same closed road.

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
4. **A grouping that is optimised for PREDICTABILITY, not for mass.** §3 says the quantity that
   decides the fitted column is the oracle→fitted gap, and none of the five criteria targeted it.
   This is the only version of "the partition is the lever" that E41 leaves alive.
