# E42 — is PREDICTABILITY the criterion? **NO REGISTERED VERDICT — the planted control went VOID**

**Brief:** `briefs/BRIEF_E42_IS_PREDICTABILITY_THE_CRITERION.md`, pushed at `22653fd` before the
runner existed; **addendum A** (the `G-E42C` diagnosis) written after the run.
**Runner:** `benchmarks/donor_adaptation/engine/e42_predictability.py` (`defec04`).
**Results:** `engine/results/e42_predictability.json`. **Log:** `e42_run.log`. 5,990 s.

**QUALITY ONLY, fp32, on a really trained donor. No speed, no synthetic weights, no GPU.**

---

## 0. What E42 is, in three sentences

1. **It has no registered verdict.** `G-E42C`, the null-arm control, went VOID. The registered
   consequence — *"no arm below counts"* — is applied in full: everything in §3 is a
   **measurement, not a verdict**, and the gate is not re-run, re-scoped or re-toleranced.
2. **What the measurement says**, with that label attached: the criterion E41 pointed at,
   built properly, is **worse than both incumbents at every seed** — `PRED` **3.812991** against
   `D0C` 3.614540 and `COACT` 3.509543.
3. **What E42 does establish cleanly, because it does not depend on which arm wins:** the
   across-seed dispersion of the fitted column is **σ = 0.027 – 0.143 BPB**, and *that* number
   retroactively explains E41. The margin E41's addendum B "won" by — 0.013308 — is **0.09σ** of
   `COACT`'s own spread.

---

## 1. The gates

| gate | what it demanded | result |
|---|---|---|
| **`G-E42A`** | `D0C` at seed 42424 reproduces E38's `3.597108 ± 0.001` | **FIRES** at **−4.73e−07** |
| **`G-E42B`** | every partition bit-inert at `k = E`, `< 1e-6` | **FIRES**, worst **0.00e+00** on all four |
| **`G-E42C`** | the null must behave like a null | **VOID** — see §2 |
| **`G-E42D`** | `PRED` is 256×35 in every layer and is not `COACT` renamed | **FIRES** — worst-layer disagreement **93.7%** |

### 1.1 Two published cells reproduce to better than 5e-07

| cell | published | E42, seed 42424 | difference |
|---|---|---|---|
| `D0C` fitted `k=16` | E38 **3.597108** | 3.597107527 | **−4.73e−07** |
| `RAND` fitted `k=16` | E41 addendum B **4.535311855** | 4.535311605 | **−2.50e−07** |

Two different sessions, two different partitions, seven digits. **This is the direct test of the
failure mode `G-E42C` names, and it passes** — which does not make the gate fire (§2).

---

## 2. `G-E42C` went VOID, the gate is mis-specified, and it is not re-run

```
G-E42C  RAND mean 4.456739  vs its E41 32-seq reading 4.535312  ->  gain +0.078573 (max 0.010)  -> VOID
```

**Both clauses fail.** The operational clause by `7.9×` the tolerance; and the prose clause —
*"`RAND`'s across-seed spread must be the smallest of the four"* — fails too: `RAND`'s σ is
**0.099072**, the **second largest**, against `D0C`'s 0.026676.

**The error is mine and it is arithmetic.** The gate compares a **three-seed mean** to a
**one-seed value** with a tolerance of **0.010**, while the arm's own across-seed σ turns out to
be **0.099** — ten times the tolerance. **A control that a null arm's ordinary dispersion is
guaranteed to trip is not a control.** I fixed that tolerance before any dispersion on this axis
had ever been measured — which is the very quantity E42 was built to measure.

**What `RAND` actually does across seeds is vary**: 4.535312 / 4.345445 / 4.489461. That is
dispersion, not monotone improvement, and §1.1 shows the harness reproducing two other sessions'
cells to seven digits. **The eval-fitting hypothesis the gate names is refuted by direct test.**

**None of that rescues the gate.** E40 addendum A's precedent is the standing one: a gate that
fires is doing its job and is not re-run to a pass. So **E42 yields no registered verdict**, and
two things are said alongside without pretending otherwise:

- **The artifact the gate guards against cannot produce E42's result.** Eval-fitting makes arms
  look *better*. E42's headline is that the new arm is *worse* than both incumbents at every
  seed. The direction is wrong for the artifact.
- **The dispersion numbers are reported as measured-not-registered**, because they are the point
  of the probe, they are what explains E41, and suppressing them would be worse than labelling
  them.

The repair is specified in the brief's addendum A: a null-arm control must be expressed
**seed-for-seed** and in units of **the arm's own measured σ**, never as a mean against a single
point.

---

## 3. The measurement (no registered verdict — §0, §2)

`S15` fp32, `E = 256` groups of 35, `k = 16` (6.25%, what E40 measured 50 tok/s buying at ten
billion). Heldout 24×512, 51,870 scored bytes, dense **0.767594964** (`−3.59e−08` from E22's
base), chance **4.069819**, window **3.302224**. Partitions built **once** on the reference
calibration slice (32 seqs, seed 42424) and held fixed; only the **router's** calibration seed
varies.

| arm | seed 42424 | seed 42425 | seed 42426 | **mean** | **σ** | % of window lost |
|---|---|---|---|---|---|---|
| **`COACT`** | 3.565004 | **3.346968** | 3.616657 | **3.509543** | **0.143143** | 83.0% |
| **`D0C`** | 3.597108 | 3.601264 | 3.645249 | **3.614540** | **0.026676** | 86.2% |
| **`PRED`** | 3.837749 | 3.769874 | 3.831349 | **3.812991** | **0.037477** | 92.2% |
| `RAND` | 4.535312 | 4.345445 | 4.489461 | **4.456739** | **0.099072** | 111.7% |

**`PRED` is last but one, behind both incumbents, at every seed.** `PRED − D0C` = +0.240642 /
+0.168609 / +0.186100; `PRED − COACT` = +0.303448 on the means. The runner's band for the cell is
`PREDICTABILITY-IS-NOT-THE-CRITERION`, and it is printed for the record with no registered force.

### 3.1 The criterion did what it was designed to do, mechanically, and lost anyway

`PRED` groups neurons by **shared linear predictor direction**, so a group's mass should be
easier for a linear router to score. The mechanical signature is there: **`PRED`'s across-seed σ
is 0.037, `3.8×` tighter than `COACT`'s 0.143** — the router's job really did become less
dependent on which calibration slice it saw. **The construction succeeded and the model got
worse.**

**Hypothesis, registered here as a hypothesis and not as a claim:** predictability is bought with
**coverage**. Neurons predicted by the same direction of `x` carry similar information, so 16
predictability-coherent groups span fewer distinct directions of the FFN's output than 16 diverse
ones. Testable without a GPU: measure the effective rank of the kept `down_proj` rows under each
partition at `k = 16`. **Nothing in E42 measures this and nothing here asserts it.**

### 3.2 The number that outlives the probe: σ on the fitted column

**σ = 0.027 (`D0C`) to 0.143 (`COACT`), from changing nothing but the router's calibration seed.**

That is the instrument constant this programme has been missing, and it settles E41 from behind:

| comparison | size | in units of `COACT`'s own σ |
|---|---|---|
| E41 addendum B's `COACT − D0C` | **−0.013308** | **0.09σ** |
| E42's `COACT − D0C`, three-seed mean | **−0.104997 ± 0.129309** | 0.73σ — **unresolved at 2σ** |
| `PRED − D0C`, three-seed mean | **+0.198450** | 1.4σ of `COACT`, **5.3σ of `PRED`'s own** |
| `RAND − D0C`, three-seed mean | **+0.842199** | **5.9σ** |

**Only the gross distinction survives.** A structured partition beats a random one by 0.84 BPB
and that is real at any reading. The ordering *among* the structured partitions — the thing E41
spent a day on and the thing addendum B overturned — lives **inside the noise**.

### 3.3 And E41's `COACT` was not even the same object

E42 rebuilds `COACT` from 32 calibration sequences where E41 built it from 8. At the shared seed
it reads **3.565004** against E41 addendum B's **3.583800** — the *partition construction* budget
alone moves it **−0.018796**, which is **larger than the 0.013308 margin E41's addendum B won
by.** `D0C`, which depends on no statistics, reproduces to `4.7e-07`. **Two independent reasons
that margin was never a result, and both of them are measured.**

---

## 4. Predictions — **0 HIT, 1 split, 4 MISS**

| # | registered | outcome |
|---|---|---|
| 1 | all four gates fire, `G-E42A` within `1e-5` | **MISS.** `G-E42C` VOID. (`G-E42A` did land at `4.7e-07`, inside the tighter figure I named) |
| 2 | `PREDICTABILITY-HELPS-A-LITTLE`, **3.52 BPB** | **MISS.** 3.812991, the band above; the registered number is off by **+0.293** and in the wrong direction — I predicted `PRED` would beat `COACT` and it lost to `RAND`'s neighbours |
| 3 | σ in `[0.005, 0.030]` for every arm; `COACT − D0C` **unresolved** at 2σ | **SPLIT.** The consequence is a **HIT** — `−0.104997 ± 0.129309`, unresolved, exactly as registered. The σ range is a **MISS**: only `D0C` (0.0267) is inside; `PRED` 0.0375, `RAND` 0.0991, `COACT` 0.1431 |
| 4 | `RAND` smallest across-seed spread, `PRED` largest | **MISS both ways.** Smallest is `D0C` 0.0267, largest is `COACT` 0.1431; `RAND` is second largest and `PRED` second smallest |
| 5 | `PRED` beats `COACT` at every seed | **MISS at all three** (+0.273 / +0.423 / +0.215) |

**The one thing I got right is the one I registered against myself**: that the secondary question
would come back unresolved a second time. **Everything I predicted about magnitudes was wrong
again, in the same direction as E41 — I price this axis in units and it pays, or costs, in
hundredths and tenths that I cannot call in advance.**

---

## 5. What E42 cannot claim

- **No verdict at all.** §2. The band the runner printed has no registered force.
- **Nothing about ternary, training, or 10 B.** fp32, no gradient, one donor at 1.5 B.
- **Nothing about `E`.** 256 groups throughout; the group count remains the untouched third axis.
- **Nothing about non-linear routers.** `PRED` was built for the linear router this engine ships.
- **Nothing about predictability in general** — only about *this* construction of it: a per-neuron
  ridge predictor of `|h|` from `x`, clustered by direction. A different predictability criterion
  is a different arm.
- **Nothing that re-strengthens the T4 ask.** That ask still rests on E37, E38 and E40, exactly
  as it did before E41 and E42 were run.

## 6. Owed

1. **A repaired null-arm gate**, seed-for-seed and in units of the arm's own measured σ — the
   brief's addendum A specifies it. Any successor on this axis needs it before its arms count.
2. **The coverage hypothesis** (§3.1): effective rank of the kept `down_proj` rows per partition
   at `k = 16`. Cheap, CPU-only, and it would say whether predictability and coverage really
   trade off or whether `PRED` simply has a construction bug.
3. **More seeds.** Three gives σ to ~40% accuracy; the comparisons that matter here are at 0.7σ,
   so five to eight seeds is what a resolvable answer costs.
4. **Still owed from before, untouched by E42:** the selector's criterion sweep (E38 §7 item 1),
   the group count `E`, and the float64 oracle repair (E41 addendum A) for the next probe that
   needs an oracle.
