# E37 — what E36's speed costs a model that was actually trained

**Read this first, because it is the registered limit (brief §6 item 6).** This settles the
**quality** half of E36's activation rate on **one donor, at 1.5 B, with one label set and one
router family**. It does **not** prove that no sparse 10 B exists. It measures that *this
conversion* does not produce one — a statement about **adaptation**, not about the architecture.
The lever it leaves standing is **training inside the format**, not further conversion.

---

**Verdict: `SPARSITY-DEGRADES`.** The verdict cell — a really-trained donor forced to **1.17%
FFN activation**, the exact rate E36's 10 B artifact needs — reads **`S15-K3 = 4.029398` BPB**.
That is **+0.5537 above dense** (3.475706) and **0.0404 below the 4.069819 chance line**. It
beats coin-flipping by four hundredths of a bit and it is not a model.

**The structural gap is closed.** For the first time in this branch **one artifact carries both
a stopwatch and a BPB**: `k=3` runs at **107.84 tok/s** against the dense **30.52** — **3.53×** —
and the exchange rate is **0.00716 BPB per tok/s gained**.

**And the result I did not expect, which is the one worth keeping**: a router that captures
**71.3%** of the oracle's group mass and a **random** one that captures **6.1%** produce the
**same model** (−0.0088 BPB apart at the verdict cell — the random one nominally *better*). The
damage is about **how few** neurons survive, not **which**.

**Brief**: `briefs/BRIEF_E37_WHAT_THE_SPEED_COSTS_A_TRAINED_MODEL.md`, pushed at `3072941`
before any runner existed; **addendum A** (`8cc405a`, two of my own gates were wrong) and
**addendum B** (`341b505`, the router control, pushed before it ran).
**Runners**: `engine/e37_fit_routers.py`, `engine/e37_sparsity_cost.py`, `engine/e37_router_control.py`.
**Results**: `engine/results/e37_routers.json`, `engine/results/e37_sparsity_cost.json`,
`engine/results/e37_router_control.json`.

---

## 1. The gates — all four fire, two of them only after being rewritten

| gate | as registered | as run | measured | fires |
|---|---|---|---|---|
| `G-E37A′` | BPB parity at `k=E` < 1e-6 | **< 1e-4**, E26's own `TOL_P` | BPB diff **2.80e-07**, worst rel L2 **4.99e-05**, top-1 **1.0000** | **yes** |
| `G-E37B1` | anchor 3.475706372 ± 0.001 | unchanged | **3.47570637184527**, diff **−1.5e-10** | **yes** |
| `G-E37B2` | (new) fold-matched dense reference | `--fold none` build | see §2 | — |
| `G-E37C` | exporter `active_weights_per_token` == runner | exporter side recomputed from the artifact's **own header** | agrees at **all nine `k`**, zero tolerance | **yes** |
| `G-E37D` | fitted router beats seed-matched random | unchanged | recall@3 **0.5075** vs **0.0102**; mass **0.7132** vs **0.0608**; **28/28** layers | **yes** |

**Two gates as I first wrote them could not have been satisfied**, and that was found by reading
the code, not by seeing a result (addendum A, pushed before any artifact existed):

* **`G-E37A` demanded 1e-6 BPB parity.** `ffn_carved` accumulates `down` in *router-selection*
  order over `F = 8960`; the dense path walks index order. Same trits, different fp32 summation
  order. A 1e-6 bar tests floating-point associativity, not my plumbing. `G-E37A′` adopts
  `e26_parity_carve.py:49`'s existing `TOL_P = 1e-4` rather than inventing a number.
* **`G-E37B` compared against a reference the carve is forbidden to match.**
  `qwen_export.py:390` refuses `--quant carved` unless `--fold none`, while the published anchor
  used the default `--fold layers`. Split into B1 (reproduce the record) and B2 (the carve's
  actual reference).

**`G-E37C` named a field nobody writes.** `active_weights_per_token` had never been emitted by
`qwen_export.py`. I did not widen the gate to fit: the exporter side is now computed by applying
`synth_export.active_weights()` to the **exported file's own header**, and `qwen_export.py`
records the count going forward.

## 2. The fold cost is exactly zero, for a reason worth writing down

Addendum A registered the fold difference as a **measured cost with no predicted sign**. It is
**0.000000**: `e37_dense_nf.bin` is **byte-identical** to the published anchor
(sha256 `5d50e3778917a46f…`, 1,709,047,348 bytes). The anchor's JSON has **no `fold` key** — it
predates the flag, so "the default `--fold layers`" was never actually applied to it. The gate
split was still right to make; it just found that the two objects were already one.

## 3. The quality sweep

E1's protocol, unchanged: `--seqlen 512`, 51,870 scored bytes, 12,264 predictions,
`BYTES_PER_TOK = 51870/12264`, **chance line 4.069819**. Deterministic, so measured under load.

| arm | active | BPB | vs dense | vs chance |
|---|---|---|---|---|
| `S15-DENSE-NF` | 100% | **3.475706** | — | −0.594 |
| `S15-K256` | 100% | 3.475707 | +0.000000 | −0.594 |
| `S15-K64` | 25.00% | 3.927384 | +0.4517 | −0.142 |
| `S15-K32` | 12.50% | **4.074431** | +0.5987 | **+0.005** |
| `S15-K16` | 6.25% | 3.986801 | +0.5111 | −0.083 |
| `S15-K8` | 3.12% | 3.996693 | +0.5210 | −0.073 |
| `S15-K4` | 1.56% | 4.023439 | +0.5477 | −0.046 |
| **`S15-K3`** | **1.17%** | **4.029398** | **+0.5537** | **−0.040** |
| `S15-K2` | 0.78% | 4.014866 | +0.5392 | −0.055 |
| `S15-K1` | 0.39% | 3.989839 | +0.5141 | −0.080 |

**The curve is a cliff and then a plateau.** Everything from `k=16` to `k=1` sits in
**3.99–4.03** — a **40× change in activation that moves quality by less than the spread between
neighbours.** The single point *over* the chance line is `k=32`, in the middle of the range.

## 4. The exchange rate — both axes on one artifact, the first time in this branch

Idle box, five reps, reps outermost, one warm token before `--bench`. CPU busy during the timing
phase: **13.0% mean / 23% peak** on the verdict rep.

| arm | tok/s | spread | charged/token | numerator | BPB | ΔBPB per Δtok/s |
|---|---|---|---|---|---|---|
| `DENSE-NF` | **30.52** | 10.1% | 1.5436 G | 47.116 G-w/s | 3.475706 | — |
| `K256` | 28.34 | 13.1% | 1.5546 G | 44.060 | 3.475707 | −0.000000 |
| `K64` | 55.05 | 13.2% | 0.6875 G | 37.852 | 3.927384 | 0.018413 |
| `K32` | 73.24 | 17.5% | 0.5430 G | 39.771 | 4.074431 | 0.014017 |
| `K16` | 88.71 | 13.0% | 0.4708 G | 41.762 | 3.986801 | 0.008784 |
| `K8` | 95.19 | 18.8% | 0.4347 G | 41.376 | 3.996693 | 0.008056 |
| `K4` | 104.79 | 10.0% | 0.4166 G | 43.653 | 4.023439 | 0.007375 |
| **`K3`** | **107.84** | **4.8%** | 0.4121 G | 44.436 | 4.029398 | **0.007162** |
| `K2` | 107.71 | 11.4% | 0.4076 G | 43.899 | 4.014866 | 0.006985 |
| `K1` | 101.59 | 26.1% | 0.4030 G | 40.944 | 3.989839 | 0.007235 |

**The dispersions are large** (up to 26.1% on `K1`, whose arm is the shortest) and every absolute
tok/s here carries that on top of the standing ±5%. **What survives is the ratio column**, and
three of these numbers are cross-checks rather than new claims:

* **`DENSE-NF` reads 47.116 G-w/s** — inside E36's base envelope of 47.2–48.5 G-w/s, measured on
  a different shape at a different scale with real weights instead of noise.
* **`K256` is 7.15% slower than `DENSE-NF` at byte-neutral cost.** E26's planted control read
  **−6.80%** for the same comparison. **The carve machine costs ~7% before a single group is
  dropped**, confirmed independently.
* Fitting `time = c₀ + c₁·k` over the eight arms below `K256` (where `K256`'s leverage does not
  dominate) gives **9.16 ms + 0.140 ms per group**, i.e. a base of **43.5 G-w/s** and a marginal
  group of **32.2 G-w/s** → **gather penalty 0.74**; restricting to `k ≤ 16` gives **0.81**.
  **E36 measured 0.80 at A10B.** Two shapes, two scales, noise weights and real weights, and the
  marginal FFN weight costs about **1.25×** an attention weight in both.

## 5. Addendum B — the control that landed on my own gate

`G-E37D` fired convincingly. Addendum B was pushed *before the control ran* to ask whether
firing meant anything, because E14 §3's failure mode has a mirror: **a RANK result standing in
for a SCORE result it does not imply.**

Same donor, same revision, same `--rule R3`, same `--fold none`, same labels, same
`--carve-k 256` in the file — **only the router differs** (`--carve-seed 26`, the synthetic
matrix every carved artifact before E37 used).

| k | fitted | synthetic | synthetic − fitted |
|---|---|---|---|
| 64 | 3.927384 | 4.059717 | **+0.1323** |
| 32 | 4.074431 | 3.998745 | −0.0757 |
| 16 | 3.986801 | 4.074449 | +0.0876 |
| **3** | **4.029398** | **4.020624** | **−0.0088** ← decides |
| 1 | 3.989839 | 3.974800 | −0.0150 |

**`ROUTER-IS-NOT-THE-CONSTRAINT`** (bar 0.05). At the deciding `k` the **synthetic** router is
nominally *better*. `G-E37D`'s win is **real and inert**: the fitted router predicts group mass
far better than random and the model does not care. **The damage is about how few neurons
survive, not which ones.**

**Addendum B's rule held: the verdict cell did not move.** `S15-K3 = 4.029398` and its band were
measured and committed at `341b505` before this control existed, and this control could not
re-open them. All three of its predictions HIT.

## 6. Predictions, scored as registered

**Brief §6:**

| # | registered | outcome |
|---|---|---|
| 1 | all four gates fire | **HIT** (after addendum A rewrote two of them, which is recorded, not hidden) |
| 2 | **`SPARSITY-DESTROYS`** — verdict worse than chance | **MISS**. `SPARSITY-DEGRADES`: 4.029398, **0.0404 under** the line. I over-predicted the collapse. |
| 3 | knee: within +0.10 of dense down to `k ≥ 64`; crosses chance in `k ∈ [4,16]` | **MISS on both halves**, and badly. `k=64` already costs **+0.4517**, not +0.10 — the cliff is between 100% and 25% activation, above everything I sampled. And the curve **never settles above chance**: the only point over the line is `k=32`. |
| 4 | speed not binding at S15 — "the dense donor already clears 50 tok/s" | **MISS**. `DENSE-NF` reads **30.52 tok/s**. The premise was wrong; the conclusion (E37 is not a speed probe) stands anyway. |
| 5 | exchange rate **worse** than the charged-weight model predicts | **HIT**. Charged weights fall **3.746×** from dense to `k=3`; measured speedup **3.533×** — **0.943** of what a weight count promises, and the slope fit attributes it to the same gather penalty E36 measured. |
| 6 | registered limitation (one donor, one scale, one label set, one router family) | **held** — restated at the top of this document. |

**Addendum B §4:** 1 **HIT** (`ROUTER-IS-NOT-THE-CONSTRAINT`), 2 **HIT** (fitted does win at
`k=64`, +0.1323 — the only arm where enough FFN survives for the choice to matter), 3 **HIT**
(the synthetic curve is non-monotone too, worst point mid-range at `k=16`).

**Three MISSes, and they point the same way.** I predicted a collapse *below* chance and got a
plateau *just above* it; I predicted a gentle knee at 25% activation and the cliff is already
past by then. Both errors are the same error: **I was modelling the damage as a function of how
much FFN mass is missing, and it is not.** §7's registered hypothesis is the reading that
survives.

## 7. The non-monotonicity, and the hypothesis registered before it could be fitted

Addendum B §5 registered this **before the control ran**: the measured curve is worst at `k=32`
and *improves* as activation falls further, reaching 3.989839 at `k=1` (0.39%).

> **A partial FFN is worse than almost no FFN.** `down_proj` sums contributions from the neurons
> that survive; dropping most of them does not shrink the update toward zero, it produces a
> **systematically wrong** update of roughly the original magnitude. As `k → 0` the FFN
> contribution vanishes and the model falls back on attention plus the residual stream; in the
> middle it is loudly wrong.

The control **strengthens** this: the synthetic curve is non-monotone too, so the anomaly is not
a property of the fitted router. **It is not tested here**, and it cannot be with this engine:
`ffn_carved` clamps `k ≥ 1`, so the zero-FFN arm does not exist as an arm.

**A plumbing defect was the first suspect and has been ruled out.** `--carve-dump` at `k=32` and
`k=3` shows exactly `k` distinct in-range groups per layer per token, **0 malformed**, and
token-to-token selection overlap 0.731 / 0.798.

## 8. What this does not say

* **Nothing about a trained-sparse model.** This is what happens when a **dense-trained** donor
  is *forced* sparse. Whether training inside the format works is H0's and rung-1's axis.
* **Nothing about 10 B.** S15 is 1.5 B. The activation *rate* matches A10B's; the shape does not.
* **Nothing about healing.** No fine-tuning, no distillation, no recovery step of any kind.
* **Nothing about other groupings.** One label set (a single D0c partition), one ridge router
  family, one target. A different grouping is untested.
* **The absolute tok/s are soft.** Up to 26.1% spread on the weakest arm. The ratios and the
  slope are the load-bearing parts.

## 9. Owed

1. **A zero-FFN arm**, to test §7. The engine cannot express it today (`ffn_carved` clamps
   `k ≥ 1`); it needs either an engine flag or an export with `F` collapsed.
2. **A second grouping.** If the damage is "how few, not which", that should be invariant to the
   partition — and if it is not, §7's reading is wrong.
3. **Healing after the carve.** The question E37 was designed to make worth asking, and it is
   now worth asking: +0.55 BPB is the gap a recovery step would have to close.
4. **A `GSZ` sweep on `A10B`** (E36 §9 item 1) — the marginal group is the dominant lever and
   both probes now agree on its price.
5. **E34 §7's attention-shape sweep** — E36 put 17.2 ms of a 20 ms budget in attention+head.
6. Still open from before: E33 §7 (`PT_BLK`), E32 §8 (`--lut-group 32` has never been timed),
   E29 §8, E26 §9 items 1–3, E22 §8, E20 §8.
