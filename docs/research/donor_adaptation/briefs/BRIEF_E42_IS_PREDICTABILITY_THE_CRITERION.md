# BRIEF E42 — is PREDICTABILITY the criterion, and does it have headroom?

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

**QUALITY ONLY, fp32, on a really trained donor. No speed, no synthetic weights, no GPU.**

---

## 0. Why this, and why it is not a re-run of E41

E41's verdict cell came out **unresolvable**: two calibration budgets, one seed each, and the
ordering of the six partitions reversed between them. It left two things owed, and they are the
same probe:

1. **The dispersion on the axis that reversed the ordering was never measured.** Two points, no
   replication. `COACT` beats `D0C` by `0.013308` on the known-good instrument and I refused to
   promote that, because a margin measured on an axis whose noise is unknown is not a result.
2. **The mechanism E41 did identify is untested as a design criterion.** The gain from 4×
   calibration was not uniform and not noise: `RAND` **lost** 0.0048 (nothing to learn),
   `STRIPE` 0.055, `CONC` 0.089, the D0c-derived arms 0.207, **`COACT` 0.303, the most of six**.
   What pays is how **predictable** a grouping is from `x` — and the only arm that even gestures
   at predictability is a crude 1-D co-activation proxy built in an afternoon.

**The goal-relevant question is not whether `COACT` beats `D0C` by a hundredth.** At 6.25% the
best of six reads 3.5838 against a usability bar of 1.005039 — the gap is **2.58 BPB**, and a
hundredth does not touch it. The question that matters is:

> **Does building the partition FOR predictability have HEADROOM — does it move the number by
> units, or does it saturate in hundredths like everything else on this axis?**

E42 asks both at once, with the dispersion measured so the answer is not unresolvable again.

## 1. The question

> **Hold the donor, the rate, the group count and the router recipe fixed. Add a partition built
> explicitly so that each group's mass is LINEARLY PREDICTABLE from the layer input, measure it
> against `D0C` and `COACT` over several calibration seeds, and ask whether predictability is a
> design criterion with room in it or another hundredth.**

## 2. Setup, inherited unchanged so the numbers compose with E38 and E41

Qwen2.5-1.5B (`S15`), **fp32**, `D=1536, F=8960, L=28`, `E=256` groups of **35**. Frozen heldout
slice `("heldout", 24, 512, 1234)`, `ids_sha a1a48dc9…`, 51,870 scored bytes. Dense **0.767595**,
chance **4.069819**, window **3.302224**. Masking hook is E38's `Carve`; the ridge router is
`e23_router.fit_routers`, refit per partition per seed.

**Router budget is fixed at the reference: 32 sequences** — the budget `G-E41D` proved reproduces
E38's published `3.597108` to `4.7e-07`. **The calibration SEED is the replicated axis:
`42424` (E23's own, and E38's), `42425`, `42426`.** Three seeds, not two budgets.

`k = 16` only — 6.25%, what E40 measured 50 tok/s buying at ten billion. **No `k` sweep**: E41
already measured that 1.17% is above chance on every partition and the sweep is not the question.

## 3. The arms

| arm | how the 8,960 neurons are grouped | why it is here |
|---|---|---|
| **`D0C`** | E37's label family | **the control**, and the known positive at seed 42424 |
| **`COACT`** | E41's 1-D co-activation proxy, rebuilt identically | **the incumbent** — what beat the control by 0.013308 once |
| **`PRED`** | **the new arm** — see §4 | the criterion, built properly instead of gestured at |
| `RAND` | random equal partition, seed 38038 (E38's grouping B) | **the null** — it has no structure to learn and must not improve with anything |

## 4. How `PRED` is built, fixed here before it exists

For each layer, with the calibration activations `x` (the same `s→xb` the router reads) and the
FFN intermediate `h = gate*up`:

1. Solve one ridge system per layer for the **per-neuron** predictor of activation magnitude:
   `W ≈ argmin ‖X W − |H|‖² + λ‖W‖²`, `W` of shape `D × F`. This is the *same* normal equations
   `fit_routers` already forms (`XᵀX` is `D × D`, solved once; `XᵀY` is `D × F`), so it costs one
   extra right-hand side and no new algorithm.
2. Normalise each neuron's predictor `w_j` to unit length. **Two neurons whose `w_j` point the
   same way are predicted by the same direction of `x`.**
3. Spherical k-means on `{ŵ_j}` into 256 clusters, seed **42042**, 25 iterations, then a
   **balanced** assignment to exactly 35 neurons per group (each neuron to its nearest cluster
   with remaining capacity, processed in order of assignment confidence).

**The claim being tested is mechanical**: the router is a linear map from `x` to 256 group
scores, so a group whose members share a predictor direction is a group the router can score.
`COACT` groups by *when* neurons fire together; `PRED` groups by *what direction predicts them*.

## 5. The verdict cell and the bands, named before the run

**The cell is `PRED`'s mean `fitted` BPB at `k = 16` across the three seeds.**

| band | name | what it would mean |
|---|---|---|
| **≤ 1.005** | `PREDICTABILITY-IS-THE-LEVER` | usable at the rate the box affords; the T4 ask is withdrawn |
| 1.005 – 2.50 | `PREDICTABILITY-HAS-HEADROOM` | units, not hundredths — a second CPU probe before any GPU |
| 2.50 – 3.449 | `PREDICTABILITY-HELPS` | beats `D0C`'s own **oracle** (3.449466) with an attainable router |
| 3.449 – 3.5838 | `PREDICTABILITY-HELPS-A-LITTLE` | beats `COACT`, the incumbent, but stays in hundredths |
| **> 3.5838** | **`PREDICTABILITY-IS-NOT-THE-CRITERION`** | the criterion was named and built and it buys nothing |

**Reported alongside and worth more than the cell: the across-seed standard deviation of every
arm**, which is the number E41 did not have and the reason its cell was unresolvable.

**Secondary registered question, scored separately and not part of the verdict:** is
`COACT − D0C` stable across seeds? It is resolved **only** if the three-seed mean differs from
zero by more than **2×** the across-seed standard deviation of that difference. Otherwise E41's
ordering is declared noise and stays unresolved — **a second unresolvable is a legitimate
outcome and will be reported as one, not massaged.**

## 6. The gates

**`G-E42A` — the known positive.** `D0C` at seed 42424 must reproduce E38's **3.597108 ± 0.001**.
Same demand `G-E41D` met at `4.7e-07`; if it fails, this session is not the instrument E41's
addendum B was and nothing composes.

**`G-E42B` — inertness.** At `k = E` every partition, `PRED` included, must be bit-inert against
dense within `1e-6`. A new partition construction is new code and gets the same test E38 and E41
put theirs through.

**`G-E42C` — the null must behave like a null.** `RAND`'s across-seed spread must be the
**smallest improvement** of the four arms — concretely, `RAND`'s three-seed mean must not beat
its own E41 32-seq reading (4.535312) by more than `0.01`. **A random partition has nothing to
learn; if it improves with a change of calibration seed, the harness is fitting the eval and no
arm below counts.**

**`G-E42D` — `PRED` must be a real partition.** Exactly 256 groups of exactly 35 neurons in every
one of the 28 layers, and its label vector must differ from `COACT`'s in at least 50% of neurons.
**If the two criteria produce nearly the same partition, `PRED` is not a new arm and the probe
says so instead of reporting a difference.**

## 7. Predictions — fixed here, before the runner exists

1. **All four gates fire**, and `G-E42A` within `1e-5` rather than merely `1e-3`.
2. **`PREDICTABILITY-HELPS-A-LITTLE`, and specifically `3.52` BPB** for `PRED`. I am predicting
   the criterion is real and its headroom is not: better than `COACT`'s 3.5838, nowhere near the
   3.449 oracle, and 2.5 BPB from usable. **If E41 taught me anything it is that I price this
   axis in units and it pays in hundredths.**
3. **The across-seed standard deviation is between `0.005` and `0.030` for every arm**, which
   would make `COACT − D0C = −0.0133` **not** resolvable at 2σ — i.e. **I predict my secondary
   question comes back unresolved a second time**, and I am registering that rather than hoping.
4. **`RAND` has the smallest across-seed spread of the four** (nothing to learn means nothing to
   vary), and `PRED` the largest.
5. **`PRED` beats `COACT` at every one of the three seeds**, not just on the mean. If the
   criterion is real it should not need averaging to show.

## 8. What E42 will NOT be able to claim

- **Nothing about ternary, training, or 10 B.** fp32, no gradient, one donor at 1.5 B — E38's,
  E40's and E41's limits, unchanged.
- **Nothing about `E`.** 256 groups throughout; group count is still the untouched third axis.
- **Nothing about non-linear routers.** `PRED` is built for the *linear* router this engine ships.
  A partition optimal for a linear selector need not be optimal for any other.
- **Nothing about the selector's criterion.** E38 §7 item 1 stays owed; E41 moved the partition
  and E42 moves how the partition is *built*, which is still not the selector.
- **Nothing that resolves E41's cell by fiat.** If §5's secondary question comes back inside 2σ,
  E41 stays unresolvable and this brief says so in advance.

---

# ADDENDUM A — `G-E42C` went VOID, and the gate is mis-specified. It is NOT re-run to a pass.

**Written after the run. §0–§8 above are unedited.** Precedent: E40 addendum A, E41 addendum A.

## What fired

```
G-E42C  RAND mean 4.456739  vs its E41 32-seq reading 4.535312  -> gain +0.078573 (max 0.010)  -> VOID
```

Both clauses of §6's `G-E42C` fail, not one. The operational clause fails by `7.9×` the
tolerance. The prose clause — *"`RAND`'s across-seed spread must be the smallest improvement of
the four arms"* — also fails: `RAND`'s across-seed standard deviation is **0.099072**, the
**second largest** of the four, against `D0C`'s 0.026676.

**Prediction 1 (all four gates fire) is a MISS and is scored as one.**

## Why the gate is mis-specified, and it is my error

The gate compares a **three-seed mean** against a **one-seed value** with a tolerance of
**0.010**, while the arm's own across-seed standard deviation turns out to be **0.099** — ten
times the tolerance. **A control that a null arm's ordinary dispersion is guaranteed to trip is
not a control.** I set the tolerance before any dispersion on this axis had ever been measured,
which is precisely the number E42 was built to measure; I should have specified the gate in units
of the arm's own measured spread, or compared seed-for-seed.

## The inference the gate names is separately refuted

`G-E42C`'s registered consequence reads: *"if it improves with a change of calibration seed, the
harness is fitting the eval and no arm below counts."* The eval-fitting hypothesis is testable
directly and it fails:

| cell | published elsewhere | E42, seed 42424 | difference |
|---|---|---|---|
| `D0C` fitted `k=16` | E38 **3.597108** | **3.597107527** | **−4.73e−07** |
| `RAND` fitted `k=16` | E41 addendum B **4.535311855** | **4.535311605** | **−2.50e−07** |

**Two independently published cells reproduce to better than `5e-07` on this harness at the
shared seed.** A harness fitting the eval does not reproduce other sessions' numbers to seven
digits. What `RAND` actually does across seeds is **vary** (4.345445 / 4.489461 / 4.535312) —
dispersion, not monotone improvement.

## What happens to E42's verdict, and I am applying the harsh reading

**The registered consequence stands: no arm below the gate counts, so E42 yields NO REGISTERED
VERDICT.** `PREDICTABILITY-IS-NOT-THE-CRITERION` is what the runner printed and it is reported as
a **measurement, not a verdict.** The gate is not re-run, not re-scoped and not re-toleranced
after seeing the numbers.

Two things are said alongside, neither of which rescues it:

- **The failure mode the gate guards against cannot manufacture E42's result.** A harness fitting
  the eval makes arms look *better*; E42's headline is that the new arm is *worse* than both
  incumbents at every seed. The direction is wrong for the artifact.
- **The dispersion figures are the point of the probe and are reported as measured-not-
  registered.** They are what a repaired successor must re-establish, and they are what finally
  explains E41.

## The repair, specified here for the successor

A null-arm control must be expressed in units the probe itself measures:

> `RAND` must not beat its own seed-matched reading at any shared seed by more than **1σ of its
> own across-seed spread**, with σ estimated from the same run — and the comparison must be
> **seed-for-seed**, never a mean against a single point.

On E42's own numbers that repaired gate **would have fired** (`RAND` at seed 42424 reproduces
E41's reading to `2.5e-07`, and there are no other shared seeds). **That is stated to show the
repair is not a loophole — it is not applied, and E42's gate stays VOID.**
