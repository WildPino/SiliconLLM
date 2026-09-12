# BRIEF E41 — is the PARTITION the lever? Can a better grouping make ~6% activation usable?

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

**QUALITY ONLY, on a REALLY TRAINED donor, in fp32.** No speed, no synthetic weights. This is the
half of the goal that E37–E40 kept handing to a GPU.

---

## 0. Why this, and why it is worth a CPU afternoon before any GPU hour

E40 closed the shape axis: with every real attention lever pulled, **50 tok/s buys ~6% FFN
activation at ten billion parameters** and 110 tok/s buys 1.17%. E38 measured what a really
trained donor does at those rates — **3.597 BPB at 6.25%**, against a dense **0.767595** and a
chance line of **4.069819** — and concluded no *router* can fix it, because even a per-token
oracle cannot.

**But E38 also found something it could not follow up, and left it owed in its §7 item 3.** The
same static selector, on the same donor at the same `k`, reads **2.2119 on the D0c labels and
4.5833 on a random equal partition** — **a 2.37 BPB swing from the grouping alone, which is four
times the entire window E37's whole sweep lived in.**

> **Every carve in this programme — E19, E26, E36, E37, E38, E39, E40 — has used ONE family of
> labels, and nobody has ever asked whether it is any good.**

E38 proved no router beats the oracle *of a given partition*. **It says nothing about a different
partition, whose oracle is a different function.** If the grouping is the lever, ~6% becomes
usable, the goal is reachable without training, and the T4 ask in `COMMUNICATION.md` should be
withdrawn. If it is not, the ask is justified by one more independent measurement instead of by
argument.

**This costs a CPU afternoon and no GPU.** That is the whole reason it goes first.

## 1. The question

> **Hold the donor, the rate and the selector recipe fixed. Change only HOW NEURONS ARE GROUPED
> into the 256 carve groups, and ask whether any grouping makes the rate the box affords
> produce a model.**

## 2. Setup, inherited unchanged from E38 so the numbers compose

Qwen2.5-1.5B (`S15`), **fp32**, `D=1536, F=8960, L=28`, `E=256` groups of **35** neurons.
Frozen heldout slice `("heldout", 24, 512, 1234)`, `ids_sha`
`a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65`, 51,870 scored bytes.
Dense anchor **0.767595**, chance **4.069819**. Calibration on a **separate** slice
(`"calib"`, seed 424242) — never the eval one.

**Nothing is reimplemented.** The masking hook is E38's `Carve`; the ridge router is
`e23_router.fit_routers`, **refit per partition** with that partition's own label map, which is
the same code E23 and E37 used.

## 3. The partitions

| arm | how the 8,960 neurons are grouped | why it is here |
|---|---|---|
| **`D0C`** | E37's label family | **the control** — what every carve in this programme has used |
| `RAND` | random equal partition, seed 38038 | E38's grouping B, so one cell reproduces |
| **`CONC`** | rank by calibration mass, group 0 = top 35, group 1 = next 35, … | **maximum concentration**: if you may keep only `k` groups, put the heavy neurons together |
| `STRIPE` | rank by calibration mass, deal round-robin | **maximum balance** — every group carries near-identical mass |
| `COACT` | cluster by co-activation correlation on the calib slice | neurons that fire *together* land together, which is the structure a router can predict |
| `PERM` | `D0C` with the 256 group IDs permuted, seed 41041 | **the planted control** — see `G-E41C` |

Selectors: **`oracle`** (per-token, the ceiling of every router on that partition) and
**`fitted`** (the attainable ridge router, refit for each partition).

`k ∈ {16, 3}` — **6.25%, what E40 measured 50 tok/s affording at ten billion**, and 1.17%, what
~110 tok/s affords. Plus `k = 256` for the inertness gate.

## 4. The gates

**`G-E41A` — inertness.** At `k = E` every partition must be bit-inert: BPB equal to dense within
`1e-6`. A partition that changes the answer while dropping nothing is not a partition, it is a
bug. (E38's `G-E38A`, re-run per partition.)

**`G-E41B` — the anchor.** Unmasked fp32 must read E22's `base` `0.767595 ± 0.001`, or the
session is not measuring the published object and stops.

**`G-E41C` — the planted control, and it must be BIT-EXACT.** `PERM` is `D0C` with the group IDs
permuted: the *sets of neurons* are identical, only their names change. **Under the `oracle` it
must therefore read EXACTLY `D0C`'s BPB — difference 0.0, not "small".** If permuting names moves
the number, the instrument is selecting on labels rather than on neurons and **every partition
comparison below it is meaningless.** (Under `fitted` it may legitimately differ, because the
router is refit and ridge is not permutation-equivariant in general; that difference is reported,
not gated.)

## 5. The verdict cell and the bands, named before the run

**The cell is the BEST partition's `fitted` BPB at `k = 16`** — the attainable selector, at the
activation E40 says 50 tok/s buys — against `D0C`'s measured **3.597108**.

| band | name | what it would mean |
|---|---|---|
| **≤ 1.005** | `PARTITION-IS-THE-LEVER` | **usable at the rate the box affords — the goal is reachable post-hoc and the T4 ask is withdrawn** |
| 1.005 – 2.50 | `PARTITION-HELPS-A-LOT` | a large real gain; worth a second probe before GPU |
| 2.50 – 3.45 | `PARTITION-HELPS` | beats `D0C`'s own **oracle** (3.449466) with an attainable router |
| 3.45 – 3.597 | `PARTITION-HELPS-A-LITTLE` | beats `D0C`'s fitted router but not its ceiling |
| **> 3.597** | **`PARTITION-IS-NOT-THE-LEVER`** | **one label family was as good as any; the T4 ask stands, strengthened** |

**The 1.005 bar is not invented.** E22 measured its `QO512+V52` fp32 arm at **1.005039 BPB** and
that same arm scored **126/160** teacher-forced — a measured anchor in this programme for "still
predicts", not a threshold chosen to be reachable.

## 6. Predictions — fixed here, before the run

1. **All three gates fire, and `G-E41C` is EXACTLY zero**, not merely small.
2. **`PARTITION-IS-NOT-THE-LEVER` is what I expect to be wrong about, so I am registering against
   myself**: I predict the best partition lands in **`PARTITION-HELPS`**, specifically
   **2.90 BPB** at `k = 16`. Better than `D0C`'s fitted 3.597 and better than its oracle 3.449,
   but nowhere near 1.005.
3. **`CONC` wins at `k = 3` and does NOT win at `k = 16`.** Concentrating mass pays most when you
   may keep almost nothing; by 6.25% the heavy neurons are already inside the selection whatever
   the grouping.
4. **`COACT` beats `D0C` at both `k`.** Grouping by co-activation is the structure a linear router
   can actually predict, and `D0C`'s family was never designed for that.
5. **`STRIPE` is the worst attainable partition at every `k`**, worse than `RAND` — deliberately
   equalising mass across groups is exactly wrong when groups must be dropped, and it should be
   worse than grouping at random.

## 7. What E41 will NOT be able to claim

- **Nothing about ternary.** fp32, like E38 and for the same reason: E37's window was 0.5941 BPB
  wide and could not resolve effects this size.
- **Nothing about training.** No healing, no straight-through, no gradient ever taken. If a
  partition helps, whether *training* into it helps more is still H0/H1's question.
- **Nothing about 10 B.** One donor at 1.5 B. The activation *fraction* is what transfers, not
  the BPB.
- **Nothing about criteria beyond these five.** E38 §7 item 1 (a criterion sweep for the
  *selector*) stays owed; this probe moves the *partition*, which is the other half.
- **One `E`.** 256 groups throughout. Group count is a third axis and is not touched.

---

# ADDENDUM A — `G-E41C` went VOID, and the cause is a TIE, not a bug

**Written after the sweep, before addendum B's runner exists. §0–§7 above are unedited.**
This addendum reports a forensic measurement of a gate that already fired. **It does not re-run
any cell to a pass** — E40 addendum A's precedent, and E14 §6.

## What fired

```
G-E41C  PERM oracle == D0C oracle EXACTLY ?  -> VOID
        k=16  diff -1.221e-03     (D0C 3.449466, PERM 3.448246)
        k=3   diff +0.000e+00     (D0C 4.131817, PERM 4.131817)
```

`§4` demanded bit-exactness and `§6` prediction 1 registered it. **Prediction 1 is a MISS and is
scored as one.**

## What actually happened — measured, not argued

`e41_g41c_diag.py` replays the `D0C` oracle trajectory at `k=16` on the registered 24×512 slice
and, at every masked layer, computes the `PERM` selection **from the identical activations**.
Until the first divergence the two runs are the same run, so the first row it finds is the cause
of the whole `1.22e-3`.

| quantity | measured |
|---|---|
| group mass bit-identical under relabelling | **True**, max abs diff **0.000e+00** |
| rows (layer × token) examined | 344,064 |
| rows whose KEPT SET differs | **2** (5.8e-6 of rows), in layers **1** and **11** |
| rows with an EXACT tie at the `k`-th/`(k+1)`-th boundary | **2** |
| differing rows that ARE exact ties | **2 of 2** |
| first divergence | seq 2, layer 11, row 175; masses around the boundary `… 7.731276, 6.865127086639404, 6.865127086639404, 6.751404 …`, relative gap **0.000e+00** |

**My first hypothesis was wrong and is recorded as wrong.** I expected float32 accumulation order
inside `mass.index_add_(1, lab, flat.float()**2)` to make the masses themselves differ. It does
not: permuting labels does not change which neurons land in a group nor the order they are
summed in, and the masses come out **bit-identical**. A two-minute synthetic check said so before
the model was loaded, and the full-slice run confirms it at 344,064 rows.

**The real cause is that the oracle is not a single-valued function.** Two groups carried
*bitwise equal* float32 mass astride the `k`-th boundary; `topk` breaks that tie by index, and
permuting the names changes which index is smaller. Two flipped selections out of 344,064 — one
in layer 1, one in layer 11 — cascade through the remaining depth into `1.22e-3` of BPB.

Exact float32 collisions at that rate are what the arithmetic predicts, not a surprise: near
`6.87` the float32 spacing is `~4.8e-7`, so a handful of collisions among a few boundary
candidates over 344,064 rows is the expected order of magnitude. **`k=3` saw none**, which is why
it read exactly zero, and the 2-sequence smoke saw none, which is why it fired there.

## The repair, verified

Accumulate group mass in **float64**. Re-running the same diagnostic with `--f64`, on the same
slice:

```
group mass bit-identical under relabelling: True  (max abs diff 0.000e+00)
rows whose KEPT SET differs        : 0  (0.000e+00 of 344064)
rows with an EXACT tie at boundary : 0
layers with any divergence: none
```

**`e38_oracle_ceiling.py` is NOT edited.** Its results are published and E38's numbers must keep
reproducing bit-for-bit; the repair is specified here for the next probe that needs an oracle,
which must carry it in its own subclass.

## What this does and does not license

- **The gate stays VOID.** It is not re-run to a pass. It was asked for bit-exactness, it did not
  get it, and that is what it is for.
- **It gives the instrument a measured jitter floor of `1.2e-3` BPB on oracle cells.** Every
  oracle difference E41 reports is 0.35–0.81 BPB, i.e. **290–660×** the floor; the tightest
  comparison in the whole probe — `D0C` 3.804346 against `CONC` 3.815911 at the verdict cell — is
  **9.5×** the floor. No conclusion in this probe is inside the jitter.
- **The verdict cell is not an oracle cell and shows zero tie flips.** `PERM fitted` reproduced
  `D0C fitted` to all sixteen digits at **both** `k` (3.8043462251479396 and 4.574331071853627).
  That is a planted control passing on the selector that decides — **and because it was not
  registered as a gate it may not be promoted to one** (E14 §6). It is reported, not counted.

---

# ADDENDUM B — the band's reference number and E41's own numbers were fit on different routers

**Pre-registered. Pushed before the runner exists.** Additive; nothing above is edited.

## The defect, and it is mine

`§2` registered the calibration slice as `"calib"`, **seed 424242** — which is
`e38_oracle_ceiling.NCAL/SEEDCAL = 8, 424242`, the slice E38 used **only for its `static`
selector**. But `§5`'s band boundary, `D0C` fitted **3.597108**, was measured by E38 with the
router npz `e37_fit_routers.py` wrote, and **that fit used `e23_router.NCAL/SEEDCAL = 32, 42424`
— four times the calibration data, on a different slice.**

So every `fitted` number in E41 comes from a router fit on 8 sequences and is compared to a
boundary set by a router fit on 32. The size of the mismatch is measured:

| | `D0C` fitted, `k=16` |
|---|---|
| E38, router fit on 32 seqs (seed 42424) | **3.597108** |
| E41, router refit on 8 seqs (seed 424242) | **3.804346** |
| difference | **+0.207238** |

**That is 29% of the entire spread E41 measures across all six partitions** (0.726138). The
instrument difference is larger than most of the effects the instrument is used to compare.

**The registered verdict is unaffected in direction** — the best partition at the verdict cell
*is* `D0C`, the control, so "no partition beats the control" holds on E41's own instrument
whatever the absolute. But whether a *better-fit* router reorders the partitions is an open
question, and it is exactly the question the T4 ask turns on.

## What B runs

Refit all six routers with **`NCAL, SEEDCAL = 32, 42424`** — `e23_router`'s own constants, the
ones E38's boundary was fit at — and re-run **only the six `fitted` cells at `k = 16`**. Model,
slice, labels, partition construction, `Carve`, eval loop: unchanged. ~40 minutes of CPU.

## The rule that governs it, inherited verbatim from E36

> **Run 1 is the registered measurement and its verdict stands.** B's only question is whether
> the *ordering* of partitions depends on the router's calibration budget. **If the ordering is
> unchanged the verdict is UNCHANGED — B may not promote it over the bar no matter what number it
> prints.** If B's best partition crosses **3.597**, both are reported and **the verdict cell is
> declared unresolvable**; in that case E41 does not strengthen the T4 ask and must say so.

## The gate

**`G-E41D` — the planted control, and it is a known positive.** `D0C` refit at 32/42424 must
reproduce **E38's published 3.597108 within ±0.001**. If it does not, the calibration budget is
not what separates the two numbers, B has explained nothing, and B's table is reported as
uninterpretable rather than as a correction.

## Predictions — fixed here, before the runner exists

1. **`G-E41D` fires**: `D0C` at 32/42424 reads `3.597108 ± 0.001`.
2. **The ordering at `k=16` fitted is unchanged**: `D0C` < `CONC` < `COACT` < `STRIPE` < `RAND`.
3. **No partition crosses 3.597.** The closest is `CONC`, and I predict its gap above `D0C`
   stays near the `+0.011565` measured at 8 sequences — **`+0.012 ± 0.030`**.
4. **`COACT` gains the most from the larger fit.** It has the best oracle (3.095548) and the
   worst realization gap (0.791455 against `D0C`'s 0.354880), so more calibration should help it
   most — **I predict it closes to within `+0.04` of `D0C` and still does not overtake it.**
5. **`RAND` stays last, and `STRIPE` crosses BACK below the chance line.** At 8 sequences *two*
   partitions read above chance (`RAND` 4.530484 and `STRIPE` 4.415238) — I am predicting the
   larger fit moves `STRIPE` below **4.069819** and leaves `RAND` as the only one above it. That
   asks `STRIPE` for `-0.346` where `D0C` gained `+0.207`'s worth going the other way, so it is
   the prediction here most likely to be wrong, and it is registered because of that.
