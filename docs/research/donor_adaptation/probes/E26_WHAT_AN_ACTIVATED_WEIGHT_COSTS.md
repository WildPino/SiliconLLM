# E26 — what an activated weight costs, in the engine

**Brief**: `briefs/BRIEF_E26_WHAT_AN_ACTIVATED_WEIGHT_COSTS.md`, pushed before the runner existed.
**Code**: `engine/donor_engine.c` (`MK_PACKED_T`, `quant == 4`, `--carve-k`), `engine/synth_export.py`,
`engine/e26_carve_cost.py`.
**Result**: part A in `engine/results/e26_parity_carve.json`; part B in
`engine/results/e26_carve_cost.json`. The VOID first attempt is kept as
`engine/results/e26_carve_cost_contended.json`.
**Cost**: part B 243 s of measurement on an idle box, 3 interleaved repetitions, 12 arms, 2 shapes.

## VERDICT — `A GATHERED WEIGHT COSTS MORE THAN A STREAMED ONE`

E25 established the invariant this whole programme's budget arithmetic rests on: at `T10` four arms
whose rates differ by 15% delivered **charged throughput inside a 1.54% band** (49.59–50.36 G active
weights/s). The engine converted active weights into time at a rate that did not care how they were
arranged — which is what makes `2·D·r` the right charge for a rank cut.

**E26 part B asked whether that survives a carve, and it does not.**

| | E25 (rank) | E26 (carve), `T10` | E26 (carve), `S15` |
|---|---|---|---|
| charged throughput band | **1.54%** | **12.54%** | **18.00%** |
| range (G active weights/s) | 49.59–50.36 | 38.03–43.12 | 34.34–41.12 |

A rank cut buys exactly the weights it removes. **A carve does not**, and the shortfall grows as the
carve deepens. **Every `k` in every budget table in this programme — E18 §31, E19, E23 §7, E24, E27
— is therefore optimistic**, by a factor this probe measures for the first time.

**The second result, and it is a new number rather than a correction.** At `k = E` the carve keeps
every group, so it moves the same weights as the dense arm plus the router — byte-neutral to a
**registered** `−0.47%` at `T10` and `−0.71%` at `S15`. Whatever it loses below that offset is the
carve machinery's own cost, and unlike E25's rank machinery **it is resolvable**:

| shape | `k = E` measured | bytes predicted | **the machinery's own cost** |
|---|---|---|---|
| `S15` | −4.98% | −0.71% | **−4.30%** |
| `T10` | −6.80% | −0.47% | **−6.36%** |

E25's planted control at `r = D/2` read `−0.43%` / `−2.46%` with dispersions of 7.3% / 6.4% —
neither resolvable from zero, i.e. the *rank* path's own cost was below what this box can measure.
**The carve path's own cost is not.** It is 4–6%, it is charged to nothing in any table written so
far, and it is paid before a single group is dropped.

---

## §1 — validity of the record, stated before any number is used

The first attempt at part B was run with a game open, 5.82 of 12 cores held, and was **VOIDed**. This
run therefore carries three separate idleness instruments, and the honest reading of them is mixed.

**1. The contention witness, and it was rebuilt for this run.** The first version gated on the
*maximum* of a short sample sequence at a 12% bar, and it refused an idle desktop three times in a
row on ordinary background transients. Lowering a gate in order to pass it is how an instrument
stops being one, so instead: the witness now gates on the **mean** over a longer window, keeps the
max as a recorded diagnostic, and the bar was re-derived from measurement rather than intuition.

`--selftest` is a **planted control on the instrument itself**: read the quiet box, then hold 6 of
12 cores with spinning processes and require the witness to fire.

```
quiet:  12.1% mean / 18% peak      (bar 25% on the mean)
loaded: 61.5% mean / 82% peak      (6 spinning processes, 6 of 12 cores)
FIRES ON THE KNOWN-POSITIVE: YES   (reads the quiet box as idle: yes)
after:   5.3% mean                 (the load is gone)
```

The bar sits at 25% because the two populations are 12% and 62%, and because the failure this
instrument exists to catch was 48% of the box. **The 12% original was inside this desktop's own idle
noise**, which is why it misfired. The derivation, including that admission, is written into
`e26_carve_cost.py` above `IDLE_BAR`.

During the run the witness read **10.6% / 5.8% / 6.5% / 8.0%** mean. The box was genuinely idle.

**2. The E25 anchor, which is the load-bearing test and which nearly failed.** A CPU percentage is a
proxy for idleness; reproducing a rate that was measured on an idle box and published is the thing
itself. E25 ran this exact shape in this exact untagged container and recorded `S15-PACKED` at
**29.70 tok/s** (29.59 / 29.44 / 30.07, spread 2.1%).

```
E25 ANCHOR  S15-PACKED reads 26.79 tok/s against E25's published 29.70 (-9.81%, bar +-10%) -> PASS
```

**It passed by 0.19 of a point against a bar I set myself, and that is not a comfortable margin.**
The witness says the box was not contended, and both `S15` arms decline monotonically across the
three reps (`S15-DENSE` 28.04 → 27.53 → 24.34; `S15-PACKED` 27.27 → 27.32 → 25.77), which is the
signature of thermal drift, not of another process.

**What follows from that, and it constrains the rest of this document:**

- **The absolute rates in this record are ~10% low and are not usable as absolutes.** In
  particular, prediction 3's registered form — *"every carved arm inside 45–55 G active weights/s"*
  — is **untestable from this record**, because the *dense control itself* reads 43.12 G/s where
  E25 read ~49.9. I do not count its failure as evidence of anything.
- **The ratios are usable**, because the protocol interleaves arms by repetition precisely so a
  drift hits every arm equally. This is checkable and I checked it: recomputing every ratio
  **paired within each repetition** and averaging the three pairs changes nothing —
  `T10-K256` reads 0.9312 paired vs 0.9320 as a mean of means, `T10-K4` 4.4038 vs 4.3984, and the
  largest disagreement anywhere in the table is 0.4%. **The drift is common-mode and the ratios
  survive it.**

So: this record is read as a **ratio** record. Every claim below is a ratio or a band. No absolute
tok/s from this run enters any table.

---

## §2 — part A, for completeness

Part A built the thing. `donor_engine.c` gained a transposed block-major packed kind `MK_PACKED_T`
and a `quant == 4` container in which each layer's FFN carries a permutation, a group table and a
row list, driven by `--carve-k`. It was checked on the **real donor** (`Qwen2.5-1.5B`, rev `8faed761…`) at
`E = 256`, `k = 64`, rule `R0`, router seed 26, over 8 tokens. **Three gates**, all fired
(`e26_parity_carve.json`, `all_fire: true`):

| gate | demanded | reading |
|---|---|---|
| `E26-P` | the carved path reproduces the dense path | worst rel. l2 **`4.481e-06`** against a `1e-4` bar, **top-1 `1.0000`** |
| `E26-S` | the permutation is a bijection on every layer | **224 comparisons, 0 disagreements** |
| `E26-R` | the transposed kernel matches the untransposed one | worst rel. l2 **`1.156e-06`** against a `2e-3` bar, top-1 `1.0000` |

**And part B adds a fourth check, which part A could not make**: `S15-PACKED` vs `S15-DENSE` are the same
weights in the *untagged* and the *carve* container respectively, with no carving. They read
**1.0056** (paired 1.0079) against a byte prediction of exactly 1.0000, dispersion 5.8%. **The
container itself costs nothing measurable** — so everything in §3 is attributable to the carve and
not to the format it lives in.

---

## §3 — the measurement

Weights are noise and the router is random: E3 §4's Gate V1 authorises a synthetic shape for **time
and nothing else**. Which `k` a real model survives is E19's, E24's and E27's question.

`active = (q + k + v + o + E·D + 3·D·(F/E)·k)·L + V·D`, the router charged at `E·D` per layer.

| arm | active/token | mean tok/s | spread | measured gain | bytes predict | charged G-w/s |
|---|---|---|---|---|---|---|
| `S15-PACKED` | 1.5436 G | 26.79 | 5.8% | +0.56% | +0.00% | 41.35 |
| `S15-DENSE` | 1.5436 G | 26.64 | 13.9% | — | — | 41.12 |
| `S15-K256` | 1.5546 G | 25.31 | 6.6% | **−4.98%** | **−0.71%** | 39.35 |
| `S15-K128` | 0.9766 G | 35.17 | 9.5% | +32.02% | +58.06% | 34.34 |
| `S15-K64` | 0.6875 G | 52.56 | 7.6% | +97.31% | +124.51% | 36.13 |
| `S15-K16` | 0.4708 G | 76.84 | 9.3% | +188.49% | +227.88% | 36.18 |
| `S15-K4` | 0.4166 G | 92.85 | 6.0% | +248.58% | +270.53% | 38.68 |
| `T10-DENSE` | 10.6032 G | 4.07 | 12.8% | — | — | 43.12 |
| `T10-K256` | 10.6535 G | 3.79 | 16.1% | **−6.80%** | **−0.47%** | 40.38 |
| `T10-K128` | 6.4257 G | 6.05 | 3.3% | +48.85% | +65.01% | 38.90 |
| `T10-K64` | 4.3117 G | 8.82 | 9.0% | +116.89% | +145.91% | 38.03 |
| `T10-K16` | 2.7263 G | 15.19 | 5.0% | +273.52% | +288.92% | 41.41 |
| `T10-K4` | 2.3299 G | 17.89 | 10.0% | +339.84% | +355.09% | 41.67 |

**Every carved arm underperforms its byte prediction, at both shapes, at every depth.** That is the
whole result in one sentence. Net of the machinery offset measured by the planted control:

| arm | gross | net of machinery | bytes predict | still short by |
|---|---|---|---|---|
| `S15-K128` | +32.02% | +38.94% | +58.06% | **19.1 pt** |
| `S15-K64` | +97.31% | +107.65% | +124.51% | **16.9 pt** |
| `S15-K16` | +188.49% | +203.61% | +227.88% | **24.3 pt** |
| `S15-K4` | +248.58% | +266.85% | +270.53% | 3.7 pt |
| `T10-K128` | +48.85% | +59.72% | +65.01% | **5.3 pt** |
| `T10-K64` | +116.89% | +132.72% | +145.91% | **13.2 pt** |
| `T10-K16` | +273.52% | +300.79% | +288.92% | −11.9 pt (over) |
| `T10-K4` | +339.84% | +371.94% | +355.09% | −16.9 pt (over) |

---

## §4 — why the crude test missed it, and what the band actually says

Prediction 3 registered a trigger: *"if the deep arms (`K16`, `K4`) fall below 40 G weights/s, then a
gathered weight costs more than a streamed one."* **`T10-K16` reads 41.41 and `T10-K4` reads 41.67.
The registered trigger does not fire.** I am not going to claim it did.

**But the trigger was mis-specified, and the band shows why.** The deep arms are not where the carve
hurts. Charged throughput relative to each shape's dense control:

| `k` | `T10` | `S15` |
|---|---|---|
| 256 | 0.936 | 0.957 |
| 128 | 0.902 | 0.835 |
| 64 | **0.882** | 0.879 |
| 16 | 0.960 | 0.880 |
| 4 | 0.967 | 0.941 |

**It is a U, not a slope.** At small `k` the FFN is a small share of the token — 5.7% of the active
weights at `T10-K4` — so the arm is almost entirely the dense non-FFN floor, and its charged
throughput returns to the dense rate whatever the gathered path costs. **The dense floor masks the
effect exactly where the registered trigger was looking for it.**

Isolating the gathered FFN's own per-weight efficiency, by attributing the non-FFN floor to the
dense rate and solving `1/r = (1−f) + f/x` where `f` is the FFN's share of active weights:

| arm | FFN share `f` | `x` (gathered FFN efficiency) | interval from the arm's own dispersion |
|---|---|---|---|
| `S15-PACKED` | 0.749 | **1.008** | [0.969, 1.047] ← the null, and it behaves |
| `S15-K256` | 0.751 | 0.943 | [0.903, 0.985] |
| `S15-K128` | 0.603 | 0.754 | [0.701, 0.808] |
| `S15-K64` | 0.436 | 0.760 | [0.705, 0.819] |
| `S15-K16` | 0.177 | **0.564** | [0.479, 0.673] |
| `S15-K4` | 0.070 | 0.526 | [0.422, 0.685] |
| `T10-K256` | 0.794 | 0.921 | [0.831, 1.015] |
| `T10-K128` | 0.658 | 0.858 | [0.838, 0.879] |
| `T10-K64` | 0.490 | 0.786 | [0.724, 0.852] |
| `T10-K16` | 0.194 | 0.825 | [0.740, 0.925] |
| `T10-K4` | 0.057 | 0.621 | [0.390, 1.339] ← **not resolvable, do not use** |

**At `S15` the degradation is resolvable**: `K256` [0.903, 0.985] and `K16` [0.479, 0.673] do not
overlap, and neither do `K256` and `K128`. **At `T10` it is suggestive but the intervals overlap**,
and `T10-K4`'s interval spans 0.39 to 1.34 — a difference of two numbers near 1, and it constrains
nothing. It is in the table so that nobody quotes the 0.621.

**Two things this decomposition is not.** It is a **post-hoc model**, not a registered metric, and
per E14 §6 it does not become a gate — it is a reading that tells a follow-up what to measure
directly. And it cannot separate the *gather's* poor locality from **fixed per-token overhead that
does not scale with `k`** (the top-`k` over 256 groups on every one of 48 layers, and building the
row list). Both would produce this shape. Separating them needs an arm with the selection machinery
run and its result discarded, which E26 does not have.

---

## §5 — prediction 5 lands, and it is the mechanism

Registered as a direction and not a number: *"the `S15` deep arms will disappoint relative to `T10`.
At `S15` a kept row is 768 B and at `T10` it is 2048 B; if scattering costs anything, it costs more
where the runs are shorter."*

**It holds on every comparison available.** The charged-throughput band is **18.00% at `S15` against
12.54% at `T10`**. The gathered-FFN efficiency at `k = 16` is **0.564 at `S15` against 0.825 at
`T10`**, intervals [0.479, 0.673] and [0.740, 0.925] — **disjoint**. And in the net-of-machinery
table the `S15` arms fall short of their byte prediction by 17–24 points while the `T10` deep arms
*overshoot* theirs.

So the cost is a **locality** cost, and it is a function of how long a contiguous run is. That is a
mechanism with a lever attached: it says a carve at a coarser granularity — fewer, longer runs for
the same activated fraction — should recover part of what this probe measures as lost.

---

## §6 — the predictions, scored

| # | prediction | outcome |
|---|---|---|
| 1 | `E26-P` fires | **HIT** — `4.481e-06` |
| 2 | the planted control reads within ±3 pt of its byte prediction; registered alternative at 8% | **MISS** — off by 4.3 pt at `S15` and **6.3 pt** at `T10`. The 8% alternative does **not** fire (it came within 1.2 pt), but the correction is real and must be applied |
| 3 | charged throughput stays on E25's line, every carved arm 45–55 G-w/s; alternative if `K16`/`K4` < 40 | **the absolute form is UNTESTABLE here** (§1 — the dense control itself reads 43.12); **the registered trigger does NOT fire** (41.41, 41.67); **but the primary claim is FALSIFIED by the band** — 12.54% and 18.00% against E25's 1.54% |
| 4 | `T10` saturates ~22 tok/s and the carve alone cannot reach 50; `K64` ≈ 11.6, `K16` ≈ 18.3, `K4` ≈ 21.4, each ±10% | **the headline HOLDS decisively** — `K4` keeps 1.6% of the FFN and reads 17.89 tok/s. On the cell numbers, anchor-corrected for §1's level shift: `K16` 16.84 (band 16.47–20.13) **HIT**, `K4` 19.83 (19.26–23.54) **HIT**, `K64` 9.78 (10.44–12.76) **MISS** |
| 5 | the `S15` deep arms disappoint relative to `T10` | **HIT**, on three independent comparisons (§5) |

Two hits, one miss, one hit-with-a-missed-cell, and one prediction whose registered trigger was
mis-specified while its primary claim was falsified anyway. **The mis-specification is the
instructive one**: I wrote the trigger as an absolute floor on the deep arms, and the deep arms are
precisely where the dense floor hides the effect. A band was the right statistic and it was already
in the brief — I just did not key the alternative to it.

---

## §7 — what this costs the budget tables

The correction is a factor on the **FFN term**, not on the token. Using this shape's own measured
ratio at the relevant depth:

- **E24's `K156` at `T10`** (E24 §6: `6.14 G` ≈ 8.1 tok/s). Interpolating `T10`'s charged ratio
  between `K128` 0.902 and `K256` 0.936 gives ≈ 0.91, so **≈ 7.4 tok/s**, not 8.1.
- **E27's `FLOOR-MIN`** sits at `k = 17.3`, where `T10` measures 0.960 — so its `9.37 tok/s` becomes
  **≈ 9.0**. Small, because at that depth the floor is nearly the whole token.
- **The general statement**: at the depths the 50 tok/s budget actually permits at `T10`
  (`k ≈ 2…18`), the carve correction on the *token* is **3–5%**, because the floor dominates. On the
  *FFN allowance itself* it is **≈ 17%** (`x ≈ 0.83`), i.e. the budget permits about a sixth fewer
  groups than the charge model says.

**None of this changes any verdict.** E19's "FFN-only carving cannot reach the target", E24's
`DEPTH-RECOVERS`, E27's `FLOOR-IS-NOT-ENOUGH` all survive — they are all negative or
quality-conditioned results, and a correction that makes the carve *worse* strengthens every one of
them. What changes is that the tok/s figures attached to carved arms are now known to be optimistic,
and by how much.

---

## §8 — what E26 does **not** claim

- **Nothing about quality.** Noise weights, random router. Which `k` a model survives is E19/E24/E27.
- **No absolute rate from this record.** §1: the anchor passed by 0.19 of a point, the box was ~10%
  slow from thermal drift, and only ratios are used here.
- **`T10` is a shape, not a model.** It carries a 32,768 vocabulary, so its head (134 M) is the
  favourable case; a Qwen-vocabulary 10 B carries 622 M.
- **The gather's locality cost and the fixed selection overhead are not separated** (§4).
- **Nothing about the router's accuracy.** It is charged and executed here, never evaluated.
- **`6.79 tok/s` is untouched** — that is the real 7.072 B packed donor, a different artifact.

---

## §9 — what is owed

1. **Separate locality from fixed overhead.** One arm: run the selection machinery and then use
   every group anyway. The difference from `k = E` is the fixed cost; the rest is the gather.
2. **Test the granularity lever §5 implies.** Same activated fraction, fewer and longer runs. If the
   cost is locality, this recovers part of it, and it is a change to the exporter and not to the
   kernel.
3. **Re-run the anchor on a cold box.** This record passed at −9.81% with a monotone decline across
   reps. A cold-start run with a cooldown between reps would say how much of that is thermal, and
   would make the absolute form of prediction 3 testable.
4. **The bar in `IDLE_BAR` is a coarse pre-filter and should stay labelled as one.** A CPU
   percentage cannot resolve the few percent that separates a good timing from a bad one; the E25
   anchor is the real instrument and it should be adopted by every future timing probe here.
