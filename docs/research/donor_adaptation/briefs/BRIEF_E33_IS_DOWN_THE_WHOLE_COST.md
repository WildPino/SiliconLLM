# BRIEF E33 — is `down` the whole carve locality cost, and does a coarser carve pay the predicted `1.25×`?

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

---

## 0. Why this probe: it exists to try to kill a claim I made yesterday

E31 measured what a gathered byte delivers on this box. Its §6.1 then applied that curve to the
carve's layout and produced a **desk** conclusion, corrected once already the same day:

> `gate` and `up` are written group-major, so a carve group is **already** one contiguous run
> (26.9 KB at S15 with `E=256`). The fine granularity is **`down`**, which the engine *requires*
> to be `MK_PACKED_T` at `PT_BLK = 64` — **one cache line per selected neuron** — so a group
> contributes `GSZ × 64` bytes: **2,240 B at S15**, in the steep part of the curve.
> **Therefore `down` alone carries the entire carve locality cost, and coarsening the carve from
> `E=256` to `E=16` should be worth `1.25×` at S15 and `1.14×` at T10.**

**That is arithmetic on a measured table, not a measurement**, and it is exactly the kind of
plausible mechanism this programme's own law says must fire on a known positive before it is
believed. E31 §7 item 1 and item 2 both point here.

## 1. The question

> **Does the carve's cost live where E31 §6.1 says it lives, and does making the group coarser
> buy what §6.1 predicts — end to end, in the engine, at two shapes?**

## 2. Arms

All `quant==4` carved artifacts from `synth_export.py --carve E`, **byte-identical activation
volume across `E`** by construction: the exporter keeps `k/E` fixed, so every arm charges the
same active weights per token and only the *contiguity* changes. That is what makes this a clean
locality experiment and not a size experiment.

| arm | shape | `E` | `GSZ` | `gate`/`up` run | **`down` run** |
|---|---|---|---|---|---|
| `S15-E256` | S15 | 256 | 35 | 26.9 KB | **2,240 B** |
| `S15-E64` | S15 | 64 | 140 | 107.5 KB | **8,960 B** |
| `S15-E16` | S15 | 16 | 560 | 430 KB | **35.8 KB** |
| `T10-E256` | T10 | 256 | 56 | 114.7 KB | **3,584 B** |
| `T10-E16` | T10 | 16 | 896 | 1.84 MB | **57.3 KB** |
| `S15-DENSE` | S15 | — | — | contiguous | contiguous |

`k` is set so the **activated fraction is identical across every carved arm of a shape**, and the
exporter records it. Three interleaved reps, reps outermost (E28's methodology, and the fix that
finally let E31 fire). Idle box or the run does not happen.

## 3. The gates

**`G-E33A` — the planted control, and it is a byte-neutral one.** Every carved arm of a shape
charges the **same** active weights per token. E26 registered a byte-neutral control at `k=E` and
measured `−6.80%` against a registered `−0.47%`; that machine cost is real and is *not* what this
probe is measuring. So: **`S15-E256` must reproduce E26's published carved rate within ±10%**, or
this session is not comparable to the measurement it is correcting and the probe is VOID.

**`G-E33B` — the anchor.** `S15-DENSE` must reproduce E25/E28's published dense S15 rate within
±10%, so the carve's cost is measured against a same-session dense number and not a remembered
one.

**`G-E33C`** — every arm's exporter JSON must report the same `active_weights_per_token`. If the
arms do not charge identically, the comparison is a size comparison and the probe is VOID.

## 4. The verdict cell, named before the run

**`S15-E16 tok/s ÷ S15-E256 tok/s`.**

| band | name | what it would mean |
|---|---|---|
| **≥ 1.18** | `LOCALITY-CONFIRMED` | §6.1's mechanism and size both hold; the exporter knob is real and worth building into every carved export |
| **1.06 – 1.18** | `LOCALITY-PARTIAL` | the direction holds, the size is smaller than the desk model — the desk model over-credits `down` |
| **< 1.06** | `LOCALITY-ABSENT` | **§6.1 is wrong a second time**, the carve's cost is not contiguity at all, and E26's locality reading needs a different explanation entirely |

## 5. Predictions — fixed here, before the run

1. **`G-E33A`, `G-E33B`, `G-E33C` all fire.**
2. **The verdict lands `LOCALITY-CONFIRMED`, ratio in `1.18–1.32`.** The desk model says 1.25.
3. **T10's ratio is SMALLER than S15's** — desk model 1.14 vs 1.25 — because T10's `down` run at
   `E=256` is 3,584 B against S15's 2,240 B and is therefore already further up the curve. This
   is the direction E26 measured independently and is the closest thing here to a second known
   positive.
4. **`E64` sits between `E256` and `E16`, nearer to `E16`** — desk model 3.231 vs 3.830 and 3.057,
   i.e. 77% of the way. A monotone-but-saturating curve, not a straight line.
5. **The absolute rates stay far from the goal and this probe cannot move that.** At S15 a `1.25×`
   on a carved arm is a `1.25×` on a number that is already 5–8× short at the goal's shape
   (E27). **Registered now: no reading of this probe changes E30's ceiling or E31's budget**,
   because both are properties of the machine and this is a property of a file layout.
6. **If `LOCALITY-ABSENT` comes back, the correction goes at the top of E31 as a SECOND
   correction**, and E31 §6's disposition is withdrawn rather than softened.

## 6. What this decides

* **`LOCALITY-CONFIRMED`** ⇒ the exporter grows a coarse-group default for every carved export,
  and E26 §9's owed item closes with a measured number instead of a desk one. The `PT_BLK` knob
  (E31's correction banner) becomes the next question, and it is an engine change.
* **`LOCALITY-PARTIAL`** ⇒ the knob is still worth setting, but the desk model over-credits
  `down` and E31 §6.1's arithmetic should not be used to price anything else.
* **`LOCALITY-ABSENT`** ⇒ I was wrong twice about the same mechanism in two days, E26's locality
  attribution is unexplained, and the honest next step is to stop reasoning about this layout
  from tables and instrument the engine's carve path directly.

## 7. What E33 will NOT be able to claim

- **Nothing about quality.** Synthetic weights, no BPB, nothing exported. Pure timing.
- **Nothing about the goal.** Prediction 5 registers this in advance.
- **Nothing about `PT_BLK`.** It is a compiled kernel constant; changing it is a separate build
  and a separate probe.
- **Nothing about a real router.** `k` is fixed and oracle-free here; E23 priced a real router
  end to end and that cost is additive to anything measured here.
- **Nothing about `--lutblk` on these arms.** `donor_engine.c:1437` still refuses the LUT path on
  `quant==4` (E28 §8 item 1, demoted by E30 §6.1), so every arm here runs the packed kernel at
  E30's measured **0.639** of the ceiling.
