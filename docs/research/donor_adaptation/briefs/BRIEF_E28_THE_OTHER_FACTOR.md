# BRIEF E28 — the other factor: is the throughput constant a property of the engine, or of the container?

**Pre-registered. Pushed before the runner exists.** Nothing in this file may be edited after the
push; the result document scores it as written.

---

## 0. Why this exists

The goal is a 10 B at 50 tok/s. At the goal's shape (`T10`, 10.6032 G active weights/token) that is

```
50 tok/s  x  10.6032 G active weights/token  =  530 G active weights/s
```

against a measured **49.9 G active weights/s**. The gap is **10.6x**, and it is a product of two
factors:

```
tok/s  =  (G active weights per second)  /  (G active weights per token)
             ^ the ENGINE's rate                ^ the MODEL's size
```

**Every probe from E18 to E27 attacked the denominator and none attacked the numerator.** E18's
budget, E19's carve, E21's rank, E23's router, E24's depth, E27's floor — all of them are ways to
make the model smaller. The numerator has been treated as the constant `THROUGHPUT_G = 49.9` since
**E10** returned `CORE-BOUND` and reported no streaming headroom.

**But E10's own §5.2 said the opposite about the core**, and said it in a paragraph the programme
then walked past:

> At 52 G-weights/s, 8 lanes per FMA, 6 cores, 3793 MHz, the loop issues **0.29 FMA per cycle per
> core** against a Zen 2 capability of 2 — about **14%**. So "core-bound" here does **not** mean
> "saturating the FP units", and E10 deliberately does not offer a mechanism for the remaining 7×.

E11 then went looking and returned `NO-LIFT` on the `--lut` path. **E13 overturned the reason**:
the collapse was a layout defect, and with `--lutblk` the same kernel became **the fastest weight
kernel this programme has measured** — 103.05 G-w/s at 12 MB, 67.51 at the 512 MB verdict cell,
**1.19x to 2.00x the packed kernel at every footprint**, bit-identical and sha256-gated, and
**1.217x end-to-end through `forward()`** on the 0.5 B.

**E28's question is whether any of that reaches the goal's shape, and the answer is currently
unknown for a structural reason that is worth stating as the finding it is.**

---

## 1. The obstruction, read in the source before anything was measured

`donor_engine.c:1437`:

```c
if(g_lut){
    if(M.quant!=2) die("--lut requires a --quant packed model (the tile-major copy is a transpose of those bytes)");
```

`--lut` and `--lutblk` load **only** `quant == 2`, the plain packed container. They refuse:

- **`quant == 3`** (`tagged`) — E25's container, the one that makes a **rank** cut expressible.
- **`quant == 4`** (`tagged-v2`) — E26's container, the one that makes a **carve** expressible.

**So the fastest kernel in this engine does not compose with either of the two levers the entire
programme is built on.** That is not a physical limit; it is an unwritten `build_tm` case.

And the artifacts confirm the gap has never been probed. Every `T10`-shaped file on disk —
`e25_t10_tag_r0`, `e25_t10_r256/r512/r2048`, `e26_t10_dense`, `e26_t10_carve256` — is `tagged` or
`tagged-v2`. **There is no packed `T10` artifact in existence, so `--lutblk` has never run at the
goal's shape and cannot, with anything that exists today.**

`THROUGHPUT_G = 49.9` was therefore measured on **a container the fast kernel refuses to load.**

---

## 2. The question, in one line

**Is `49.9 G active weights/s` a property of this engine at this shape, or a property of the
container E25 happened to measure it in?**

---

## 3. What will be built

One new artifact, from the exporter that already exists, with **the same shape, the same seed and
the same code distribution** as `e25_t10_tag_r0` so that the only difference is the container:

```
python synth_export.py --shape T10 --out /d/_ktmp/e28/e28_t10_packed.bin --codes mixed --seed 1234
```

(no `--tagged`, no `--v4`, no `--rank` ⇒ `QUANT_PACKED`, quant == 2, ~5.85 GB)

`S15` needs nothing built: `e25_s15_packed.bin` already exists, is `quant == 2`, and **carries two
independently published rates on it** — E25's 29.70 tok/s and E26's 26.79. It is the anchor.

**No change to `donor_engine.c` in this probe.** Whether to teach `build_tm` the tagged containers
is a decision E28 informs and does not pre-empt.

---

## 4. The arms

Six cells, two shapes × three kernels. `--threads 6`, `--fuse` off everywhere (E26's protocol, so
the region count is not a hidden variable), **≥ 3 repetitions interleaved by rep**, dispersion
printed with every rate.

| arm | shape | kernel | what it is for |
|---|---|---|---|
| `S15-PACKED` | S15 | default packed | **the anchor** — must reproduce E25's 29.70 / E26's 26.79 |
| `S15-LUT` | S15 | `--lut` | E13's shipped layout, the control that should be SLOW |
| `S15-LUTBLK` | S15 | `--lutblk` | E13's blocked layout |
| `T10-PACKED` | T10 | default packed | **must reproduce E25's `T10` dense** — the second anchor |
| `T10-LUT` | T10 | `--lut` | the shipped layout at the goal's shape, never measured |
| `T10-LUTBLK` | T10 | `--lutblk` | **the verdict cell** |

The verdict cell is named here, in advance: **`T10-LUTBLK` ÷ `T10-PACKED`.**

---

## 5. Gates

| gate | demands | why it is here |
|---|---|---|
| **`G-E28A`** | the E25 anchor: `S15-PACKED` within **±10%** of 29.70 tok/s | E26's rule. A CPU percentage is a pre-filter; a reproduced published rate is the instrument. Outside the bar the record is `VOID_AS_A_TIMING`. |
| **`G-E28B`** | the second anchor: `T10-PACKED` within **±10%** of **4.6967 tok/s** -- E25's `T10-TAG-R0`, rates `[4.76, 4.51, 4.82]`, read from `e25_rank_cost.json`'s own row | the new artifact must be the same object E25 measured, in a different container. E25 already showed the container is free at `S15` (`S15-PACKED` 29.70 vs `S15-TAG-R0` 29.50, 0.7% apart), so a miss here means the artifact is wrong and nothing else in the run means anything. |
| **`G-E28C`** | **the planted control**: `--lut` and `--lutblk` produce **byte-identical logits** on both shapes (sha256) | E13's `G-M0`. The blocked layout is the same bytes permuted, so it MUST be bit-identical. If it is not, the layout change is not inert and no timing from it is readable. |
| **`G-E28D`** | **the known-positive on the instrument**: `S15-LUTBLK ÷ S15-LUT` must reproduce E13's DIRECTION (blocked faster at donor-scale footprint) | an instrument must fire on a known-positive before its nulls count. If E13's effect does not reappear on a shape E13 covered, this run cannot speak about a shape it did not. |
| **`G-E28E`** | the contention witness: mean CPU below the bar before the run and after every rep | E26's rebuilt witness, `--selftest`-validated. |

`G-E28C` is the one that makes the rest readable, and it is deliberately a **sha256** gate and not
a parity gate, because a permutation of the same bytes with the same accumulate order has no
licence to move a single bit.

---

## 6. Predictions — fixed here, before the run

1. **`G-E28C` fires on both shapes.** If it misses, `build_tm`'s blocked case is wrong at a shape
   E13 did not cover, and that is a bug report, not a result.
2. **`G-E28D` fires**: `S15-LUTBLK` beats `S15-LUT` by **≥ 1.5×**. E13 measured 3.03× at 512 MB and
   3.45× at 2048 MB; `S15` is 1.7 GB, so ≥ 1.5× is a conservative reading of E13's own curve.
3. **The verdict cell lands between `1.10×` and `1.30×`.** E13's microbench gives 1.19× at 512 MB
   and 1.22× at 2048 MB, and `T10` at 5.85 GB is past both; but Phase 61's law says a
   compute-bound microbench does not compose to a memory-bound engine, and E13's own end-to-end
   number on the 0.5 B was **1.217×** against a microbench 1.7–2.0×. So I expect the engine to
   give roughly what the microbench gives at this footprint, because at 5.85 GB both are
   bandwidth-bound. **Registered alternative: if `T10-LUTBLK ÷ T10-PACKED` exceeds 1.30×, then the
   throughput constant is a container property and every tok/s in this programme is understated —
   and the first thing owed is teaching `build_tm` the `quant == 3` and `quant == 4` layouts.**
4. **`T10-LUT` (shipped layout) is SLOWER than `T10-PACKED`**, by 2–3×, reproducing E11's
   `NO-LIFT` at a shape E11 did not measure. This is the second known-positive.
5. **Even at the top of prediction 3's band, the goal is not reached.** `4.6967 × 1.30 = 6.11 tok/s`
   at `T10` dense, and against E27's best qualitatively-COMPARABLE floor arm (`L21-MINRES`,
   7.75 tok/s) it gives **≈ 10.1 tok/s**. **The 10.6× does not come out of the numerator alone**,
   and E28 will say so in its own verdict rather than leaving the reader to do the arithmetic.

---

## 7. What E28 will NOT be able to claim

- **Nothing about quality.** The weights are noise (E3 §4's Gate V1 authorises a synthetic shape
  for **time** and nothing else).
- **Nothing free.** The LUT path runs **int8 activations** (`AQ`), whose cost E14 already priced
  as `CHEAP-BUT-NOT-NEUTRAL` — 0.625503 below the chance line at the 1.5 B verdict cell. A speed
  win here is a **speed-for-quality trade** that E14 has already put a number on, and E28 must
  quote that number rather than presenting a ratio as a free lunch.
- **Nothing about composition.** `--lutblk` does not load `quant == 3` or `quant == 4`, so E28
  **cannot** measure the fast kernel together with a rank cut or a carve. It measures the dense
  case only, and the composed case stays unmeasured until `build_tm` learns those containers.
- **No absolute rate unless both anchors fire.** E26's lesson: this record is a ratio record by
  default, and an absolute only if `G-E28A` and `G-E28B` both land comfortably rather than at
  their bar.
- **`6.79 tok/s` stays what it is** — the real 7.072 B packed donor, a different artifact.

---

## 8. Three outcomes, registered

- **`CONTAINER-COSTS`** — the verdict cell exceeds 1.30×. The throughput constant is an artifact of
  the container; every budget table in this programme is understated; `build_tm` must learn the
  tagged layouts and E25's invariant must be re-measured on the fast kernel.
- **`KERNEL-TRANSFERS`** — the verdict cell is inside 1.10–1.30×. E13's lever is real at the goal's
  shape, it is worth roughly what it was worth at 0.5 B, the budget line moves by that factor and
  no more, and the composition gap (§1) becomes the thing worth engineering.
- **`NO-TRANSFER`** — the verdict cell is below 1.10×. E13's gain is a small-shape effect, `T10` is
  bandwidth-bound in a way that flattens it, and **the numerator is confirmed closed at ~50 G-w/s**
  — which would be a genuinely useful negative, because it would mean the entire 10.6× must come
  out of the model and E27 has already shown the model cannot give it.
