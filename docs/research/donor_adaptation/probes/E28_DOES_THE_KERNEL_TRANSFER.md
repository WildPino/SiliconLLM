# E28 — does E13's kernel lever reach the goal's shape?

**Brief**: `briefs/BRIEF_E28_THE_OTHER_FACTOR.md`, pushed at `bb48974` **before the runner
existed**.
**Code**: `engine/e28_kernel_transfer.py`, committed at `a067875` (`--gates-only` at `9ef0754`).
**Result**: `engine/results/e28_kernel_transfer.json`; deterministic-only run at
`engine/results/e28_kernel_transfer_gates.json`.
**Cost**: 254 s, six arms, three interleaved reps, `VOID_AS_A_TIMING: false`. Idle box —
witness **5.5% mean / 14% peak** before the run against the 25% bar.

## VERDICT — `CONTAINER-COSTS`. **The numerator was never a constant, and it is 23.5% larger than the number every budget table in this programme is built on.**

E18 through E27 all attacked one side of the goal's arithmetic:

```
tok/s = (G active weights per second) / (G active weights per token)
              ^ THE ENGINE                      ^ THE MODEL
```

Every probe since E18 shrank the **denominator**. The numerator has been the constant **49.9
G-w/s** since E10, and E25 measured it inside **1.54%** across arms that differed by 15% — which
is precisely what licensed reading tok/s off a weight count. **E28 is the first time it was
measured at the goal's shape with the fastest kernel we own.**

| arm | shape | kernel | mean tok/s | spread | **charged G-w/s** |
|---|---|---|---|---|---|
| `S15-PACKED` | S15 | packed | 29.30 | 6.2% | 45.23 |
| `S15-LUT` | S15 | `--lut` | 22.54 | 17.4% | 34.79 |
| **`S15-LUTBLK`** | S15 | `--lutblk` | **36.80** | 9.4% | **56.81** |
| `T10-PACKED` | T10 | packed | 4.35 | 6.7% | 46.16 |
| `T10-LUT` | T10 | `--lut` | 2.49 | 8.4% | 26.37 |
| **`T10-LUTBLK`** | T10 | `--lutblk` | **5.81** | 12.4% | **61.64** |

**`61.64` against `49.9` is `1.235×`.** The constant was a property of *one kernel and one
layout*, not of the machine, and nobody had run the other kernel at this shape because of the
obstruction in §2.

**And the lever GROWS with the shape** — this is the only quantity in the whole donor programme
that has transferred *upward*:

| shape | `--lutblk` ÷ packed, end to end | source |
|---|---|---|
| 0.5 B | **1.217×** | E13 §, published |
| S15 (1.5 B) | **1.258×** | here, paired |
| **T10 (10 B)** | **1.338×** | here, paired — the verdict cell |

The registered verdict cell was named in the brief before the run: `T10-LUTBLK ÷ T10-PACKED`,
with bands `> 1.30 CONTAINER-COSTS`, `1.10–1.30 KERNEL-TRANSFERS`, `< 1.10 NO-TRANSFER`. It reads
**1.3354** on means of means (**1.3378** paired per rep) → **`CONTAINER-COSTS`**.

## 1. The verdict NAME is not robust to the dispersion. The direction is.

I registered a boundary at 1.30 and the reading sits 0.035 above it, with a per-rep spread that
straddles it:

| rep | `T10-LUTBLK` | `T10-PACKED` | ratio | witness that rep |
|---|---|---|---|---|
| 1 | 5.35 | 4.45 | **1.2022** | 14.7% mean / 30% peak |
| 2 | 6.07 | 4.45 | **1.3640** | 0.3% mean / 1% peak |
| 3 | 6.02 | 4.16 | **1.4471** | 3.7% mean / 8% peak |

**Rep 1 alone would have read `KERNEL-TRANSFERS`, not `CONTAINER-COSTS`.** It is also the rep with
the dirtiest witness, but I am not dropping it: that would be choosing the reps after seeing the
answer, and the registered rule was the mean over three. So the honest statement is:

> **The lever at T10 is somewhere in `1.20–1.45`, central estimate `1.34`. Which of the two
> registered band-names it earns depends on dispersion I did not control well enough. Every rep
> is above 1.20, and the known-positive fires, so the EXISTENCE and rough SIZE of the lever are
> solid; the band name is not.**

The two anchors both pass, which is what makes the arms comparable at all: `S15-PACKED` reads
29.30 against E25's published 29.70 (**−1.35%**), `T10-PACKED` reads 4.35 against E25's 4.6967
(**−7.31%**), bar ±10% (`G-E28A`, `G-E28B`). The T10 anchor being 7.3% low is why the absolute
tok/s here carry the standing ±5% and the RATIOS do not — the arms are interleaved within each
rep exactly so common-mode drift cancels, and the paired ratio (1.3378) and the ratio of means
(1.3354) agree to 0.2%.

## 2. Why nobody had run this before — read in the source, not guessed

```c
if(g_lut){
    if(M.quant!=2) die("--lut requires a --quant packed model (the tile-major copy is a transpose of those bytes)");
```

`donor_engine.c:1437`. **The fastest kernel we own refuses to load the containers that every
quality lever in this programme produces**: `quant==3` is E25's tagged/factored (rank) and
`quant==4` is E26's tagged-v2 (carve). Every artifact on disk at the goal's shape was one of
those two, so 49.9 G-w/s was measured in the only container the fast kernel declines.

The T10 packed artifact had to be built for this probe: `GATE V3: 579 tensors, 5,849,628,724
bytes, matches E1's independent layout exactly` — 1,348 bytes smaller than E25's tagged T10, which
is the container tags and nothing else.

## 3. Gates

| gate | demanded | read | |
|---|---|---|---|
| `G-E28A` | `S15-PACKED` within ±10% of E25's 29.70 | 29.30, **−1.35%** | **PASS** |
| `G-E28B` | `T10-PACKED` within ±10% of E25's 4.6967 | 4.35, **−7.31%** | **PASS** |
| `G-E28C` | `--lut` and `--lutblk` byte-identical logits (sha256, not parity) | S15 `6fcf94f2…`, T10 `484d011a…`, **identical on both shapes** | **FIRES** |
| `G-E28D` | known-positive: `S15-LUTBLK ÷ S15-LUT ≥ 1.50` | **1.633** | **FIRES** |

`G-E28C` is what makes the timing readable at all: the two kernels hold the same bytes in the same
tiles and differ only in the address of tile `b` at step `t`, so a speed difference between them
cannot be a numerics difference. `G-E28D` is the planted control — the instrument must show it can
see a lift it already knows is there before its other readings count.

## 4. `--lut` on its own is still a trap, and it gets worse with scale

| shape | `--lut` ÷ packed |
|---|---|
| S15 | **0.768** |
| T10 | **0.571** |

E11 read `NO-LIFT` (0.391 per weight at the 512 MB cell) and E13 diagnosed it as a layout artifact.
**Both hold at the goal's shape, and the penalty deepens**: plain `--lut` at T10 runs at 57% of the
packed default. The entire value of the LUT path is E13's blocked layout — `--lutblk ÷ --lut` is
**1.649** at S15 and **2.341** at T10.

## 5. What this does to the budget — and the condition attached to it

The budget every table in this programme uses is `50 tok/s ⇒ ≤ 0.998 G active/token`, which is
`49.9 ÷ 50`. On the measured numerator it becomes **`61.64 ÷ 50 = 1.2328 G/token`, 23.5% more
room**:

| arm (E27's table) | T10 floor | `k` of 256 at 0.998 G | `k` of 256 at 1.2328 G |
|---|---|---|---|
| `base` (dense, 28 L) | 2.1978 G | −36.3 | −29.2 |
| `L21-MINRES` | 1.6819 G | −27.6 | −18.1 |
| `QO192+KV96` | 0.7130 G | +8.6 | **+15.7** |
| `FLOOR-MIN` | 0.5683 G | +17.3 | **+26.8** |

**THE CONDITION, and it is not a footnote: not one of those arms can use the kernel that produced
the number.** `QO192+KV96` and `FLOOR-MIN` are rank arms — `quant==3` — and §2's guard refuses
them. `FLOOR-MIN`'s `k` is a carve, and E26 part B measured that a *gathered* weight does not
convert at the streamed rate anyway (band 12.54% at T10, machinery −6.36% before a single group
drops). So the right reading of this table is:

> **This is the budget that WOULD be available if the fast kernel composed with the quality
> levers. Today it composes with none of them, and that gap is now a measured 1.34×, not a
> curiosity.**

## 6. Against the goal — registered in the brief before the run, and it holds

Prediction 5 of the brief said that even the best case would not reach the goal. It does not:

| | tok/s at T10 |
|---|---|
| dense, packed (E25) | 4.70 |
| dense, `--lutblk` (here) | **5.81** |
| E27's best COMPARABLE arm `L21-MINRES`, scaled by this lever | **10.35** |
| **the goal** | **50** |

**Still short by 4.83×.** The numerator moved the gap from `10.6×` to **`8.60×`** (530 G-w/s needed
against 61.64 measured) and that is the whole of what it can do. It is the first time since E10
that the gap has closed from the engine side at all.

## 7. What E28 does NOT establish

- **Nothing about quality.** `--lutblk` quantizes activations to int8 (`AQ`), and E14 measured
  what that costs (`CHEAP-BUT-NOT-NEUTRAL`). `G-E28C` proves `--lut` and `--lutblk` agree with
  *each other*, **not** that either agrees with the packed default. No BPB was taken here.
- **Nothing about composition.** Every arm is a plain packed container with a dense FFN. Whether
  the lever survives rank or carve is exactly what §2's guard prevents anyone from measuring.
- **The band name.** See §1: `1.30` sits inside the per-rep range.
- **Nothing about the 7 B or any real donor.** Both shapes here are synthetic (`--codes mixed`),
  as E25's were.

## 8. What this owes forward

1. **The guard at `donor_engine.c:1437` should be per-matrix, not per-model** — and the reading
   in §2 says the cost of not doing it is concentrated where it hurts most. In a `quant==3` rank
   artifact the FFN tensors are `MK_PACKED` and `matvec_sel` already recurses through a factored
   matrix into two packed halves, so a tile-major replica of those would be used **with no kernel
   change at all**; `build_tm`'s only genuine obstruction is `m->tposed`, the carve. At T10 with
   E27's rank the dense FFN is 92.7% of the token — the fast kernel is being withheld from nearly
   the whole token by a check about the other 7%. **This is the next probe and it is worth 1.34×.**
2. **Re-run this on a cold box with cooldowns between reps** so the band name is decidable. E26
   part B owes the same thing.
3. **Parity of `--lutblk` against the packed default, end to end**, before any number from it is
   quoted as a model's speed rather than a kernel's — E14's int8-activation cost applies and has
   never been measured at T10's shape.
4. **The E25 anchor as standard equipment** — it worked here exactly as intended: it is the reason
   this record is usable, and the reason §1 can be honest about what is not.
