# BRIEF E30 — is the engine still core-bound, or has the fast kernel put it against the memory wall?

**Pre-registered. Pushed before the runner exists.** Nothing here may be edited after the push;
the result document scores it as written.

---

## 0. Why this probe, and why now

E28 broke the constant this programme priced the goal against: the numerator is not 49.9 G active
weights/s, it is **61.64** at the goal's shape on `--lutblk`, and the lever *grows* with the shape.
That is the first movement from the engine side since E10, and it immediately raises the only
question that matters for the goal:

> **How much more is there?**

The goal needs `530 G-w/s` at T10 and we measure `61.64`. If the remaining `8.60×` is available to
kernel work, the engine is where the programme should spend everything it has. If it is not, then
every further kernel probe is wasted effort and the goal can only be reached by **moving fewer
weights per token** — which is an architecture question, not an engine one.

**E10 answered "core-bound" at 22.5 GB/s and that answer is three levers old.** It predates
`--fuse`, `--threads 6`, E13's blocked tile-major and E28's measurement at the goal's shape.

## 1. Charged weights are not moved bytes, and this probe is about MOVED BYTES

The standing law says the two conventions never meet inside a fraction. E30 works entirely in the
**moved-byte** convention, on both sides of every ratio, and says so in every cell.

The T10 packed artifact's bytes per token, from the shape and cross-checked against the file:

| | bytes |
|---|---|
| per layer: `q` + `o` (`4096×4096`, 2 trits/byte, + fp32 row scales) | 2 × 8,404,992 |
| per layer: `k` + `v` (`1024×4096`) | 2 × 2,101,248 |
| per layer: `gate` + `up` (`14336×4096`) | 2 × 29,417,472 |
| per layer: `down` (`4096×14336`) | 29,376,512 |
| per layer: two norms, fp32 | 32,768 |
| **× 48 layers** | **5,244,321,792** |
| head (`32768×4096`) | 67,239,936 |
| one embedding row (fp32) | 16,384 |
| **total moved per token** | **≈ 5.3116 GB** |

The artifact on disk is 5,849,628,724 bytes; the difference is the fp32 embedding table
(536,870,912), of which exactly one row is read per token. **The arithmetic must reproduce the file
size to the byte or the accounting is wrong and the probe is void** — that is gate `G-E30B`.

On that accounting E28's measured rates are also byte rates, and this is arithmetic on numbers
already published, not a new measurement:

| arm | tok/s (E28) | **GB/s moved** |
|---|---|---|
| `T10-PACKED` | 4.35 | **23.11** |
| `T10-LUTBLK` | 5.81 | **30.86** |
| *the goal, 50 tok/s* | *50* | ***265.58*** |

## 2. The question, and the instrument

> **What is the largest read bandwidth this box can actually deliver to 6 threads, and what
> fraction of it is the engine already using at the goal's shape?**

The ceiling is **measured on this box, not cited from a spec sheet**. A small C benchmark, compiled
with the same compiler and flags as the engine (and never `-ffast-math`), streams a buffer far
larger than L3 and reports GB/s, at the same `--threads 6` the engine uses.

**`G-E30A` — THE PLANTED CONTROL. The bandwidth instrument must FIRE on a known positive before
its ceiling counts.** Probe-3 measured this machine's L3 cliff at exactly 16 MB: a resident buffer
reads at ~100 GB/s and a DRAM-sized one at ~28 GB/s. The bench sweeps 4 MB → 8 GB and **must show
that cliff**. A tool that cannot tell L3 from DRAM is not measuring memory bandwidth, and if the
cliff does not appear the probe is VOID and reports no ceiling.

## 3. Arms

| arm | what |
|---|---|
| `BW-SWEEP` | read bandwidth, 4 MB → 8 GB, 6 threads, 3 reps per size |
| `BW-CEIL` | the DRAM-resident plateau: the median of the sizes ≥ 1 GB |
| `T10-PACKED` | E28's own reading, re-run here for a same-session byte rate |
| `T10-LUTBLK` | E28's own reading, re-run here |

The two engine arms are re-run rather than quoted so that the engine byte rate and the ceiling come
off the same box in the same session at the same temperature. **E28's published values are the
anchor**: each must reproduce within ±10% (`G-E30C`), or the comparison is not licensed.

## 4. The verdict cell, named before the run

**`T10-LUTBLK` byte rate ÷ `BW-CEIL`.**

| band | name | what it would mean |
|---|---|---|
| **≥ 0.85** | `AT-THE-WALL` | the engine is bandwidth-saturated; kernel work is over |
| **0.60 – 0.85** | `PARTIALLY-BOUND` | a bounded amount is left, at most `1/ratio` |
| **< 0.60** | `STILL-CORE-BOUND` | E10's answer still holds and kernel work has real room |

## 5. Predictions — fixed here, before the run

1. **`G-E30A` fires**: the sweep shows the L3 cliff, resident ≥ 3× the DRAM plateau.
2. **`BW-CEIL` lands in `25–45 GB/s`** — dual-channel DDR4-3200 class, 6 threads.
3. **The verdict cell lands in `PARTIALLY-BOUND`, ratio `0.70–0.90`.** I expect the fast kernel to
   have moved the engine most of the way to the wall but not onto it, because E10 found the packed
   path core-bound at 22.5 GB/s and `--lutblk` only bought 1.34×.
4. **`T10-PACKED` ÷ `BW-CEIL` lands below 0.75** — i.e. E10's "core-bound" verdict was right for
   the kernel it tested, and what changed is the kernel, not the machine.
5. **The gap to the goal does NOT close, and the reason changes.** Whatever the ratio reads,
   **50 tok/s at 10.6 G active weights/token needs 265.58 GB/s**, which is at least 6× any reading
   this box can produce. I register now that the conclusion **"no kernel reaches 50 tok/s at this
   shape on this machine"** follows from the ceiling alone and does not depend on the verdict cell.

## 6. What this decides, either way

* **`AT-THE-WALL` or `PARTIALLY-BOUND`** ⇒ the numerator is finished as a research direction. The
  remaining `8.60×` is not available at any kernel quality, and **the only variable left is bytes
  per token**. Every subsequent probe belongs on the denominator, and the honest recommendation for
  the 10 B goal becomes architectural — which is what `SCALEUP_ARCHITECTURE.md` and Phase 64
  already describe.
* **`STILL-CORE-BOUND`** ⇒ there is real headroom in the kernel, the `1/ratio` bound says how much,
  and the per-matrix `--lut` guard (E28 §8 item 1, worth a measured 1.34× to rank and carve
  artifacts) becomes the immediate next build rather than a nice-to-have.

**Either way this probe produces the number the goal has been missing: not "how fast is the
engine" but "how fast can the engine POSSIBLY be on this machine at this shape".**

## 7. What E30 will NOT be able to claim

- **Nothing about quality.** No BPB, no teacher-forced count, nothing exported.
- **Nothing about other machines.** The ceiling is this box's — a 3600X with its RAM, the reference
  floor this programme designs against. A machine with more channels moves the wall and does not
  move the *shape* of the argument.
- **Nothing about the KV cache.** The byte accounting in §1 is weights only, measured at the short
  contexts E28 used (40 tokens). At long context the KV traffic adds to the numerator's load and
  makes the wall **closer**, not further.
- **Nothing about `--lutblk`'s quality cost.** E14 measured that int8 activations are
  `CHEAP-BUT-NOT-NEUTRAL`, and that has still never been measured at T10's shape. E28 §8 item 3
  still stands.
