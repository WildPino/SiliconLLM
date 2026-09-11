# BRIEF E31 — what does a GATHERED byte cost, and at what granularity does the penalty go away?

**Pre-registered. Pushed before the runner exists and before the instrument has ever been timed**
(`e31_gather.c` is written and compiles, and `--selftest` — a correctness check that prints no
timing — passes). Nothing here may be edited after the push; the result document scores it as
written.

---

## 0. Why this probe, and why it is now the only kind of probe left

E30 closed the numerator by measurement: `BW-CEIL = 36.30 GB/s` on this box, `T10-LUTBLK` already
at **0.909** of it, a perfect kernel worth **6.83 tok/s** at the goal's shape. Its disposition was
written before the run and it applies: **the only variable left is bytes per token**, against a
hard target of

> **1.45 G moved weights/token — 0.726 GB — for 50 tok/s.**

Every route to that target reads a **subset** of each matrix: MoE, the carve, structured sparsity.
And **that number was derived from a DENSE stream.** `0.726 GB ÷ 36.30 GB/s = 20.0 ms` is exactly
50 tok/s *only if the subset is delivered at the dense rate*.

A subset is not read the way a matrix is read. The hardware moves whole cache lines, the
prefetcher loses the stream, DRAM row buffers stop hitting. **So the denominator has a ceiling of
its own, and nobody in this programme has measured it.** E26 saw its shadow — a carve costs
`-6.36%` before a single group is dropped, and S15's 768-byte rows suffer more than T10's 2048-byte
rows — but E26 read it end-to-end in tok/s, where it is entangled with the router, the kernel and
the container.

## 1. The question

> **If a token needs a fraction of a matrix, in contiguous groups of `G` bytes, how many of the
> bytes it ASKED FOR arrive per second — and how large must `G` be before that equals the dense
> rate?**

Everything is in the **moved-byte** convention on both sides of every ratio, and "useful bytes"
means *bytes the arm requested*, never bytes the hardware happened to move.

## 2. The instrument

`benchmarks/donor_adaptation/engine/e31_gather.c`: a 2 GB buffer, groups of `G` bytes, one group
in eight selected by partial Fisher–Yates (256 MB useful, far outside the 16 MB L3 line E30
located), summed with four accumulators per thread (E8's law), volatile sink, `--threads 6`,
built with the engine's own flags and never `-ffast-math`.

Granularities, each in the list for a reason:

| `G` | what it is |
|---|---|
| 64 | one cache line — the floor |
| 128 | the Zen 2 L2 prefetch pair |
| 256 | a 512-trit row |
| **768** | **one S15 FFN row** (1536 trits packed) — the shape E26 measured as suffering MORE |
| **2048** | **one T10 FFN row** (4096 trits packed) — the shape E26 measured as suffering LESS |
| 8192 … 2097152 | coarser carve groups: 4, 16, 64, 256, 1024 T10 rows |

Three orders per granularity:

* **`sorted`** — the selected groups in ascending address order. What a sane engine does once it
  knows which experts fired. **This is the arm the verdict is read off.**
* **`random`** — shuffled. What a naive gather does.
* **`contig`** — the same *volume* as one unbroken run. **The planted control.**

## 3. The gates

**`G-E31A` — THE PLANTED CONTROL, and it is a same-volume control.** `contig` reads exactly as
many useful bytes as the gathered arms. It must come back at the dense rate: within **±15%** of
E30's 256 MB reading (**37.2 GB/s**), at every granularity. If it does not, the instrument is
measuring *volume* rather than *granularity* and **nothing else it prints may be believed** —
the probe is VOID.

**`G-E31B` — THE KNOWN POSITIVE, taken from an independent measurement of ours.** E26 measured
the carve's locality cost at two shapes and read S15's 768-byte rows as suffering more than T10's
2048-byte rows (band `18.00%` vs `12.54%`, disjoint intervals), and attributed it to exactly this
mechanism. **`sorted` at 768 B must read below `sorted` at 2048 B.** An instrument that cannot
reproduce a difference E26 already saw end-to-end has not earned its nulls, and the probe is VOID.

**`G-E31C`** — `--selftest` must pass in the same session: the gather reads exactly the selected
groups and nothing else, checked against a serial reference at a group size that is neither a
multiple of the unrolled stride nor a cache line.

## 4. The verdict cell, named before the run

**`r_T10` = `sorted` at 2048 B ÷ `contig` at 2048 B** — the fraction of the dense rate that
survives gathering at the granularity of one T10 FFN row.

| band | name | what it would mean |
|---|---|---|
| **≥ 0.85** | `GATHER-IS-FREE` | row-granular sparsity keeps E30's budget: 1.45 G/token stands |
| **0.50 – 0.85** | `GATHER-COSTS` | the real sparse budget is `r_T10 × 1.45` G/token |
| **< 0.50** | `GATHER-DOMINATES` | row granularity cannot pay; the activation unit must be coarser, and §5's crossover says by how much |

**The derived number this probe exists to produce** is the *crossover*: the smallest `G` at which
`sorted` reaches **0.90** of `contig`. That is the minimum activation unit this machine will
serve at full speed, in bytes, and it converts directly into "how many consecutive neurons must
fire together".

## 5. Predictions — fixed here, before the instrument has been timed

1. **`G-E31A` fires**: `contig` within ±15% of 37.2 GB/s at every granularity, and the `contig`
   readings within ±15% of each other.
2. **`G-E31B` fires**: `sorted` at 768 B reads below `sorted` at 2048 B.
3. **The verdict cell lands in `GATHER-COSTS`, `r_T10` in `0.50–0.85`.** A 2048-byte run is 32
   cache lines, enough for the DRAM row buffer but not for a stride prefetcher that never sees a
   stride.
4. **The crossover to 0.90 falls between 2 KB and 32 KB** — i.e. between 1 and 16 T10 FFN rows.
5. **`sorted` beats `random` by ≥ 1.3× at 64 B, and the two converge to within 5% at ≥ 128 KB**,
   because once a group is large enough the order of the groups stops mattering.
6. **Registered in advance and independent of every band above**: E30's `1.45 G/token` is an
   **upper** bound that assumed dense streaming. **The sparse budget is `r × 1.45` and no reading
   of this probe can raise it above `1.45`.** Whatever the bands say, the 10 B goal gets *harder*
   here, never easier.

## 6. What this decides, either way

* **`GATHER-IS-FREE`** ⇒ the MoE/carve road keeps the full 1.45 G/token, the activation unit can be
  as fine as one row, and the design question is purely "which 13.7% fires".
* **`GATHER-COSTS`** ⇒ every budget table in this programme that prices a sparse arm is optimistic
  by `1/r_T10`, including E26's already-corrected ones, and **the exporter's group size becomes a
  first-class design parameter** — which is exactly the lever E26 §9 left owed ("a coarser carve,
  and it is a change to the EXPORTER, not the engine").
* **`GATHER-DOMINATES`** ⇒ fine-grained sparsity is not a road on this machine at all, and the
  architecture must activate in large contiguous blocks — which would be a hard constraint on
  Phase 64's design rather than an optimization.

**In all three cases the probe produces the missing half of E30's specification.** E30 said
`~32 M resident + ≤ 1.45 G streamed`. E31 says what "streamed" is allowed to mean.

## 7. What E31 will NOT be able to claim

- **Nothing about quality.** No BPB, no teacher-forced count, no model, nothing exported. This is
  a memory-system measurement on a noise buffer.
- **Nothing about the engine's actual gather code.** It measures what the *machine* will deliver
  to an ideal gather, not what `donor_engine.c` currently achieves. That is deliberate — the same
  separation E30 drew between a ceiling and an engine — and it means a bad reading here is a wall
  while a good reading here is only a permission.
- **Nothing about other machines.** This box, 6 threads, its RAM; the reference floor.
- **Nothing about the router.** Choosing *which* groups fire costs compute and possibly its own
  memory traffic. E23 priced a real router end-to-end; this probe hands it a free oracle.
- **Nothing about writes.** Read-only, for the same reason E30 was.
