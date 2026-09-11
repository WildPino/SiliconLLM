# E31 — what does a GATHERED byte cost, and at what granularity does the penalty go away?

**Verdict: `GATHER-COSTS`, `r_T10 = 0.579`.** Reading one T10 FFN row at a time delivers **58% of
the dense rate in the bytes you asked for**. E30's budget of 1.45 G moved weights/token was
derived from a dense stream; at row granularity the real budget is **0.840 G/token — 7.93% of
T10's dense token**. The penalty is not permanent: it is gone by **32 KB, sixteen consecutive
T10 rows**, and that number is the design constraint this probe was built to produce.

**Brief**: `briefs/BRIEF_E31_THE_GATHERED_BYTE.md`, pushed at `8003fa3` before the runner existed
and before the instrument had ever been timed.
**Instrument**: `benchmarks/donor_adaptation/engine/e31_gather.c`.
**Runner**: `benchmarks/donor_adaptation/engine/e31_gathered_byte.py`.
**Result**: `engine/results/e31_gathered_byte.json` (run 4). 12 s, witness **7.2% mean / 14%
peak** against a 25% bar.
**Runs 1–3 are VOID and kept on the repo**; §2 says why, and it is the most useful part of this
document.

Everything is in the **moved-byte** convention on both sides of every ratio, and *useful bytes*
means **bytes the arm asked for**, never bytes the hardware happened to move.

---

## 1. The gates

| gate | what it demands | reading | |
|---|---|---|---|
| `G-E31A` | **same-volume planted control.** `contig` reads the same useful volume as one unbroken run and must come back at the dense rate — within ±15% of E30's 256 MB reading (37.16 GB/s) at every granularity, and within ±15% of itself | worst **−6.5%** (at 128 KB), spread **11.2%** | **FIRES** |
| `G-E31B` | **known positive, from an independent measurement of ours.** E26 read S15's 768-byte rows as suffering more than T10's 2048-byte rows (bands 18.00% vs 12.54%, disjoint). `sorted` at 768 B must read below `sorted` at 2048 B | **18.21 < 21.50 GB/s** | **FIRES** |
| `G-E31C` | the gather reads exactly the selected groups, checked against a serial reference at a group size that is neither a cache line nor a multiple of the unrolled stride | both paths OK | **FIRES** |

## 2. Three VOID runs, and what each one caught

This is on the record because the instrument earned its nulls three times over, and because two
of the three defects were mine.

| run | why VOID | the fix, and why it is STRICTER |
|---|---|---|
| **1** | `G-E31A`: worst **−15.4%**, spread **24.4%**. The control was implemented as a walk through the group-offset array, so **it carried the gathered arms' own per-group bookkeeping** and sagged from 39.96 GB/s at 2048 B groups to 34.34 at 64 B. **A control that moves with the variable it is controlling for is not a control** — and it sat in the *denominator* of every ratio, flattering exactly the small-granularity arms the verdict turns on. | `contig` became literally one unbroken run with no indirection: the stricter reading of the brief's own words. The old behaviour survives as **`contig_ind`**, a diagnostic that prices the bookkeeping instead of hiding it in a control. |
| **2** | `G-E31A`: worst −14.5%, spread **17.4%**. Control now honest; the box was not. Witness 10.2% mean but **29% peak**. | — |
| **3** | `G-E31A`: worst −19.6%, spread **29.5%**, *with reps raised from 3 to 15*. More reps did not help, and that is the diagnosis: `contig` runs **identical code over identical volume** at every row, so 29.89 GB/s at 128 B against 40.38 at 256 B is block-correlated drift, not per-rep noise. Each arm's reps sat in **one contiguous wall-clock window**, so a slow phase took a whole block. | **Reps moved OUTERMOST, every cell visited once per rep.** E28's registered methodology is interleaved reps precisely so drift is common-mode across the arms a ratio is taken over; this instrument had it exactly backwards. |
| **4** | — | spread **11.2%**, worst −6.5%. Both gates fire. |

**The gate was never widened.** It reads ±15% in run 4 exactly as it read in run 1.

## 3. The measurement

2 GB buffer, one group in eight selected by partial Fisher–Yates (**256 MB useful**, far outside
the 16 MB L3 line E30 located), 6 threads, 9 interleaved reps, medians. Four accumulators per
thread (E8's law), volatile sink, engine flags, never `-ffast-math`.

| group | `sorted` | `random` | **`contig`** | `contig_ind` | **`sorted/contig`** | `random/contig` | bookkeeping |
|---|---|---|---|---|---|---|---|
| 64 B — one cache line | 9.65 | 7.20 | 36.60 | 32.45 | **0.264** | 0.197 | 0.887 |
| 128 B | 11.61 | 8.52 | 35.90 | 33.98 | 0.323 | 0.237 | 0.946 |
| 256 B | 13.99 | 8.90 | 35.72 | 34.56 | 0.392 | 0.249 | 0.967 |
| **768 B — one S15 FFN row** | 18.21 | 13.69 | 38.83 | 35.69 | **0.469** | 0.353 | 0.919 |
| **2048 B — one T10 FFN row** | 21.50 | 19.85 | 37.14 | 35.48 | **0.579** | 0.534 | 0.955 |
| 8 KB — 4 T10 rows | 31.31 | 30.87 | 37.21 | 36.90 | 0.842 | 0.830 | 0.992 |
| **32 KB — 16 T10 rows** | 34.37 | 34.12 | 36.41 | 36.94 | **0.944** | 0.937 | 1.015 |
| 128 KB | 34.06 | 34.84 | 34.75 | 34.23 | 0.980 | 1.003 | 0.985 |
| 512 KB | 35.49 | 35.12 | 35.39 | 36.18 | 1.003 | 0.992 | 1.022 |
| 2 MB | 35.58 | 35.65 | 36.02 | 35.66 | 0.988 | 0.990 | 0.990 |

`bookkeeping` is `contig_ind / contig`: the **same addresses** read through the group loop versus
read flat. It is the diagnostic that run 1 mistook for a control, reported here and **never
folded into a verdict**.

## 4. The verdict cell, and what it does to the budget

**`r_T10` = `sorted`(2048 B) ÷ `contig`(2048 B) = `0.579` → `GATHER-COSTS` (band 0.50–0.85).**

At the wall, `tok/s` is linear in bytes per token (E30 §4) — so a gather that delivers `r` of the
dense rate shrinks the budget by exactly `r`:

| activation unit | `r` | budget for 50 tok/s | as a fraction of T10's dense token |
|---|---|---|---|
| one cache line (64 B) | 0.264 | 0.383 G weights | **3.61%** |
| one S15 FFN row (768 B) | 0.469 | 0.681 G | 6.42% |
| **one T10 FFN row (2048 B)** | **0.579** | **0.840 G** | **7.93%** |
| 4 T10 rows (8 KB) | 0.842 | 1.222 G | 11.52% |
| **16 T10 rows (32 KB)** | **0.944** | **1.371 G** | **12.93%** |
| 64 T10 rows (128 KB) | 0.980 | 1.423 G | 13.42% |
| *dense stream (E30)* | *1.000* | *1.452 G* | *13.69%* |

**The crossover to 0.90 of the dense rate is 32 KB — sixteen consecutive T10 FFN rows.** The
sweep brackets it between 8 KB (0.842) and 32 KB (0.944); 32 KB is the smallest **sampled** point
above the line, and the true crossing lies inside that interval.

### 4.1 The design statement this probe exists to produce

> **Neuron-granular sparsity costs 42% of the budget. Activating in blocks of ≥16 consecutive
> rows costs 6%.** The difference between those two designs is **1.63×** on the only axis the
> goal still has.

Combined with E30's resident half, the specification of the 10 B goal on this machine is now
fully priced in one line:

> **≈ 32 M weights resident and free, plus ≤ 0.84 G streamed at row granularity — or ≤ 1.37 G if
> the architecture activates in ≥ 32 KB contiguous blocks. Per token, for 50 tok/s.**

### 4.2 Ordering only matters below 8 KB

`sorted` beats `random` by **1.34×** at 64 B and the two are within 5% from **8 KB upward**. So
sorting the fired experts by address is worth real money only for fine-grained designs — and a
fine-grained design is the one this probe just made expensive. **A naive unsorted gather at cache-line
granularity delivers 0.197 of the machine**, which is the worst number in this programme.

### 4.3 Post-hoc, explicitly NOT a gate

Dividing `sorted/contig` by the bookkeeping column bounds what an *ideal* gather — one with no
per-group loop overhead at all — could deliver: **0.606** at the T10 row, **0.510** at the S15
row, 0.297 at a cache line. Recorded under E14 §6 as a post-hoc quantity that may not be promoted
to a gate; the verdict stands at the measured 0.579. It moves no band.

## 5. Predictions, scored

Fixed in the brief at `8003fa3`, before the instrument was ever timed. **5 HIT, 1 PARTIAL.**

| # | prediction | reading | |
|---|---|---|---|
| 1 | `G-E31A` fires: `contig` within ±15% of 37.16 at every granularity and of itself | worst −6.5%, spread 11.2% | **HIT** *(on run 4; it failed three times first, §2)* |
| 2 | `G-E31B` fires: `sorted`(768) < `sorted`(2048) | 18.21 < 21.50 | **HIT** |
| 3 | verdict `GATHER-COSTS`, `r_T10` in 0.50–0.85 | **0.579** | **HIT** |
| 4 | crossover to 0.90 between 2 KB and 32 KB (1–16 T10 rows) | **32 KB**, bracketed in (8 KB, 32 KB] | **HIT, at the top edge** |
| 5 | `sorted` beats `random` ≥1.3× at 64 B **and** they converge within 5% at ≥128 KB | **1.34×**; converge at **8 KB** | **PARTIAL** |
| 6 | *(registered as independent of every band)* the sparse budget is `r × 1.45` and nothing here can raise it above 1.45 | 0.840 ≤ 1.452 | **HIT** |

**Prediction 5 is scored PARTIAL rather than HIT and the distinction is not pedantry.** Its
literal words are satisfied — at 128 KB the two orders are within 2.3% — but the number I meant
was *where* convergence happens, and I put it **16× too coarse**. Ordering stops mattering at
8 KB, not 128 KB. The reasoning behind the error was that group order would keep mattering until
groups were large; in fact once a group is four rows long the DRAM row buffer has already done
the work that sorting was doing.

## 6. What this decides

The brief wrote the disposition for `GATHER-COSTS` before the run and it applies:

> Every budget table in this programme that prices a sparse arm is optimistic by `1/r_T10`, and
> **the exporter's group size becomes a first-class design parameter.**

1. **E26 §9's owed item is promoted.** E26 measured the carve's locality cost end-to-end and
   concluded "a coarser carve, and it is a change to the EXPORTER, not the engine". E31 prices
   that change: at S15's 768-byte rows the carve runs at **0.469** of dense, and a 32 KB group
   would run at 0.944. **That is 2.01×, and it is the largest single lever left anywhere in this
   programme.** It is also free of quality cost by construction — the same weights, in a
   different order in the file.
2. **The 10 B target activation fraction is now a measured pair, not a guess.** E18 read "50 tok/s
   is a ~10%-activation budget" from the numerator side; E31 reads **7.93% at row granularity or
   12.93% in 32 KB blocks** from the memory side. The two roads to the same number met.
3. **Fine-grained MoE is the expensive design on this machine**, and by a factor that dwarfs the
   kernel work E30 just closed: `1.63×` between the two granularities, against the `1.24×` that
   was all the numerator had left.

## 7. What E31 could not claim, and what it leaves owed

**Could not claim** (brief §7, unchanged): nothing about quality — no BPB, no model, a noise
buffer; nothing about the engine's *actual* gather code, only what the machine will deliver to an
ideal one, which makes a bad reading a wall and a good reading only a permission; nothing about
other machines; nothing about the router, which is handed a free oracle here and was priced
end-to-end by E23; nothing about writes.

**Owed forward:**

1. **Build the coarse-group exporter and measure it end-to-end.** §6.1 says 2.01× at S15 and this
   is the first lever in a long time that is both large and free. E26 §9 item, now priced.
2. **`donor_engine.c`'s actual gather against this ceiling.** E31 measured an ideal gather; the
   engine's carve path has never been read as a fraction of it.
3. **The 1-in-8 selection density is a single point.** A sparser selection (1 in 32, 1 in 128)
   moves the DRAM row-buffer hit rate and may move the crossover. Untested.
4. **The crossover is bracketed, not located** — (8 KB, 32 KB]. A finer sweep between them would
   name the minimum activation unit to better than 4×.
