# E10 — the packed kernel is CORE-BOUND, and the engine is already sitting on its ceiling

**Verdict: `CORE-BOUND`.** Pre-registered and pushed as
`briefs/BRIEF_E10_PACKED_KERNEL_BINDER.md` (`a0366d0`) before the bench existed.

Closes E8 §9 owed item 2. **It also settles, by measurement, the assumption ledger §19.5 named as
the weakest link in its own derivation** — and the assumption fails.

---

## 1. The headline

The packed `matvec` delivers **49–53 G-weights/s no matter where the weights live.** Moving its
working set from 4 MB (L3-resident) to 2 GB (DRAM) — a **512× change in footprint** — moves its
delivered rate by **less than 8%**, while the same bench, same file, same run, moves the fp32 arm
by **4.4×**.

**The engine's own FFN organs are already at that ceiling.** From §22.3's profile on Coder-7B:

| | G-weights/s |
|---|---|
| engine `gate+up` (73.597 ms, 3.802 G weights) | **51.66** |
| engine `down` (36.403 ms, 1.901 G weights) | **52.22** |
| **bench, every footprint from 4 MB to 2 GB** | **49.11 – 52.95** |

There is **no streaming headroom left in this kernel.** Not 1.6×, not 1.2× — none.

## 2. The sweep

Same `matvec` **source**, unmodified: `kbench.c` `#include`s `donor_engine.c` with its `main`
renamed, so the kernel under test is byte-for-byte the one the engine runs. Row length `n_in` held
**fixed at 3584** in every cell, so per-row and per-call cost is identical; only the number of rows
moves. `--mvacc 4`, 6 threads, 7 repetitions per cell, ~8 GB of weight traffic per repetition.

| footprint | packed GB/s | packed G-w/s | spread | fp32 GB/s | fp32 G-w/s | spread |
|---|---|---|---|---|---|---|
| 4 MB | 26.04 | 52.07 | 30.6% | **163.45** | 40.86 | 22.5% |
| 8 MB | 25.75 | 51.51 | 7.5% | 157.12 | 39.28 | 24.2% |
| 12 MB | 25.20 | 50.40 | 6.8% | 161.72 | 40.43 | 5.7% |
| 24 MB | 24.70 | 49.40 | 17.5% | 104.48 | 26.12 | 37.0% |
| 48 MB | 24.55 | 49.11 | 0.8% | 45.10 | 11.28 | 28.4% |
| 512 MB | 25.49 | 50.97 | 4.0% | 37.12 | 9.28 | 10.4% |
| 2048 MB | 26.48 | 52.95 | 17.2% | 38.58 | 9.65 | 2.2% |

`D:/_ktmp/e8/e10_kbench7.log`; the 3-rep sweep that preceded it is `e10_kbench.log` and agrees.

## 3. The gates, in the order the brief fixed

**G-K0 — the planted control, and it is read FIRST.** fp32 4 MB ÷ 512 MB = **4.40**, band was
**≥2.0**. **PASS.** The bench can see residency. It also reproduces, unprompted, the thing
probe-3 measured independently: the **L3 cliff at 16 MB** — 161.72 → 104.48 → 45.10 GB/s across
12 / 24 / 48 MB. Nobody told this bench where the cliff was.

**G-K1 — the discriminator.** packed 4 MB ÷ 512 MB = **1.022**. Bands: ≤1.15 CORE-BOUND,
1.15–1.6 MIXED, ≥1.6 STREAM-BOUND. → **CORE-BOUND**.

**G-K2 — is the bench measuring the engine's kernel at all?** packed 512 MB cell = 25.49 GB/s
against the engine's own measured **26.1 GB/s** (`down`, §22.3). Ratio **0.977**, band ±15%.
**PASS** — and this is also what rules out a hoisted or elided loop, which is why it was
pre-registered.

**G-K3 — dispersion.** Honestly: **the 4 MB cell's own spread is 30.6%, which exceeds the 15%
distance from 1.022 to the 1.15 boundary, so that single ratio cannot decide by itself** (E4's
law). What decides is the flatness of the whole arm: **all seven packed cells lie within ±4% of
their grand median 25.49 GB/s** across a 512× footprint range that moves fp32 by 4.4×. The verdict
rests on the contrast between arms and on seven cells, not on one ratio.

**A correction that goes against the verdict, stated rather than buried.** Per-call cost — the
serial `g_xe`/`g_xo` deinterleave of 1792 elements plus one OpenMP region at §12.4's 2.7 µs — is
amortised over 2,340 rows in the 4 MB cell and 299,593 rows in the 512 MB cell, so it penalises the
*resident* cells by roughly 3%. Correcting for it moves G-K1 from 1.022 to about **1.05** —
**still inside CORE-BOUND**, and the direction is named because it is the direction that would
have hurt.

## 4. What this does to ledger §19 — the assumption §19.5 flagged has now failed

§19 priced the entire remaining engine budget by assuming the weight organs are **bandwidth-bound**,
and then asked what they would deliver at streaming bandwidths this machine has been measured
providing: **28 GB/s** (probe-3 post-L3), **37.0** (proj-GEMV floor), **42** (DRAM aggregate).
§19.5 wrote that the assumption was the weakest link in the derivation.

**It is false for the packed kernel.** The kernel stops at ~25.5 GB/s with the data *in L3*, so
37.0 and 42 GB/s are not reachable by feeding it faster — they are not its constraint. The §19.4
budget rows that assumed them are withdrawn for this path.

| what §19.4 offered the packed path | status after E10 |
|---|---|
| 37.0 GB/s → 1094 M active weights at 50 tok/s | **not available** — not a bandwidth problem |
| 42 GB/s → 1242 M | **not available** |
| **~25.5 GB/s / ~53 G-w/s** | **this is the kernel's ceiling, and the engine is at 51.7–52.2** |

**The engine budget shrinks.** §19.3 said engine work had 1.7–2.5× in it; E8 and E9 took 1.52× and
the arithmetic left 1.15–1.65×. E10 says **the weight path's share of that remainder is ~1.03×**,
because it is already at the kernel's own ceiling. Whatever is left is in the *other* organs, or it
requires **replacing the kernel**, not feeding it.

**This makes §19.3's strategic conclusion stronger, not weaker.** The missing 7.4× was already a
property of the model. It now has less engine cover than it had this morning.

## 5. Two observations this bench returned unasked-for

**5.1 — Per weight, the format's byte advantage is already fully spent.** Cache-resident, fp32
delivers **40.9 G-weights/s** and packed **52.1** — packed is only **1.27×** faster per weight
while reading **8× fewer bytes**. The 8× only pays while bytes are the constraint, and for this
kernel they are not. Out of cache the picture inverts and the format earns its keep (52.1 vs 9.3,
**5.6×**) — which is precisely why the packed format is right for the engine and why its *kernel*
is nonetheless the thing now standing in the way.

**5.2 — It is not FMA-bound either, and that is the next question.** At 52 G-weights/s, 8 lanes per
FMA, 6 cores, 3793 MHz, the loop issues **0.29 FMA per cycle per core** against a Zen 2 capability
of 2 — about **14%**. Arithmetic on measured rates and the published clock; no port table is
involved and none is asserted. So "core-bound" here does **not** mean "saturating the FP units",
and E10 deliberately does not offer a mechanism for the remaining 7×. **That is E11.**

One structural fact worth carrying into it, read off the emitted assembly rather than modelled: each
`vpshufb` in `.LBB15_11` produces **16 bytes of which the following `vpmovsxbd` consumes 8**. Half
the shuffle output is discarded, four times per iteration.

## 6. Owed

1. **E11 — what makes the per-weight ceiling 53 G-w/s?** The half-wasted shuffle (§5.2) is a
   candidate with a cheap test. The `--lut` path (`matvec_lut`, probe-1's int8-accumulate kernel)
   is a second, and it is **already in this engine and unmeasured at donor scale** — it may have a
   different per-weight ceiling entirely.
2. **The same sweep at `n_in` = 18944**, the FFN `down` shape. E10 held `n_in` fixed at 3584 by
   design; the engine's `down` organ lands at 52.22 G-w/s anyway, which is evidence the ceiling is
   row-length-independent, but it is one point, not a sweep.
3. **The non-packed ternary path still has a single accumulator** — E8 §9 item 3, still open.
4. **A vectorised `expf`** — E9 §5, still open.
