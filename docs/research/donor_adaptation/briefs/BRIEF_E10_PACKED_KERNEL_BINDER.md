# E10 — what binds the packed kernel at 22.5 GB/s?

**Pre-registered. To be pushed before any measurement is taken.** Opened by E8 §9 owed item 2,
the largest engine item left after E9.

---

## 1. The question, and why the obvious answer is not available

E8 removed the dependency chain and the packed weight path went 16.7 → 22.5 GB/s (24.0 after E9).
**It did not go to memory.** The same `matvec`, in the same run, on the same machine, delivered
**33.3 GB/s** on the fp32 arm — so the packed path is not sitting on a DRAM wall it shares with
fp32, and probe-3's post-L3 ~28 GB/s is not a wall either, since fp32 is above it.

E8 §9 named a suspect: **the unpack work**. The E9 build's emitted assembly, `.LBB15_11`, consumes
**16 bytes of packed weights per iteration** (32 weights, 0.5 B/weight) and spends on them:

| per iteration | count |
|---|---|
| `vmovq` (weight load, 8 B) | 2 |
| `vpshufb` xmm (nibble → byte) | 4 |
| `vpmovsxbd` ymm ← xmm | 4 |
| `vcvtdq2ps` ymm | 4 |
| `vfmadd231ps` ymm, mem | 4 |

Four independent accumulators `ymm0..ymm3`, one FMA each — E8's fix is intact and the chain is
**not** the binder any more.

**What this brief will NOT do:** convert that table into a cycle count. Doing so needs Zen 2 port
assignments and uop counts I have not read in a table I can cite, and this programme's own law is
that a number I have not read in its own source does not enter a decision. A hand-built port model
here would be exactly the kind of plausible artefact that costs this project its findings.

## 2. So the question is made empirical

**Which side of the memory system is the packed kernel's ceiling on?**

- **CORE-BOUND** — the unpack + FMA issue cost per byte is the binder. Then the ceiling barely
  moves when the weights are already in cache, and the remedy is a cheaper unpack (fewer uops per
  byte), not a cheaper stream.
- **STREAM-BOUND** — the binder is DRAM latency/prefetch at this access pattern. Then the ceiling
  rises sharply when the weights are L3-resident, and the remedy is prefetch/layout, not ALU work.

These predict **opposite curves**, so one measurement separates them.

## 3. The instrument

A standalone bench that `#include`s the engine's `matvec` **source, unmodified**, and calls it on a
synthetic weight matrix, sweeping the matrix footprint across the 16 MB L3 boundary probe-3 already
located, **with the row length `H` held fixed** so per-row and per-call cost is identical in every
cell. Reported as delivered GB/s of weights.

**Footprints:** 4 MB, 8 MB, 12 MB (L3-resident) · 24 MB, 48 MB (over) · 512 MB, 2 GB (DRAM,
matching the engine's real organs).

**Phase 61's law applies and is stated up front:** a microbench that is compute-bound does not
compose to an engine that is memory-bound. **E10's verdict is therefore scoped to the kernel's own
ceiling** — it may not be quoted as an engine rate, and any remedy it motivates has to come back
through an end-to-end `--bench` before a tok/s is published.

## 4. The planted control — the instrument must fire before its nulls count

The same sweep on the **fp32** kernel, same file, same harness. fp32 moves 8× the bytes per weight
and is the arm that already showed 33.3 GB/s in-engine.

**If the bench cannot show the fp32 arm rising when its weights become L3-resident, the bench
cannot see residency at all and every packed cell it produced is void.** The known-positive is the
L3 cliff probe-3 measured at ~100 GB/s in, ~28 GB/s out — a 3.5× step this instrument must
reproduce on *some* arm or it is not an instrument.

## 5. Gates and bands, fixed here, before anything is run

| gate | test |
|---|---|
| **G-K0** | **planted control**: fp32 arm, 4 MB cell ÷ 512 MB cell. **Must be ≥ 2.0.** FAIL ⇒ the whole run is void, no packed cell is read |
| **G-K1** | packed arm, 4 MB cell ÷ 512 MB cell — the discriminator |
| **G-K2** | the packed 512 MB cell must land within **±15%** of the engine's own measured 26.1 GB/s (`down`, §22.3) — if the bench does not reproduce the engine's kernel rate at the engine's footprint, it is measuring something else |
| **G-K3** | 3 repetitions per cell on an idle machine, dispersion reported. A cell whose spread exceeds the effect it is being asked to resolve **cannot decide** (E4's law) |

**Verdict bands for G-K1, fixed now:**

| packed 4 MB ÷ 512 MB | verdict |
|---|---|
| **≤ 1.15** | **CORE-BOUND** — residency buys nothing; the unpack is the ceiling |
| 1.15 – 1.6 | **MIXED** — both terms are live, and neither remedy alone is worth much |
| **≥ 1.6** | **STREAM-BOUND** — the unpack is not the ceiling; layout/prefetch is |

**No magnitude is predicted for G-K1.** E8 §3's rule permits a band derived from a measured
quantity over a structural factor; here I have no measured L3-resident packed rate to divide, and
the two previous first-principles predictions in this programme missed by 150× and by 1.3×.
**Naming the three verdicts in advance, with no favourite, is what this brief can honestly fix.**

## 6. What each verdict is worth

**CORE-BOUND** is the more useful outcome even though it is the more annoying one: it says the
remaining engine headroom is bought by making the unpack cheaper per byte — a different packing,
or a table that skips a conversion step — and it **retires the assumption, load-bearing since §12,
that the weight organs are bandwidth-bound.** §19's whole budget is built on that assumption, and
§19.5 already named it as the weakest link in the derivation.

**STREAM-BOUND** says the ~1.15–1.65× the engine has left is reachable by layout and prefetch work
on a kernel that is otherwise finished.

**MIXED** is a real possible answer and is not a failure to report.

## 7. Honest ceiling

Ledger §19.3 caps *all* remaining engine work at **1.15–1.65×** from here. E10 cannot beat that
cap; at best it says which of two doors the remaining factor is behind. **Coder-7B is 7.4× short of
50 tok/s and no outcome of this brief changes that** — the gap stays a property of the model.

---

## 8. VERDICT — appended after the run

**`CORE-BOUND`.** Full write-up: `probes/E10_PACKED_KERNEL_BINDER.md`. Ledger `§23`.

| gate | fixed above | measured |
|---|---|---|
| **G-K0** planted control | fp32 L3 ÷ DRAM **≥ 2.0** or everything is void | **PASS 4.40** — and it reproduced probe-3's 16 MB L3 cliff unprompted (161.7 → 104.5 → 45.1 GB/s) |
| **G-K1** discriminator | ≤1.15 CORE-BOUND / 1.15–1.6 MIXED / ≥1.6 STREAM | **1.022 → CORE-BOUND** |
| **G-K2** does the bench measure the engine's kernel | ±15% of 26.1 GB/s | **PASS 25.49, ratio 0.977** |
| **G-K3** dispersion | a spread wider than the effect cannot decide | **partly FAILED as written** — see below |

**G-K3 did not do what §5 assumed it would, and that is recorded rather than smoothed.** The 4 MB
cell's own spread is **30.6%**, wider than the 15% gap between 1.022 and the 1.15 boundary. **That
single ratio therefore cannot decide, by E4's law, and it is not what the verdict rests on.** What
decides is that **all seven packed cells sit within ±4% of their grand median** across a 512×
footprint range that moves the fp32 arm 4.4× — a between-arms contrast the gate table did not
anticipate needing. **The right gate would have been on the arm's flatness, not on one cell's
ratio.** Third time in this programme that a pre-registered rule was aimed at the wrong statistic
(cf. E8's discard rule, which discarded zero pairs).

**§5's refusal to predict a magnitude was correct and is not retro-fitted:** the brief fixed three
verdicts with no favourite, and the one that came out was the one that costs the project the most.

**What the brief did not anticipate at all:** the kernel is not FMA-bound either — 0.29 FMA per
cycle per core out of 2 — so `CORE-BOUND` names the side, not the mechanism. §6 said this verdict
would "retire the assumption that the weight organs are bandwidth-bound"; it does, and ledger
§19.4's 37 and 42 GB/s budget rows go with it. §7's honest ceiling holds: **Coder-7B is still 7.4×
short of 50 tok/s, and the engine's cover for that gap is now smaller, not larger.**
