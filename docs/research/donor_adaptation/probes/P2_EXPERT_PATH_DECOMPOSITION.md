# P2 — Decomposing the expert path: how much of 2.86 µs/expert is memory?

**Outcome: MIXED** (pre-registered label, §3 of the brief, read verbatim — at **both** thread counts,
in **all five** repeats).
**Date: 2026-09-04. Machine: 3600X reference, generic `x86-64-v3` build, `OMP_PLACES=cores`,
`OMP_PROC_BIND=close`.**
**Brief: `briefs/BRIEF_P2_EXPERT_PATH_DECOMPOSITION.md` @ `e77b790` — pushed before the bench existed.**
**Raw: `benchmarks/donor_adaptation/speed/results/p2_expert_decomp.log`.
Apparatus: `benchmarks/phase60/engine.c`, `--expert-decomp` (new flag; no existing path modified).**

> **This report has not been audited.** Written by the figure that specified the brief and built the
> arm. A Controller pass is owed.

---

## 1. Why this was run

`SPEED_LEDGER.md` §4 left a **2.1× bracket** and flagged it as the widest thing in the ledger: on
Qwen2.5-1.5B with a 1.6-bit pack, dense throughput is **54.1 tok/s if the expert path is
bandwidth-bound** and **25.7 tok/s if it is not**. The ledger's optimistic row divides a *byte* rate
by fewer bytes per weight, which is only legitimate if time per expert is set by bytes moved.
`PHASE64_BUDGET.md` §1b(b) calls the path "gather/kernel-bound", but that sentence explains why the
path sits *below* the DRAM ceiling — **it never decomposed the 2.88 µs, and nobody had.**

## 2. The design, and the one thing that varied

Three arms over the **same kernel** (`matvec_lut_full`), same `M=384, Mpad=384, T=128`, same
`EB = 49152 B`, same LUT, same 500-token loop, same 48 touches/token. The **only** difference is
which expert each touch reads:

| arm | index per touch | footprint | isolates |
|---|---|---|---|
| **R** | i.i.d. random over 10,922 | 512 MB, ≫ L3 | the registered 64.1b measurement |
| **S** | sequential, stride 1 | 512 MB, ≫ L3 | prefetch-friendly edge of the memory term |
| **C** | **always expert 0** | **48 KB, stays in L1** | **the pure arithmetic floor** |

`memory_term = µs/expert(R) − µs/expert(C)`.

## 3. Gates and controls, all passed before the decomposition was read

**3.1 Kernel selftest green** — 73,024 checks, `worst |S − S_ref| = 0`, `exp256_ps` max rel err
1.191e-07. The new arm is reached only through `--expert-decomp`; no existing code path moved.

**3.2 Replication of the registered 64.1b number** (brief §3: void if >25% off):

| | 64.1b registered | P2 arm R |
|---|---|---|
| t1 | 7.45 GB/s | **7.65 / 7.96 GB/s** |
| t6 | 17.0 GB/s | **17.27 / 17.21 GB/s** |
| µs/expert t6 | 2.88 | **2.857** (median of 5: 2.857; range 2.811–3.233) |

**0.8% from the registered value.** PASS.

**3.3 Ordering control (brief §2.1): `C ≤ S ≤ R` held in every arm, every thread count, every
repeat.** Sequential is strictly easier than i.i.d. gather and strictly harder than staying in L1,
and the bench reproduces that ordering rather than assuming it.

**3.4 The planted-negative control (brief §2.1): arm C did NOT come back equal to R.** `c/r = 0.490`
to `0.632` across five repeats, far below the 0.95 INSTRUMENT-FAILURE threshold. Arm C's nominal
throughput (29–31 GB/s at t6, computed as if it had moved 48 KB per touch) is far below any plausible
L1 bandwidth, which is the positive sign that arm C is **compute**-limited rather than reading
anything. The instrument is isolating what it claims.

## 4. The result

Median of five t6 repeats; the full spread is given because it is not small.

| | t1 | **t6 (the ledger's operating point)** |
|---|---|---|
| **r** = R, i.i.d. random | 6.174 µs | **2.857 µs** |
| **s** = S, sequential | 5.849 µs | **2.690 µs** |
| **c** = C, arithmetic floor | 2.932 µs | **1.693 µs** |
| **memory_term = r − c** | 3.242 µs (52.5%) | **1.164 µs (40.7%)** |
| **c / r** | 0.475 | **0.593** |
| spread of `c/r` over 5 repeats | — | **0.490 – 0.632** |
| **pre-registered label** | **MIXED** | **MIXED** |

> **Roughly 40% of the expert path is memory and 60% is arithmetic.** Every one of the five repeats
> landed inside the MIXED band (0.30 < c/r < 0.70); neither the BANDWIDTH-BOUND nor the
> COMPUTE-BOUND edge was approached.

### 4.1 A cross-check nobody arranged

Divide the isolated memory term by the bytes it moves: `1.164 µs / 49152 B` = **42.2 GB/s**.

`PHASE64_BUDGET.md` §1 registers the machine's DRAM cold-stream **aggregate ceiling at 40–44 GB/s**,
measured separately, months earlier, by a different bench. **The decomposed memory term lands on an
independently measured constant.** Across the five repeats it ranges 29.8–46.8 GB/s, median 42.2 —
noisy, but centred on the right physical number. That is evidence the split is real and not an
artefact of the arm construction.

The arithmetic term, likewise: `98304 weights / 1.693 µs` = **58.1 G-weights/s** at t6
(range 54.4–62.1), the LUT kernel's throughput with memory removed.

## 5. What it does to the ledger — the optimistic row is WITHDRAWN

The expert path is now a two-term model, both terms measured here:

    t_FFN  =  W / 58.1e9   +   B / 42.2e9          (W = active weights, B = active bytes)

A denser pack shrinks `B` but not `W`. So going 4-bit → 1.6-bit (2.5× fewer bytes):

| | per-weight time | FFN of Qwen2.5-1.5B (1156.06 M weights) |
|---|---|---|
| 4-bit (B = 0.5 W) | 0.02906 ns | 33.60 ms |
| 1.6-bit (B = 0.2 W) | 0.02196 ns | 25.39 ms |

> **The denser pack buys 1.32× on the FFN, not 2.5×.** Across the repeats' spread, 1.28×–1.44×.

Substituting into `SPEED_LEDGER.md` §3, ctx 4096:

| packing | attn+head | FFN dense | KV | **dense tok/s** | FFN activation for 50 t/s |
|---|---|---|---|---|---|
| 4-bit | 5.24 ms | 33.60 ms | 2.80 ms | **24.0** | ≤ 35.6% |
| 1.6-bit | 2.09 ms | 25.39 ms | 2.80 ms | **33.0** | ≤ 59.5% |
| 1.6-bit, conservative* | 5.24 ms | 25.39 ms | 2.80 ms | **29.9** | ≤ 47.1% |

\* *conservative = the denser pack is assumed to buy nothing on attention and the head either. That
assumption is not tested here: P2 decomposed the **expert** path only. The proj path is priced at
37 GB/s and `PHASE64_BUDGET.md` §1 notes the fp32 projections "already run at aggregate BW when
threaded", which is evidence it is bandwidth-bound — but that was measured on **fp32** matrices, and
P2's whole result is that a kernel can have a large compute floor at small byte counts. **The same
decomposition on the proj path is now owed.***

> **`SPEED_LEDGER.md` §3's headline — "with the denser pack Qwen2.5-1.5B reaches 50 tok/s dense, with
> no carve and no quality damage" — is WITHDRAWN.** The measured figure is **30–33 tok/s**, and the
> 50 tok/s target is not reachable on this donor without sparsity after all.

**The bracket did collapse, which is what it was for: 25.7–54.1 tok/s (2.10×) becomes 29.9–33.0
(1.10×).** It collapsed toward the pessimistic edge, and it refuted the claim the figure that ran it
had written down an hour earlier — which is the entire reason the thresholds were pushed first.

## 6. What this changes for the programme

**6.1 The packing work is worth doing, but it is not the lever it looked like.** 1.32× on the FFN,
plausibly 2.5× on attention and the head *if* the proj path is bandwidth-bound (untested, §5). It
does not on its own reach the target on any donor in the ledger.

**6.2 The ~8.4 µs/expert integrated overhead is now unambiguously the largest expert-path lever.**
It is a *different* quantity from the one decomposed here — it is the gap between the integrated
4.2 GB/s and this kernel-pure 17.0 GB/s, i.e. it is **3.9× and it sits entirely outside the 2.86 µs
this report just split**. Nothing in P2 touches it and nothing in P2 makes it smaller.

**6.3 The activation target for 50 tok/s is much more generous than the one the carve probes were
measured at** — **35.6% at 4-bit, 47–60% at 1.6-bit**, against the **25%** D0/D0c carved at. That is
a real reopening: the quality probes have been paying for a sparsity level roughly 1.4×–2.4× more
aggressive than the speed target actually demands, and D0's own fidelity table shows the damage
falling steeply as the activation fraction rises. **What the carve costs at 40–60% activation has
never been measured.** It is the obvious next quality probe and it did not look worth running before
this number existed.

**6.4 The compute floor is 60% of the expert path, and no probe in this programme attacks compute.**
Every quality result here reduces *bytes* (fewer experts, fewer weights). At t6 the kernel spends
1.693 of 2.857 µs doing arithmetic that a denser pack, a better router and a finer carve all leave
untouched.

## 7. What this report does NOT claim

- It does **not** measure a 1.6-bit kernel — **none exists**. Arm C is the arithmetic of the **4-bit**
  kernel, and a 1.6-bit decode would do *more* work per byte, so its floor is **≥ c**. Per the brief's
  §3 extrapolation ban, this report puts an **upper bound** on what denser packing buys and
  **refutes** the ledger's optimistic row; it may not be read as confirming 1.32× either. 1.32× is
  the ceiling, not the prediction.
- It does **not** decompose the proj-GEMV path (attention, head). Owed — see §5's footnote.
- It does **not** touch the ~8.4 µs/expert integrated overhead (§6.2), BPB, or any donor.
- It does **not** run on a transformer. `engine.c` still has never executed one.

## 8. Reproduce it

```bash
clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp benchmarks/phase60/engine.c -o bin/engine_p2.exe -lm
OMP_PLACES=cores OMP_PROC_BIND=close ./bin/engine_p2.exe --kselftest      # must stay green
OMP_PLACES=cores OMP_PROC_BIND=close ./bin/engine_p2.exe --expert-rate    # replicates 64.1b
OMP_PLACES=cores OMP_PROC_BIND=close ./bin/engine_p2.exe --expert-decomp  # the three arms
```

Synthetic weights, no model, no gate, seconds of wall clock. `--expert-decomp` prints the
decomposition, both controls and the pre-registered label itself, so no reader has to recompute them.
**Run it more than once**: `c/r` moved between 0.490 and 0.632 across five repeats on an otherwise
busy machine, and a single run would have looked more precise than the measurement is.
