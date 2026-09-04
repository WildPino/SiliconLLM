# BRIEF P2 — Is the expert path bandwidth-bound or compute-bound? (the 2.1× bracket)

**Status: PRE-REGISTERED. Written and pushed before the run, per standing practice.**
**Author: the Adapter / Principal. Date: 2026-09-04.**
**Depends on: `SPEED_LEDGER.md` §4; `docs/PHASE64_BUDGET.md` §1b(b).**

---

## 1. The question

`SPEED_LEDGER.md` §4 leaves one bracket open, and it is the widest thing in the ledger:

> On Qwen2.5-1.5B with the 1.6-bit pack, dense throughput is **54.1 tok/s if the expert path is
> bandwidth-bound** and **25.7 tok/s if it is not.** A factor of **2.1×**, and it decides whether the
> 50 tok/s target is reachable with no quality damage at all.

The ledger's optimistic row divides a **byte** rate (17.0 GB/s) by fewer bytes per weight. That is
only legitimate if the time per expert is set by the **bytes moved**. `PHASE64_BUDGET.md` §1b(b) says
in terms that it is not — it calls the expert path "gather/kernel-bound, not stream-bound" at
**2.88 µs/expert** — but that sentence was written to explain why the path is *below* the DRAM
ceiling, not to decompose the 2.88 µs. **The decomposition has never been measured.**

> **The question this brief pre-registers:**
> **Of the measured 2.88 µs/expert, how much is memory traffic and how much is arithmetic?**
> If the arithmetic floor is most of it, a denser pack buys nothing on the FFN and every
> "1.6-bit" figure in the ledger collapses toward its pessimistic edge.

## 2. The design — one arm removes memory, nothing else changes

The existing `--expert-rate` bench walks a 512 MB pool (10,922 × 48 KB experts) with an i.i.d.
gather, 48 touches per token. The new arm changes **exactly one thing**: which expert is touched.

| arm | expert index per touch | pool footprint touched | what it isolates |
|---|---|---|---|
| **R** | i.i.d. random over 10,922 | 512 MB, ≫ L3 | the registered 64.1b measurement — **replication** |
| **S** | sequential, stride 1 | 512 MB, ≫ L3 | prefetch-friendly upper edge of the memory term |
| **C** | **constant — always expert 0** | **48 KB, L1/L2-resident** | **the pure arithmetic floor** |

**Arm C is the whole point.** Identical kernel, identical `M`, `Mpad`, `T`, identical LUT, identical
number of calls, identical output buffer — the *only* difference is that the 48 KB it reads is the
same 48 KB every time and therefore never leaves cache. Its µs/expert is compute with the memory
term removed. Then:

    memory_term  =  µs/expert(R)  −  µs/expert(C)

and the fraction of the expert path that a denser pack can possibly address is
`memory_term / µs/expert(R)`.

### 2.1 The control that makes arm C trustworthy

Arm C is a **planted negative for the memory term**: it is constructed so that the memory traffic is
known to be ~zero. If arm C returns a µs/expert *equal* to arm R, one of two things is true — either
the path is entirely compute-bound, or **the bench is not measuring what it claims** (e.g. the
compiler hoisted the load, or the pool access was never the cost). To tell those apart, arm C must
report a GB/s figure computed the same way as R's; if C's nominal "GB/s" exceeds the machine's L1
bandwidth by an implausible margin the arm is working, and if C is indistinguishable from R the
result is reported as **INSTRUMENT-FAILURE**, not as a finding.

Second control: **arm S must land between C and R.** Sequential access is strictly easier for the
memory system than i.i.d. gather and strictly harder than staying in L1. If `S < C` or `S > R`
outside noise, the ordering is violated and the run is void.

### 2.2 What is held fixed

Same binary, same build policy as 64.1b (generic `x86-64-v3`, `-ffp-contract=on`, no `-march`), same
`OMP_PLACES=cores OMP_PROC_BIND=close`, threads {1, 6}, same 500-token loop, same
`matvec_lut_full` kernel, same `M=384, Mpad=384, T=128, EB=49152`. **The existing `--expert-rate`
code path is not modified**; the new arms are reached only through a new flag, and the kernel
selftest must stay green to prove it.

## 3. Pre-registered decision rule — fixed before any result

Let `c = µs/expert(C)` and `r = µs/expert(R)`, both at **t6** (the operating point the ledger uses).
Reference: 64.1b registered `r = 2.88 µs`.

| outcome | condition | what it means for the ledger |
|---|---|---|
| **BANDWIDTH-BOUND** | `c ≤ 0.30 · r` | ≥70% of the expert path is memory. **The ledger's optimistic row stands**; the 1.6-bit pack is worth up to 2.5× on the FFN and becomes a priority build |
| **MIXED** | `0.30 · r < c < 0.70 · r` | report the split verbatim. The denser pack is worth **at most** `(r − c)/r × 2.5×` on the FFN; the ledger must be restated with that ceiling substituted for 2.5× |
| **COMPUTE-BOUND** | `c ≥ 0.70 · r` | ≤30% is memory. **The ledger's pessimistic edge is the truth**: denser packing does ~nothing for the FFN, its value is confined to attention, head and DRAM footprint, and the ~8.4 µs/expert overhead becomes the only expert-path lever |
| **INSTRUMENT-FAILURE** | `c ≥ 0.95 · r`, or the §2.1 ordering control fails | the bench is not isolating what it claims. Report as such. **Do not report a decomposition.** |
| **REPLICATION-FAIL** | arm R's `r` differs from 64.1b's 2.88 µs by >25% | the harness has drifted. Fix that first; no decomposition is readable until R reproduces |

**The 0.30/0.70 thresholds are set here, before the run.** They are coarse on purpose: the bracket
they arbitrate is 2.1× wide, so only a decomposition that is lopsided should be allowed to collapse
it, and anything in between must be reported as the mixed number it is rather than rounded to the
nearer story.

**No extrapolation to a pack this bench does not run.** Arm C measures the arithmetic of the
**4-bit** kernel. A 1.6-bit kernel would do *more* arithmetic per byte (denser decode), so its
compute floor is ≥ `c`. Therefore this brief can establish an **upper bound** on what denser packing
buys and can **refute** the optimistic row; it cannot **confirm** a 2.5× gain, and no result here may
be written as if it had. Confirming that requires a 1.6-bit kernel, which does not exist.

## 4. What this brief does NOT claim and does NOT test

- It does **not** measure engine.c on a donor, or any transformer. Nothing here touches attention,
  the head, or the KV cache — the three terms `SPEED_LEDGER.md` §2 says have never been probed.
- It does **not** test the ~8.4 µs/expert integrated overhead. That is the gap between 4.2 and
  17.0 GB/s and it is a *different* quantity from the one decomposed here.
- It does **not** build or evaluate a 1.6-bit pack. See the extrapolation ban in §3.
- It does **not** revisit any BPB result. This is a speed measurement end to end.

## 5. Cost

One build, two flags, six numbers (three arms × two thread counts), a few seconds of wall clock on
CPU. **No training, no GPU, no model weights.** It is the cheapest decision-relevant measurement
available and it gates whether the packing work is worth starting.

## 6. Reporting

`probes/P2_EXPERT_PATH_DECOMPOSITION.md`. Report: `c`, `s`, `r` at both thread counts with the GB/s
each, the §2.1 ordering control, the replication of 64.1b's 2.88 µs, the outcome label from §3
verbatim, and the restatement of `SPEED_LEDGER.md` §4 in its light — including, if the outcome is
COMPUTE-BOUND, an explicit note that the ledger's headline "54.1 tok/s dense at 1.6-bit" is
**withdrawn** and replaced by its pessimistic edge.
