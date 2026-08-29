# Brief P2 — the weight-once block kernel spends 30× more traffic on its LUTs than on its weights

**Author:** the Adapter / Principal · **Date:** 2026-08-29 · **Status:** pre-registered, **NOT yet dispatched — queued behind P1.**

---

## 0. Scope, stated first, because it limits this brief severely

> **This brief does NOT touch the single-stream decode path, and therefore does NOT touch the 20 tok/s
> headline the donor sizing rests on.**

Decode calls `matvec_lut_full` (`benchmarks/phase60/engine.c:111`, via `:260`, `:263`, `:271`) with a single
LUT and `__m256i acc[4]` — four accumulators, register-resident, no spill. **That path is healthy and this
brief has nothing to say about it.**

What follows concerns `matvec_lut_full_K` (`engine.c:146`, via `forward_block_lm` at `:684`, `:685`, `:688`)
— the **weight-once layer-major block forward**, used for prefill and batched positions. That path matters
for time-to-first-token at long context, which the donor plan cares about at 128K, and for any batched
serving. **It is a throughput question, not a latency question. Priority is below P1 accordingly.**

**It is also NOT what D5 measured.** `gemv_donor_bench.c`'s `matvec_lut_full` is the plain `Kn=1` kernel with
register accumulators. **So nothing here explains D5's 39.7%-of-ceiling anomaly on `gate/up`, and this brief
must not be cited as if it did.** They are different kernels.

---

## 1. The observation, read from our own source

```c
// engine.c:146
static void matvec_lut_full_K(const int8_t* codes,const int8_t* luts,int lstride,
                              int32_t* Y,int ys,int M,int Mpad,int T,int Kn){
    for(int base=0;base<M;base+=32){
        __m256i acc[62][4];                                   // <-- 62 x 4 x 32 B = 7936 B
        ...
        for(int t=0;t<T;t++){
            __m256i idx=_mm256_loadu_si256(... codes + t*Mpad + base ...);   // 32 B, amortised over Kn
            for(int k=0;k<Kn;k++){
                __m256i tbl=_mm256_broadcastsi128_si256(... luts + k*lstride + t*16 ...);
                acc_add_i8x32(&acc[k][0], _mm256_shuffle_epi8(tbl,idx));
            } } ... } }
```

`forward_block_lm` caps `Kn` at **62** (`engine.c:642`). AVX2 has **16** `ymm` registers.

### 1.1 Traffic accounting — the arithmetic to be checked, not believed

Take a donor-width MLP: `D = 8192`, `d_ffn = 28672`, so `T = TUP = D/2 = 4096`, `M = 28672`, `lstride = T*16 = 65536`.

| quantity | expression | at `Kn = 62` |
|---|---|---|
| `codes` read (whole array, once) | `M · T` | **117 MB** |
| LUT working set (all `k`) | `Kn · T · 16` | **4.06 MB** — *exceeds L2 (512 KB/core, Zen2)* |
| LUT bytes read | `(M/32) · T · Kn · 16` | **3.64 GB** |
| **LUT : codes ratio** | `Kn / 2` | **31 ×** |
| accumulator bytes touched | `(M/32) · T · Kn · (4 loads + 4 stores) · 32 B` | **~465 GB of L1 RMW** |
| weight bytes per `(t,k)` unit | `32 / Kn` | **0.52 B** |

> **Per `(t,k)` inner iteration the kernel does one 16 B LUT load, one `pshufb`, and — because `acc[62][4]`
> cannot live in 16 registers — four 32 B accumulator loads and four 32 B accumulator stores, to move
> 0.52 bytes of weight.**
>
> Zen2 retires 2 loads and 1 store per cycle. **Eight memory ops per `(t,k)` puts a floor of ~4 cycles on an
> iteration whose useful work is one shuffle and four adds.** If this arithmetic holds, the block kernel is
> **port-bound on its own bookkeeping**, and the weight stream — the thing all our packing work optimises —
> is not the binding constraint on this path at all.

### 1.2 Why this is consistent with something already on the record

The block-decode work was built and verified **lossless**, weights-once was verified, and **its speed claim
was scoped out at 8.3M** — it was never demonstrated at scale. `Kn/2` is a *ratio*, so at the 8.3M geometry
the same defect exists but the absolute working sets are small enough to sit in cache. **A defect that
hides at 8.3M and dominates at donor width is exactly the class this programme keeps finding**, and it is
the same shape as the `h=128`-is-not-scale-free error: a quantity that looked benign at the width where it
was validated.

---

## 2. Why there is no obvious fix — this is a design question, not a patch

The naive reorderings each fail, and the brief states this so the Builder does not waste a day rediscovering it:

| reordering | LUT traffic | `codes` traffic | verdict |
|---|---|---|---|
| **current** (`base` outer, `k` inner) | 3.64 GB (L3) | 117 MB (DRAM), read once | LUT-dominated |
| `k` outer, `base` inner | unchanged | `Kn ×` 117 MB = **7.2 GB DRAM** | **worse** |
| tile `k` into groups of 4 (LUT set → 256 KB, fits L2) | L2-resident | `⌈62/4⌉ ×` 117 MB = **1.9 GB DRAM** | trades L3 for DRAM |
| tile `t` into slabs (`Tb=256` → LUT set 254 KB) | L2-resident | slab-resident | but accumulators must persist across slabs: `M · Kn · 4 B` = **7.1 MB**, spilling to L3 |

**There may be a two-dimensional blocking that wins, and there may not be.** That is the question, and it is
a real one. **A negative — "no blocking of this kernel beats the current one at donor width" — is a
perfectly good deliverable** and would tell us the weight-once block path does not scale, which is itself
load-bearing for the donor plan's prefill story.

---

## 3. Stage 1 — the counting audit, machine-agnostic, run first

**Before any kernel is written or any time is measured**, settle whether §1.1 is even true:

1. **Confirm or refute the register spill.** Compile `matvec_lut_full_K` at the project's real flags and
   **read the generated assembly.** Does `acc[62][4]` spill for `Kn > 3`? At what `Kn` does the compiler stop
   keeping accumulators in registers? **Report the `Kn` threshold as measured from the disassembly, not
   assumed.**
2. **Confirm or refute the traffic table.** Re-derive every row of §1.1 from the source. **If I have made an
   arithmetic error, that is the finding and the brief dies here** — say so plainly and stop.
3. **Establish the `Kn` actually used in the donor plan's prefill**, from the code path, not from this brief.
   If `Kn` is small in practice, `Kn/2` is small and **P2 is not worth building.** Report that outcome as
   readily as the other.
4. **Hardware counters if available** (`L1`/`L2`/`L3` miss counts, load/store port utilisation) at the 8.3M
   geometry, which does not require a clear machine to be *ratio*-informative even if absolute times are not.

**STOP after Stage 1 with a verdict on whether Stage 2 is justified.** Do not write a new kernel until the
counting says there is something to fix.

## 4. Stage 2 — only if Stage 1 confirms, and only on a clear machine

A blocked variant, alongside the existing one (**never replacing it**), with the same absolute controls P1 carries:

| # | control | must |
|---|---|---|
| **C1 BIT-EXACT** | blocked kernel vs current `_K` kernel vs scalar reference, several `Kn` including 1, 2, 3, 4, 62 | **identical `int32`** |
| **C2 PLANTED** | corrupt one accumulator lane; re-run C1 | **C1 must FAIL** |
| **C3 END-TO-END** | real engine, real weights, real prompt, block path | **identical tokens and BPB** |
| **C4 TRAFFIC ACHIEVED** | measured bytes/misses per level, both arms, **one stated convention applied to both** | reported, not derived |

---

## 5. What this brief does not claim

- **No speedup is claimed.** §1.1 is a traffic count; whether traffic is the binding constraint on this
  kernel is precisely what Stage 1 must establish. Phase-61's standing law cuts both ways: *a compute-bound
  microbenchmark does not compose to a memory-bound engine*, and a traffic count does not compose to a
  speedup.
- **No relation to D5 §12.6 is claimed** (see §0). Different kernel, different path.
- **No effect on decode latency is claimed** (see §0). Prefill and batch only.
- **No quality effect exists.** Any change here is bit-exact or it is rejected.

## 6. Standing rules

> Report **ACHIEVED**, printed from the objects — here, from the **disassembly** and the **counters**, not
> from the C source's apparent intent.
> An instrument must **fire on a known positive** before its passes count.
> A result that flatters the hypothesis earns more scrutiny — **and §1.1 flatters me, because a 31× ratio is
> the kind of number that makes an author stop checking.** Check it first.

---

## AMENDMENT 1 — 2026-08-29, the Adapter / Principal

**Appended, not edited in place.** Written in response to a direct observation in
`audits/CONTROLLER_D5_126_AUDIT.md`, which flagged that this brief *"correctly disclaims explaining the
39.7% anomaly, but is written as though there were an anomaly still standing."* **That is a fair reading of
§0 and it is corrected here.**

### A1.1 There is no anomaly to disclaim

§0 said this brief "must not be cited as if it" explained D5's 39.7%-of-ceiling result. That phrasing
**presupposed the result existed.** It does not.

The audit's **A9 BLOCK**: the `39.7% / 76%` pair divided a **charged-byte** numerator by a **moved-byte**
denominator, with the numerator's bias differing ~2× between rows. Corrected, the two sit at **85.8%
(`k/v`)** and **~79.5% (`gate/up`)**. **The wide-shape anomaly does not exist.** It is a units error, so no
re-run can restore it.

> **Everywhere §0 says "does not explain the anomaly", read "there is no anomaly."** The distinction
> matters: the first invites a reader to go looking for the explanation elsewhere; the second closes the
> question.

### A1.2 What the audit leaves standing, correctly sized

The half-line waste that got double-counted into that 46-point gap is **itself real**. The audit prices its
fix at **~2× on the charged rate at past-L3 shapes and nothing at `k/v`-class shapes**, bounded by the
moved-byte plateau rather than by the withdrawn `39.7 → 85.9` span.

**That is a smaller, better-bounded, and still worthwhile target** — and it is a *different* target from
this brief's. Do not merge them:

| | this brief (P2) | the half-line waste |
|---|---|---|
| kernel | `matvec_lut_full_K` (block/prefill) | the shape measured by D5 |
| quantity | LUT re-reads and spilled accumulators | moved bytes per charged byte |
| size | `Kn/2` ratio, to be established in Stage 1 | ~2× charged rate, past-L3 shapes only |

### A1.3 What does NOT change in this brief

**§1.1's traffic arithmetic is independent of everything the audit touched.** It is a static count read off
`engine.c:146–155`: `acc[62][4]` against 16 `ymm` registers, and a LUT-to-codes traffic ratio of `Kn/2`.
That count stands or falls on its own, which is exactly why **§3 Stage 1 requires it to be re-derived and
the disassembly read before a single line of new kernel is written.** If the count is wrong, the brief dies
there — as it was written to.

**§0's scope limits also stand, and are now more important, not less:** this brief does not touch
single-stream decode, does not touch the 20 tok/s figure, and concerns a **different kernel** from the one
D5 measured. Nothing in the audit changes any of that.

### A1.4 Priority, restated

This brief was already queued behind P1. **It stays there**, and the audit gives a third item that outranks
it: a **stride-conflict probe**. Every `Mpad` in D5's sweep is a power of two or a multiple of 4096, so
set-conflict eviction is completely unseparated from capacity. Padding each stride by 64 bytes is a shape
change rather than a kernel change, costs an allocation offset, and can *separate* two mechanisms that every
measurement so far has only ever reported the sum of. **It has been added as an arm to P1's Stage 2**
(see `BRIEF_P1_NIBBLE_PACKING.md`, Amendment 1 §A1.5) rather than given its own brief.
