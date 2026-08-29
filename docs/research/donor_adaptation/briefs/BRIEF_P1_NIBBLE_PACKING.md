# Brief P1 — the upper nibble is dead: 2 bits/weight with the pshufb path intact

**Author:** the Adapter / Principal · **Date:** 2026-08-29 · **Status:** pre-registered, dispatched Stage 1.
**Provenance:** the Owner's instruction of 2026-08-28 — *"c'è ancora qualcosa da attaccare che non abbiamo
attaccato ... la dobbiamo scrivere noi la letteratura in materia"* — read against D5 §12.6, which reversed
§12.5 and found the donor-width path **memory-bound**.

---

## 0. The observation, read from our own source

The engine packs two trits into one `int8_t`:

```
benchmarks/phase60/engine.c:161        (production engine, bc_tm)
benchmarks/phase60/engine.c:164        (production engine, bc_rm)
benchmarks/donor_adaptation/gemv_donor_bench.c:147   (donor bench, bc_tm)
    codes[...] = (int8_t)((w0+1)*3+(w1+1));
```

That value lies in **`[0,8]`**. The kernel consumes it as:

```
benchmarks/phase60/engine.c:150-152
    __m256i idx = _mm256_loadu_si256(... codes + t*Mpad + base ...);   // one byte per 2 trits
    __m256i tbl = _mm256_broadcastsi128_si256(... luts + k*lstride + t*16 ...);
    acc_add_i8x32(&acc[k][0], _mm256_shuffle_epi8(tbl, idx));
```

and the table is built **16 entries wide, with 9..15 zero-filled**:

```
benchmarks/phase60/engine.c:158-159 (build_lut_t3)
    for(int c=0;c<16;c++){ int s=0; if(c<9){ int w0=c/3-1,w1=c%3-1; s=w0*x0+w1*x1; } lut[t*16+c]=(int8_t)s; }
```

> **`_mm256_shuffle_epi8` selects from the 16-byte table using bits `[3:0]` of each index byte** (and zeroes
> the lane if bit 7 is set). Our codes are `0..8`, so **bits 4–7 of every byte of `codes` are structurally
> zero and the instruction already ignores them.**
>
> **We are streaming a full byte to carry an index the hardware reads as a nibble.**

**4 bits/weight today. 2 bits/weight is reachable. That is exactly 2× fewer bytes on the weight stream.**

### 0.1 Why this was not visible before

D5 §12.5 asked whether **5-trit packing** (1.6 bits/weight, 2.5×) pays, concluded "compute-bound, no", and
§12.6 reversed the conclusion. **Both rounds framed the question as base-3 density, which needs a
divide-by-3 decoder and cannot use `pshufb` at all** — so the cost side always looked heavy. The nibble is a
different axis: it changes **the container, not the alphabet**. The alphabet stays base-3, 2 trits per code,
the LUT stays 16 entries, the decode stays one `pshufb`.

**Nibble packing captures 80% of 5-trit's byte reduction (2.0× of 2.5×) while keeping the entire decode path
we already have, bit-for-bit.**

### 0.2 This is our constraint, not the donor's

Consistent with the Owner's reframe: `0.5 bytes/weight` is **our packing choice**, not a property of any
donor. So is the 48 KB block granularity, so is the gate-at-full-width floor. This brief attacks one of ours.

---

## 1. The proposed layout and kernel — hypothesis, not instruction

The Builder owns the implementation and may reject this shape for a better one; what follows is the
existence argument, so that the brief is falsifiable rather than aspirational.

**Layout.** Pack the code for `t` in the low nibble and the code for `t+1` in the high nibble of one byte,
keeping `m` contiguous exactly as today:

```
    codes2[(size_t)(t/2)*Mpad + m]  =  code(t, m) | (code(t+1, m) << 4)
```

**Inner loop, per byte** (against today's per-byte cost of 1 broadcast + 1 pshufb + 1 accumulate for 2 trits):

| step | note |
|---|---|
| 1 × load | **half as many loads — this is the whole point** |
| `lo = b & 0x0F` | 1 × `vpand` |
| `hi = (b >> 4) & 0x0F` | AVX2 has no byte shift: `_mm256_srli_epi16` + `vpand` |
| 2 × broadcast (`t`, `t+1`) | **same count per trit as today** |
| 2 × pshufb | **same count per trit as today** |
| 2 × accumulate | **same count per trit as today** |

> **Net per 4 trits: one shift and two masks added; one 32-byte load removed.**
> Per-weight `pshufb` count is **unchanged**. This is why the idea is cheap and why 5-trit is not.

**Not proposed, and why:** three trits do not fit a nibble (27 > 16), so **2 trits/nibble is the `pshufb`
optimum** — there is no denser packing that keeps a single-instruction decode. This brief therefore claims
the *last* free doubling on this axis, not a step along a ladder.

**Note on the production kernel's shape.** `phase60/engine.c:150` already streams `codes` **once** and reuses
the same `idx` across `Kn` LUTs. Halving the bytes halves that shared stream, so the benefit does not depend
on `Kn` — but **state the `Kn` used**, because the ratio of bytes to arithmetic does.

---

## 2. The trap the Builder must not walk into

**Code `0` does NOT mean "zero weights".** `c=0 → w0 = 0/3−1 = −1, w1 = 0%3−1 = −1`. The neutral code is
**4** (`(0+1)*3+(0+1)`). Today `bc_tm` writes `0` into the padding rows `m ∈ [M, Mpad)` and gets away with it
only because the store guard `for(int r=0;r<32 && base+r<M;r++)` discards those lanes.

> **Nibble packing changes the `t` blocking, therefore changes the padding geometry.** The argument that
> padding is harmless must be **re-derived for the new layout and stated**, not inherited. If in doubt, write
> `4` (the true neutral) and say so — but do not silently assume `0` remains safe.

**Second trap: odd `T`.** `T = K/2`; if `T` is odd the final byte carries only a low nibble. State what goes
in the high nibble and why it cannot contaminate the accumulator.

---

## 3. Pre-registered controls — all four, before any timing number exists

Registered now, before the measurement, per standing rule.

| # | control | must | why it is here |
|---|---|---|---|
| **C1 BIT-EXACT** | nibble kernel vs `ref_t3` scalar, and vs today's byte kernel, same weights, same `x` | **identical integers, not close** | the primary correctness gate |
| **C2 PLANTED** | corrupt one nibble (swap hi/lo at a single `t`) and re-run C1 | **C1 must FAIL** | a check that cannot fail proves nothing. Standing law: an instrument must FIRE on a known positive before its passes count |
| **C3 END-TO-END** | run the real engine, real weights, real prompt: token stream and BPB vs today | **identical** | Phase-60 law, in force: *kernel-bit-exact does NOT compose to system-correctness → end-to-end parity always* |
| **C4 BYTES ACHIEVED** | print bytes actually streamed per matvec, read from the allocation, for **both** arms | must equal `2×` | D5's byte-accounting defect was that **the two arms were not charged comparably**. State the convention once, apply it to both |

**C2 is the one that has historically been skipped. It is not optional here.**

---

## 4. The prediction, registered before the number exists

This is the part that makes P1 worth running even if it wins nothing, because **it discriminates between the
two live explanations of D5 §12.6.**

D5 measured, on the ablated kernel with compute almost entirely stripped:

- **k/v rows: ~76% of the sequential fp32 ceiling** → close to bandwidth-limited.
- **gate/up rows: ~39.7% of the ceiling** → something in the **access pattern**, not the decode logic, caps it.

Therefore:

> **If bytes are the binding constraint, halving them buys ~2× on k/v and materially less on gate/up.**
> **If the access pattern is the binding constraint, gate/up barely moves regardless.**
>
> **Registered prediction: k/v gains more than gate/up.** If gate/up gains *equally*, my access-pattern
> reading is wrong and I want that on the record. If **neither** gains, the memory-bound verdict of §12.6 is
> itself in question and P1 has falsified something more valuable than it proposed.

**Every outcome of this probe is informative.** That is the test of whether it deserved to be built.

---

## 5. Sequencing — and the machine

The Owner's `llama-server` (22.7 GB resident) is back on the machine, so **bandwidth numbers are not
trustworthy right now**. Split accordingly:

- **Stage 1 — correctness, machine-agnostic, run now.** Layout, kernel, C1, C2, C3, C4. Produces no GB/s.
  **STOP with the control table.** A green C1/C2/C3/C4 is already a deliverable: it means the 2× is *available*
  whenever we want it, banked, independent of when the machine frees up.
- **Stage 2 — timing, gated.** Only on a demonstrably clear machine, with the D5 protocol: per-rep **mean**
  (not min-of-reps, which is a maximum order statistic and biases upward), multiple invocations, dispersion
  reported, and the `d5cd` resident-vs-streamed control re-fired to show the meter is honest at donor shape.

**Nothing from Stage 2 enters `ADAPTER_MEMO_01` until the Controller has audited it.** Stage 1 may be
recorded as a correctness fact on its own.

---

## 6. Scope — what this brief does NOT claim

- **It does not claim a 2× end-to-end speedup.** It claims a 2× reduction in weight-stream bytes, on a path
  D5 §12.6 found memory-bound, with per-weight `pshufb` count unchanged. **The end-to-end factor is the
  measurement, not the premise.** Phase-61's law stands: *a compute-bound microbenchmark does not compose to
  a memory-bound engine* — and its converse deserves the same suspicion.
- **It does not touch quality.** The arithmetic is bit-identical; there is no quantisation change, no
  approximation, no accuracy/BPB trade. That is unusual for this programme and is precisely why the
  correctness gate is absolute rather than statistical.
- **It does not retire 5-trit.** If P1 lands, 5-trit's remaining margin shrinks from 2.5× to **1.25× over
  P1**, against a base-3 decoder that cannot use `pshufb`. That is a much worse trade than it looked in
  §12.5, and **P1 should be measured before 5-trit is costed again.**

---

## 7. Prior art — dispatched in parallel, not blocking

Sub-byte `pshufb`-indexed ternary/low-bit kernels are a live area (T-MAC, bitnet.cpp, and the LUT-quant
line). **The question for the Researcher is narrow and answerable:** *do published LUT-based ternary kernels
already pack two trits per nibble, or do they also spend a byte per 2-trit index?*

- **If they pack the nibble**, then this is a defect in our engine we should simply fix, we cite them, and
  we make no novelty claim. **That is a perfectly good outcome — fixing it is worth the same either way.**
- **If they spend the byte too**, note it and move on. **This brief still does not rest on novelty**, and
  the standing law applies: *a gap is confirmed only for the literature searched.* We were burned on exactly
  this in R2 §0. **No novelty claim enters any document from this brief.**

---

## 8. Standing rules inherited by this brief

> Report **ACHIEVED**, not requested, parameters — printed from the objects themselves.
> An instrument must **fire on a known positive** before its nulls or its passes mean anything.
> A result that **flatters** the hypothesis earns more scrutiny, not less — **and this one flatters me.**
> **The producer of a measurement must not be told the expected conclusion.** §4's prediction is registered
> here, in the Principal's brief, and goes to the **Controller**. It is **not** to be restated to the
> Builder as an expectation, and the Builder is to report what it measures without reference to it.
