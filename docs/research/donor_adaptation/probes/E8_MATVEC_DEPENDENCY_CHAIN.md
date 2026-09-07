# E8 — the weight path was not bandwidth-bound. It was a dependency chain.

**Verdict: `CHAIN-CONFIRMED`.** Pre-registered and pushed as
`briefs/BRIEF_E8_MATVEC_DEPENDENCY_CHAIN.md` (`406b02b`) before any measurement was taken.

Donor: `Qwen2.5-Coder-7B`, packed arm, **7.072 B active weights/token** — the same artefact E7
validated at rel L2 1.070e-05 and 160/160 greedy against PyTorch. Supporting shape: Qwen2.5-0.5B.

---

## 1. The headline

| | `--mvacc 1` (the loop as it has always been) | `--mvacc 4` |
|---|---|---|
| **Coder-7B, `--bench 300`, 10 interleaved pairs** | **4.73 tok/s** (spread 1.9%) | **6.37 tok/s** (spread 2.8%) |
| `ffn~` witness | 168.550 ms (spread 2.7%) | 124.469 ms (spread 2.6%) |
| **paired median ratio** | — | **1.3491** |

Per-pair ratios: 1.3553 1.3368 1.3615 1.3475 1.3397 1.3425 1.3510 1.3383 1.3507 1.3591 —
**a 1.8% spread across ten pairs, no pair dissenting.** Band fixed before the run: ≥1.25
CONFIRMED. **CONFIRMED.**

**Delivered weight rate: 33.5 → 45.0 G-weights/s; 16.7 → 22.5 GB/s moved** (0.5 B/weight,
§12.2's convention — *moved* bytes, not E7's 0.515 charged).

> **±5% applies to each absolute, not to the ratio.** The ratio is paired and interleaved.

## 2. What was wrong, and it was one line of shape

Every weight organ in this engine — `qkv`, `o`, `ffn`, `head` — goes through the same `matvec`,
and every `matvec` inner loop accumulated into **one** register. Verified in the emitted assembly
**before** the claim was made, not after (`.LBB13_17`, the packed kernel; clang unrolls 2×):

```
vfmadd132ps  -32(%r14,%rdx,4), %ymm0, %ymm4    # ymm4 = ymm4*mem + ymm0
vfmadd132ps  -32(%r15,%rdx,4), %ymm4, %ymm0    # ymm0 = ymm0*mem + ymm4
vfmadd132ps     (%r14,%rdx,4), %ymm0, %ymm4    # ymm4 = ymm4*mem + ymm0
vfmadd132ps     (%r15,%rdx,4), %ymm4, %ymm0    # ymm0 = ymm0*mem + ymm4
```

`ymm4 → ymm0 → ymm4 → ymm0`: **one unbroken chain.** On Zen 2 an FMA has **5-cycle latency** and
2/cycle throughput, so the loop ran at *latency*, and no amount of spare bandwidth could be used
past it. **The compiler could not fix this, by our own rule**: breaking the chain re-associates a
float sum, and `-ffast-math` is forbidden here since Phase 35.

`--mvacc {1,2,4}` gives the loop 1, 2 or 4 independent accumulators. Same bytes, same FMA count,
same order *within* a chain.

## 3. The prediction was right in direction and wrong in size — again

**Predicted before the run: 1.5–1.7×. Measured: 1.349×.** Below my own band.

The derivation assumed that removing the chain would let **memory** become the binder at probe-3's
28 GB/s. It did not. The kernel stopped at **22.5 GB/s**, at a *second* kernel-shaped limit — the
`vpshufb` / `vpmovsxbd` / `vcvtdq2ps` front-end work the packed format needs, roughly 11 vector
uops per 8 bytes of codes.

This is the **second** time in two days a cost/benefit prediction of mine has missed while its
direction held (the contention witness cost 1.2% against a 0.008% prediction, 150× off). The
corollary already written into the working rules applies to what comes next: **derive the next band
from the measured 22.5, not from first principles a third time.**

## 4. Every gate, as run

| gate | test | result |
|---|---|---|
| **G-Z0a** | E8 binary at `--mvacc 1` vs the pre-E8 binary, `--logits` sha256 | **PASS** — byte-identical, `d4960bc72b696c2b…950424` |
| **G-Z0b** | 8 interleaved pairs, does the new instrumentation cost anything? | **PASS** — paired median **1.0076** (band 0.99–1.01) |
| **G-Z1** | the FFN decomposed; do the four sub-timers sum to the organ? | **PASS** — `sum/ffn` = **1.0000** at 7 B, **0.9996** at 0.5 B |
| **G-Z2a** | rel L2 of `--mvacc 2` / `4` against `--mvacc 1`, 0.5 B fp32, 8 positions | **PASS** — **2.642e-06** and **3.349e-06** (gate ≤1.0e-05); top-1 **8/8** |
| **G-Z2b** | 160-token greedy generation, all three arms | **PASS** — sha256 `dc9d62a2b544afdd…` **identical across all three** |
| **G-Z3** | 0.5 B packed, 10 interleaved triples | m2/m1 **1.2125**, m4/m1 **1.2301** |
| **G-Z5** | **negative control** — the same flag on the fp32 kernel, which must NOT speed up | **PASS** — **1.0029**, inside the predicted 0.99–1.03 |
| **G-Z4** | Coder-7B, 10 interleaved pairs — **the headline** | **CONFIRMED, 1.3491** |

**The parity gates are the load-bearing ones.** `--mvacc 4` is *not* bit-identical and is never
claimed to be; re-partitioning a float sum changes rounding. It is 3.349e-06 from the single-chain
arm, which is itself 1.5e-05 from PyTorch — **the reorder is five times smaller than the engine's
existing distance from the reference**, and 160 greedy tokens come out byte-for-byte the same.

## 5. The negative control, which is the part that makes §2 more than a story

The same flag, the same code change, on the **fp32** kernel — where the derivation says it *cannot*
help, because fp32 moves 32 bytes per FMA against packed's 8, giving the chain 3.5× slack over the
42 GB/s DRAM aggregate. **Predicted 0.99–1.03 in the pushed brief, with ≥1.10 declared in advance
to refute the whole mechanism.**

**Measured 1.0029.**

And it returned something it was not designed for. The fp32 arm delivers **33.3 GB/s** (494 M
weights × 4 B × 16.86 tok/s) through *this same `matvec`*, on *this same machine*, in *this same
run*, while the packed arm was managing 16.7. **That is an in-run demonstration that the packed
path was never at a memory limit** — independent of any cycle counting.

## 6. §19.5's owed item: the FFN, decomposed

`--profile`, Coder-7B packed, `--bench 100`. Sub-timers of `T_FFN`, not members of the organ table:

| | `--mvacc 1` | `--mvacc 4` | × |
|---|---|---|---|
| **ffn** (80.6% of the token) | **177.928** | **126.304** | 1.409 |
| gate+up | 109.599 | 76.661 | 1.430 |
| down | 55.070 | 37.973 | 1.450 |
| **glue(silu)** | **13.215** | **11.628** | — |
| residual | 0.038 | 0.035 | — |
| `sum/ffn` | 1.0000 | 0.9999 | |

and the organs outside it moved by the same factor, which is the point:

| | `--mvacc 1` | `--mvacc 4` | × |
|---|---|---|---|
| qkv_proj | 14.372 | 10.069 | **1.427** |
| o_proj | 10.670 | 7.456 | **1.431** |
| head | 15.726 | 10.524 | **1.494** |
| attention (no `matvec`) | 1.780 | 1.673 | — |

**One change to one inner loop moved all four weight organs by 1.41–1.49×.** That is the signature
of a shared kernel ceiling being lifted, not four separate wins, and it is the strongest internal
evidence that §12.5's "no outlier left to chase" was reading a *shared limit* as an absence.

### 6.1 An invariant that kills the per-row-overhead explanation without a timing

`gate+up` moves **exactly 2×** the weight bytes of `down` — 2×(F×D) against D×F. Structural, not
measured. **Under a pure bandwidth limit the time ratio must be 2.000.**

| | measured ratio | vs 2.000 |
|---|---|---|
| Coder-7B, `--mvacc 1` | 109.599 / 55.070 = **1.990** | −0.5% |
| Coder-7B, `--mvacc 4` | 76.661 / 37.973 = **2.019** | +0.9% |
| 0.5 B, `--mvacc 1` | 6.095 / 3.138 = **1.942** | −2.9% |

Yet those two matrices differ **10.6×** in how the work is chopped: 37,888 output rows against
3,584. **Per-row and per-call cost therefore cannot be what holds the FFN** — brief §2.3's
alternative explanation #2, refuted by arithmetic rather than by a measurement.

### 6.2 `glue(silu)` — 11.6 ms/token for zero weight traffic

**530,432 `expf` calls per token** (F × L = 18,944 × 28), now **9.2% of the FFN** and **7.4% of the
whole token** because the matvecs around it shrank. Nothing in this ledger has ever charged it.
It is the same shape as the rope `pow()` hoist that was worth **+10.3%** in §12, and **E8 does not
touch it.** Owed, and cheap.

## 7. Two errors of mine, and a VOID run

**7.1 The corpus handed to `--generate` as a prompt.** G-Z2b's first runner passed
`ids_qwen25-05b_tqh.bin` — the whole **12,288-token** corpus — where a prompt belonged. The engine
did exactly as told: prefilled all of it and wrote **7.5 GB of prefill logits per arm**. Killed at
13 GB, re-run with a 32-token prompt. Same gate, wrong invocation — the same class of error as
G-W2's wrong argument form the day before.

**7.2 A kill that did not kill, and the witness caught it.** Stopping that runner left a child
alive: `donor_engine_e8.exe --generate … --mvacc 4`, PID 21904, created 14:12:29, **spawned by the
loop in the race with the stop**. It ran six threads against the next 45 minutes of measurements.

**The first G-Z1/G-Z3/G-Z5 sweep is VOID** (`D:/_ktmp/e8/gz1_VOID_contended.log`). The witness
read `ffn~` at **22–23 ms** against its 11.552–12.083 idle plateau, and the 0.5 B profile came out
at 27.96 tok/s where §12.2 has 56 — **a ~2× slowdown, i.e. six threads of my own engine against
six.** Re-run clean after the kill: `ffn~` 11.879 / 12.009 / 11.978, rate 53.6–54.2.

> **This is the contention witness (§11) firing on a real positive that nobody planted.** The
> voided sweep produced m4/m1 = **1.12** — plausible, publishable, and sitting squarely in the
> pre-registered UNDECIDED band. Without the witness it would have gone into the ledger as a
> genuine result and buried a 1.35× win under a shrug. The clean answer is 1.2301 at 0.5 B and
> 1.3491 at 7 B.

**And the pre-registered discard rule caught none of it.** "Discard a pair if either arm's `ffn~`
exceeds that arm's own median by >10%" is useless when *every* reading in the sweep is contended
together — the median moves with them. **A within-sweep rule cannot detect a whole-sweep
contamination**; only the *absolute* plateau, carried across sessions, can. That is a correction to
how §11's witness should be used, and it is the reason a plateau is worth publishing as a number.

## 8. What this does and does not buy

**Adopted as the default** (`g_mvacc=4`). `--mvacc 1` restores the single-chain loop **byte for
byte** and is how every rate published before E8 is reproduced — verified: the sha256 above is
identical under the new binary at `--mvacc 1`. Defaulted rather than left as a flag because
**a flag a runner must remember to pass is precisely the §9 defect**, where `--attn` sat on the
slow kernel for every E3 and E7 number.

> **Every rate published before this commit is a `--mvacc 1` reading.** They are not withdrawn and
> not wrong; they are labelled.

**It does not reach the goal.** 6.37 tok/s on 7.072 B active is **7.8× short of 50 tok/s**, where
it was 11.2× short. And it takes 1.35× out of the 1.7–2.5× that ledger §19.3 said was the *whole*
budget available to engine work, leaving **1.24–1.87×** to the measured streaming ceilings.

**§19.3's conclusion is unchanged and now better supported: the missing factor is a property of the
model, not of the code.**

## 9. Owed

1. **`glue(silu)`, §6.2** — 530,432 `expf`/token, 7.4% of the token, untouched. Cheapest thing on
   the list and the rope hoist is the precedent.
2. **The next binder.** The kernel stopped at 22.5 GB/s, not at memory. Per §3 the next band is
   derived from 22.5 and the structural factor, **not** from first principles. The suspect is the
   packed format's unpack work (~11 vector uops per 8 bytes).
3. **The non-packed ternary path still has a single accumulator.** `--mvacc` was applied to the
   packed and fp32 kernels only. Not on the measured path, so not urgent, but it is a known
   remaining chain.
4. **A witness plateau per shape, published as an absolute**, per §7.2 — the within-sweep discard
   rule cannot see a whole-sweep contamination.
