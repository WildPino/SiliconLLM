# E8 — is the weight path bandwidth-bound, or is it a dependency chain?

**Pre-registered. Pushed before any measurement was taken.** Opened by `SPEED_LEDGER.md` §19.5
(the FFN decomposition) and by E7 §12 owed item.

---

## 1. Why this exists

`SPEED_LEDGER.md` §19 is the most consequential arithmetic in the programme: of the **16.2×**
between today's rate and 50 tok/s at a 10 B dense shape, only **1.7–2.5×** is available to engine
work at all, and it is available *only* by moving the weight organs from today's **16.9 GB/s** to a
streaming rate this machine has actually been measured delivering (28 / 37 / 42 GB/s).

§19 asserts that gap exists. **It does not say what is holding the organs at 16.9 GB/s.** The
ledger's own reading, §12.5, is that nothing is:

> "After the rope hoist the four organs sit in a **1.3× band** (13.6–18.5 GB/s) and the ordering is
> by matrix size. **There is no outlier left to chase.** The remaining speed on this runtime is in
> the weight count, not in the kernel."

That paragraph reads a **narrow band** as evidence of nothing left to find. It is equally the
signature of a **single shared ceiling** — every one of those organs runs the *same* `matvec` inner
loop, so a limit belonging to that loop would pin them all at the same place, with matrix size
setting only how well the per-row costs amortise underneath it. §12.2 itself records that nothing
is "near the 40–44 GB/s DRAM aggregate or probe-3's ~28 GB/s post-L3 figure" and does not ask why.

**E4 is the precedent and the warning.** Attention was read as bandwidth-bound; it was
latency-bound; breaking the accumulator dependency was worth **6.53×** (14.647 → 2.242 ms). E8
asks that exact question of the weight path, which is **98.4% of the token** at target scale
(§19.1) against attention's 2.2%.

## 2. The mechanism, verified in the emitted assembly before it was claimed

Not an assumption. `clang -O3 -mavx2 -mfma -ffp-contract=on -fopenmp -S`, current source:

**The packed kernel** (`.LBB13_17`, the ternary path every published rate was taken on). clang
unrolls it 2×, but every FMA consumes the previous one's result:

```
vfmadd132ps  -32(%r14,%rdx,4), %ymm0, %ymm4    # ymm4 = ymm4*mem + ymm0
vfmadd132ps  -32(%r15,%rdx,4), %ymm4, %ymm0    # ymm0 = ymm0*mem + ymm4
vfmadd132ps     (%r14,%rdx,4), %ymm0, %ymm4    # ymm4 = ymm4*mem + ymm0
vfmadd132ps     (%r15,%rdx,4), %ymm4, %ymm0    # ymm0 = ymm0*mem + ymm4
```

`ymm4 → ymm0 → ymm4 → ymm0`: **one unbroken chain, 2 FMAs per 8 bytes of packed codes.**

**The fp32 kernel** (`.LBB11_78`) is the same shape, not unrolled at all — a single
`vfmadd231ps ... %ymm0` whose destination is its own operand.

**The compiler cannot fix this, and that is by our own rule.** Breaking the chain means
re-associating a float sum. `-ffast-math` is **forbidden in this repo** (Phase 35, reproducibility).
So the chain is not an oversight the optimiser will remove; it is a structural property of the
source that only a source change can alter.

### 2.1 What the chain costs, arithmetically

AMD Ryzen 5 3600X, Zen 2, 6 cores, `--threads 6`, 3.793 GHz base. `vfmadd*ps ymm`: **latency 5
cycles**, throughput 2/cycle.

A serially dependent chain runs at **latency**, so the loop cannot go faster than 5 cycles per FMA
no matter how many FMA units are idle or how much bandwidth is spare.

| | packed kernel |
|---|---|
| weight bytes consumed per FMA pair | **8** (16 ternary weights at 2 per byte) |
| cycles per FMA pair, chain-limited | **10** |
| per thread | 0.8 B/cycle |
| × 6 threads × 3.793 GHz | **18.2 GB/s** |
| × 6 threads × 4.2 GHz (all-core boost, upper) | **20.2 GB/s** |

**Predicted ceiling from the chain alone: 18.2–20.2 GB/s of packed weight bytes.**

### 2.2 Every measured organ sits at or under that ceiling

§12.2, Qwen2.5-0.5B, packed, idle, moved bytes at 0.5 B/weight:

| organ | measured GB/s | vs chain ceiling 18.2–20.2 |
|---|---|---|
| ffn | 13.6 | under |
| qkv | 14.2 | under |
| o_proj | 16.0 | under |
| **head** (largest matrix, best per-row amortisation) | **18.5** | **on it** |
| weights aggregate | 14.8 | under |
| Coder-7B, all weight organs (§13.3, 0.5 B/weight moved) | **~16.4** | under |

Against **28 GB/s** (probe-3 post-L3 streaming), **37.0** (proj-GEMV streamed floor) and **42**
(DRAM aggregate). **The organ band is not spread by bandwidth; it is pinned under a kernel-shaped
ceiling, and the organ with the least per-row overhead reaches it exactly.**

That is a hypothesis, and §2.3 is why it might still be wrong.

### 2.3 How this could be a coincidence

18.5 ≈ 18.2 is one number agreeing with one derivation. Three ways it is not the chain:

1. the sustained all-core clock under AVX2 load is lower than assumed, and the true chain ceiling
   is well above the measurements, which would leave bandwidth or something else binding;
2. the real binder is per-row cost — the horizontal sum, `scale`, store, and the OpenMP region —
   which also scales with matrix shape and would also produce a band ordered by matrix size;
3. the real binder is the activation-side deinterleave (`g_xe`/`g_xo`) or the `vpshufb`/`vpmovsxbd`
   /`vcvtdq2ps` front-end, in which case removing the chain frees nothing.

**The experiment distinguishes them, and §6 is the arm designed to embarrass this brief.**

## 3. What is built

`donor_engine.c`, one flag, `--mvacc {1,2,4}`, applied to **both** the packed and fp32 kernels.

- **`--mvacc 1` is a byte-for-byte copy of the existing loop.** The baseline arm is unchanged and
  its logits stay bit-identical to every number this programme has published. (E5's `avrep==1`
  discipline; the reason G-Z0a below can be an sha256 gate.)
- `--mvacc 2` uses two independent accumulators, `--mvacc 4` four. **Same bytes read, same FMA
  count, same order within a chain** — only the partition of the sum changes.
- Verified in the emitted assembly: `.LBB13_11` is four independent `vfmadd231ps`, into
  `ymm0`/`ymm1`/`ymm2`/`ymm3`, none feeding another.

**This is NOT bit-identical and is not claimed to be.** Re-partitioning a float sum changes
rounding. It is therefore gated on **end-to-end parity** (G-Z2), never on sha256 — Phase 60's law
that kernel-bit-exactness does not compose to system correctness applies in the other direction
too: bit-exactness is unavailable here, so the parity gate has to carry the whole load.

Separately, and answering §19.5's owed item directly: **the FFN organ is decomposed** into
`gate+up` / `glue(silu)` / `down` / `residual` by four `--profile`-only sub-timers, printed under
the organ table. They are deliberately **not** members of `g_t`, so the published organ table and
every percentage in it are unchanged.

> **DECLARED COST, on the §11 precedent.** These sub-timers add 8 timestamps per layer per token
> **under `--profile` only**. Profiled walls taken after E8's commit are therefore **not
> comparable** to profiled walls taken before it — the second time this programme has had to say
> that. Splits and slopes are unaffected. G-Z0b is the gate that the un-profiled `--bench` rate,
> which is what every headline number is, did not move.

## 4. Gates — thresholds fixed here, before anything was run

### G-Z0 — the instrumentation is inert

| | test | pass |
|---|---|---|
| **G-Z0a** | `--logits` sha256, E8 binary at `--mvacc 1` vs the pre-E8 binary (`126fa09`) | **byte-identical** |
| **G-Z0b** | 8 interleaved pairs, 0.5 B packed, `--bench 300`: E8 `--mvacc 1` / pre-E8 | paired median in **0.99–1.01** |

**FAIL on either ⇒ the instrumentation is rejected as built and moved behind a flag**, exactly as
§12's witness was rejected at 0.9877 rather than kept quietly.

### G-Z1 — the FFN, decomposed (descriptive, no verdict attached)

`--profile` at 0.5 B packed and at Coder-7B packed. **Sum check: `FFN-SUM / ffn` in 0.98–1.02**,
or one of the brackets is holding work it does not name. Reported as a table.

### G-Z2 — parity of the accumulator arms

On the 0.5 B **fp32** arm, whose `--mvacc 1` path is the one E1 validated against PyTorch at rel L2
1.5e-05, so a small delta against it bounds the delta against PyTorch by the triangle inequality:

| | test | pass |
|---|---|---|
| **G-Z2a** | `--logits`, rel L2 of `--mvacc 2` and `--mvacc 4` against `--mvacc 1` | **≤ 1.0e-05** |
| **G-Z2b** | `--generate` 160 tokens greedy, `--mvacc 2` and `--mvacc 4` vs `--mvacc 1` | **160/160 identical** |

**FAIL ⇒ the arm is rejected whatever its speed.** The comparator's planted control already
exists and already fired: E7's G-C, where the packed arm scored **0/160** against PyTorch on this
same instrument. A comparator that cannot fail is not a gate; this one has failed on purpose.

### G-Z3 — the rate, 0.5 B packed (supporting)

10 interleaved triples, `--mvacc 1` / `2` / `4`, `--bench 300`, contention witness on.

### G-Z4 — the rate at target scale, Coder-7B packed (the headline)

10 interleaved pairs, `--mvacc 1` vs whichever of {2,4} wins G-Z3, `--bench 300`, witness on.
`f` is 2.2% of the token here, so the token-level ratio and the weight-path ratio are the same
number to within the reporting precision.

**Verdict bands, fixed now:**

| paired median ratio | verdict |
|---|---|
| **≥ 1.25** | **CONFIRMED** — the weight path was chain-limited, not bandwidth-limited |
| 1.06 – 1.25 | **UNDECIDED** — a real effect, but far below §2.1's derivation, so the derivation is wrong even where its sign is right. Reported as such, not dressed up |
| **≤ 1.06** | **REFUTED** — the chain was not the binder |

1.06 is the floor because it must clear the instrument's own dispersion: §11 measured rate spreads
of 4.2% and 6.2% on this bench, and **E4's law is that a threshold inside the instrument's
dispersion cannot decide.**

**Predicted band, written before the run: 1.5–1.7× at Coder-7B.** From §2.1: removing the chain
lifts the kernel ceiling to roughly 36–72 GB/s (2 or 4 chains), above every measured streaming
ceiling, so **memory becomes the binder at ~28 GB/s** and the weight path moves 16.4 → ~28, which
is **1.71×** — and is exactly the "28 GB/s" row of ledger §19.2, i.e. E8 would realise the first
and smallest of §19's three headroom rows and nothing beyond it.

**Contention discard rule, fixed before the run:** every reading carries `ffn~` (§11). **Discard a
pair if either arm's `ffn~` exceeds that arm's own median by more than 10%** — the §11 plateau
spread was 4.6%, so 10% is roughly 2× it. Report how many pairs were discarded **and the ratio
both with and without the rule.** The witness is not the instrument under test here (it was gated
in §13-bis), so filtering with it is not the circularity §11.2 refused.

## 5. What a REFUTED verdict would be worth

It is not a wasted run. REFUTED says the weight path is genuinely memory-bound at 16–18 GB/s
against a machine that streams at 28–42, which would mean the gap is **cache and access pattern**,
not arithmetic — and it would say §19's 1.7–2.5× of engine headroom is not reachable by this route.
Either way §19.3's conclusion hardens: **the missing factor of 6.6–9.7× is a property of the model,
not of the code.**

## 6. The negative control — the arm designed to embarrass this brief

The same flag, the same code change, on the **fp32** kernel, where the derivation says it **cannot**
help:

fp32 consumes **32 bytes per FMA** against packed's 8, on a chain of 1 FMA per 5 cycles rather than
2 per 10. Per thread 6.4 B/cycle; × 6 threads × 3.793 GHz = **146 GB/s**, which is **3.5× above the
42 GB/s DRAM aggregate**. The chain has enormous slack there, so the fp32 path is genuinely
bandwidth-bound and breaking its chain must buy nothing.

| **G-Z5** | 8 interleaved pairs, 0.5 B **fp32**, `--bench 300`, `--mvacc 4` / `--mvacc 1` | predicted **0.99–1.03** |

> **If the fp32 arm speeds up by ≥ 1.10, the mechanism in §2 is wrong**, whatever the packed arm
> does — the effect would then be something the accumulator count merely correlates with (code
> layout, unrolling, the loop tail), and **the packed result must not be attributed to the
> dependency chain.** This is the closest a solo measurer can get to the no-anchoring rule: a
> prediction fixed in advance that can contradict the story I want to tell.

## 7. Run order

Cheapest first, so a failure costs the least:

1. **G-Z0** (0.5 B packed, ~2 min) — if this fails nothing else is worth running
2. **G-Z2** (0.5 B fp32, parity) — the arms must be admissible before they are timed
3. **G-Z1** (profile, 0.5 B and 7 B)
4. **G-Z3** (0.5 B packed rates)
5. **G-Z5** (the negative control) — **before** the headline, so the mechanism is tested before it
   is used to explain anything
6. **G-Z4** (Coder-7B packed, the headline)

Machine idle throughout; every rate carries its `--bench` length (§12.3) and its `ffn~` (§11).
**Every absolute tok/s carries ±5% between sweeps; the paired ratios do not.**


---

## 8. VERDICT, 2026-09-07 — `CHAIN-CONFIRMED`

Full write-up: `probes/E8_MATVEC_DEPENDENCY_CHAIN.md`. Every gate in §4 ran; nothing was added
to the gate set after the fact, and nothing in it was relaxed.

| gate | pass condition, fixed in §4 | result |
|---|---|---|
| G-Z0a | sha256 byte-identical | **PASS** `d4960bc7…950424` |
| G-Z0b | paired median 0.99–1.01 | **PASS 1.0076** |
| G-Z1 | `sum/ffn` in 0.98–1.02 | **PASS 1.0000** |
| G-Z2a | rel L2 ≤ 1.0e-05, top-1 8/8 | **PASS 2.642e-06 / 3.349e-06** |
| G-Z2b | 160/160 greedy identical | **PASS**, sha256 equal across all three arms |
| G-Z3 | descriptive | m2/m1 **1.2125**, m4/m1 **1.2301** |
| **G-Z5** | **0.99–1.03; ≥1.10 refutes §2** | **PASS 1.0029** |
| **G-Z4** | ≥1.25 CONFIRMED | **CONFIRMED, 1.3491** |

**Coder-7B: 4.73 → 6.37 tok/s.** Weight path 16.7 → 22.5 GB/s. All four weight organs moved
1.41–1.49× from one change to one inner loop.

**Three things this brief got wrong or under-specified, recorded rather than tidied away:**

1. **The magnitude prediction, §4: 1.5–1.7× predicted, 1.349× measured.** The verdict band was
   met, the derivation was not. Removing the chain did **not** hand the kernel to memory at
   28 GB/s; it hit a second kernel-shaped limit at 22.5. Second missed cost prediction in two days.
2. **The contention discard rule was useless.** "Either arm's `ffn~` more than 10% over that arm's
   own median" cannot see a contamination that moves the whole sweep together — and one did, a
   surviving child process of my own killed runner. It discarded **zero** pairs in every sweep,
   contended and clean alike. Only the *absolute* plateau carried across sessions caught it.
   Probe §7.2.
3. **§2.3's alternative #2 was refuted by arithmetic, not by the experiment.** `gate+up` moves
   exactly 2× `down`'s bytes while being chopped 10.6× differently; the measured ratio is
   1.990–2.019. Per-row cost was never the binder, and the brief did not notice it had a free
   test for that sitting in the decomposition.
