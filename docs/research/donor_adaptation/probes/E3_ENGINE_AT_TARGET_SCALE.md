# E3 — The engine at target scale, and the 1.17 ms nobody had measured above 0.5 B

**Brief:** `briefs/BRIEF_E3_ENGINE_AT_TARGET_SCALE.md` — §1–§7 pre-registered at `d4937a2` before
the runner existed; §8 (run 1's `VOID` and the repair) at `b97c8e6` before run 2 existed; §9 (run 2's
gates) at `a75e37b` before any arm above `S15` was generated.

**Runners:** `benchmarks/donor_adaptation/engine/synth_export.py`,
`benchmarks/donor_adaptation/engine/e3_bench.py`, `e3_run_arms.sh`.
**Data:** `benchmarks/donor_adaptation/results/e3/{gates,gates2,gates2_interleaved,arms}.json`.

---

> ## ⚠ AMENDED 2026-09-06 by E4 — read this before quoting anything below
>
> **1. The ceiling claim is WITHDRAWN.** §4.5 concluded that a 10 B could not pass **38.3 tok/s at
> 800 tokens of context even with a free weight path**, and §4.6 called the attention organ's
> ~4 cycles/FMA constant *corroborated, not proven*. `probes/E4_ATTENTION_ACCUMULATORS.md` proved it
> and removed it: the `Q·K` reduction was **latency-bound**, 40 lines of AVX2 took it 14.647 → 2.242
> ms, `f` 24.678 → **12.735 ms** and the ceiling 40.5 → **78.5 tok/s**, at `|ΔBPB| = 3.03e-06`.
> **50 tok/s at 800 context is a weight-side problem again**, and §4.5's budget of *zero* at 800 is
> now **259 M**. §7's twelve-donor screen is re-priced in E4 §4.7.
>
> **2. ±5% band on every absolute tok/s in this probe.** E3's own binary, rebuilt and re-run in a
> later session, missed this probe's published table by **−1.6% to −4.2%**, and the same code read
> **3.240** and **3.030** at `T10` @300 two hours apart while the within-run IQR stayed at 0.005.
> Between-sweep dispersion here is **5–10%**; within-sweep is 0.000–0.015 tok/s. Every absolute
> tok/s and G-weights/s below carries that band. **The ratios do not** — they were taken inside one
> sweep — and this probe's conclusions about *shape* (the rate improving with scale, the head
> falling to 1.1%, `f` breaking its reservation) are ratios and stand.
> `SPEED_LEDGER.md` §14.3 is the law; this is its instance here.

---

## 0. Verdict

**Run 1: `VOID`** (Gate V1 fired, Gate V2 failed — §2.2, §2.3).
**Run 2: `RESERVATION-BREAKS`**, plus a rate outcome brief §6 has no bucket for.

`SPEED_LEDGER` §12.2 reserves **1.17 ms/token** for the non-weight work and says of itself that this
is *"optimistic for a 10 B"*. Measured at the goal's own shape:

| | `f` = rope + attention + norm | vs the 1.17 ms reservation |
|---|---|---|
| `T10` @ 300 tokens of context | **10.576 ms** | **9.0×** |
| `T10` @ 800 tokens of context | **26.088 ms** | **22.3×** |

Consequences, each a measurement and not an extrapolation:

- **A 10 B shape at 800 context cannot reach 50 tok/s even if every weight were free** — `f` alone
  caps it at **38.3 tok/s**. 100 tok/s is out of reach at 300 context for the same reason (94.6).
  No weight format, sparsity scheme or MoE touches `f`.
- The 50 tok/s active-weight budget is **318 M at 300 context (3.0% of a 10 B)**, not 522 M and 5.2%;
  at 800 context it is **zero**. **`INDEX` §7's twelve-donor screen is superseded.**
- `INDEX` §7's 522 M also **charges the fixed cost twice** — `27.7 G-w/s` is already `weights / wall`.
  Self-consistently it is 554 M.
- **The rate does not degrade with scale, it improves**: 27.5 → **32.8 G-weights/s** (+18.3%) from
  0.5 B to 10 B. The extrapolation that prices this programme was conservative on the weight term and
  optimistic on `f` — and only the second one mattered.
- **The head stops being the floor.** 20.5% of the token at 0.5 B, **1.1%** at `T10`; the attention
  projections are 17.9%. The matched pair `M7`/`Q8` confirms the tokenizer sets the head end-to-end:
  4.629× the time for 4.637× the vocabulary, **0.2%**.
- `T10` measured: **3.090 tok/s** @300, **2.960** @800 — the dense path is **16.2×** from the goal,
  which is just `50 / 3.090` and needs no rate convention at all.
- `f` is **one FMA per ~4 cycles per thread across all twelve points** (±7%), invariant to KV size,
  GQA and vocabulary: a serial FP reduction the build cannot vectorise, not a bandwidth wall.
  **The KV path has never been touched** and is where the remaining distance is.

---

## 1. What this asks

Everything this programme knows about speed was measured on **Qwen2.5-0.5B**. `P3` reached 3B but
only through the **matvec stack** — no RoPE, no attention over a growing context, no KV cache, no
norms. The 522 M active-weights/token budget that screens twelve donors in `INDEX` §7 is not
`27.7 / 50`; it is `27.7 G-w/s × (20 − 1.17) ms`, and the **1.17 ms is a reservation for the
non-weight work** measured at 0.5 B. `SPEED_LEDGER` §12.2 flags its own weak point:

> *"The 1.17 ms is measured at 0.5 B and short context and is **optimistic for a 10 B**, where
> attention and the KV cache both grow."*

That sentence had never been tested. This is E1's law aimed at the goal itself: **a ceiling nothing
has reached is a ceiling nobody has tested** — the same shape as the 2 GB load limit that stayed
invisible until something big was handed to the loader.

### 1.1 Why synthetic weights

A timing needs shape and format, not training. Every file here is written by `synth_export.py` in the
real `QWENDON1` format at a real donor's shape, with noise weights. **Nothing here has a BPB and none
is computed.** That this is legitimate is not argued, it is gated: §2.

---

## 2. Gates

### 2.1 Gate V3 — the file is the format. **PASS**, at four independent definitions

| definition | `S05` | `S15` |
|---|---|---|
| `e1_bpb_through_engine.layout` + `nbytes`, walked from the header alone | 291 tensors, 793,629,748 B | 339 tensors, 1,709,047,348 B |
| the **real** export of the same shape and configuration | `qwen05b_packed.bin` = 793,629,748 | `qwen25-15b_tqh.bin` = 1,709,047,348 |
| `donor_engine.c`'s own loader | `layout OK: consumed exactly …` | idem |
| the engine emits a finite `BENCH` line | yes | yes |

Exact, not approximate. `S3` 2,793,211,444 · `M7` 4,101,128,244 · `Q8` 6,281,659,956 ·
`T10` 5,849,628,724 pass the first, third and fourth (no real export exists at those shapes).

`Q8` at **6.28 GB** and `T10` at **5.85 GB** are also the first things this programme has loaded far
above the 2 GB `ftell` limit E1 found and fixed. They load.

### 2.2 Gate V1 (run 1) — **FIRED**. The probe stopped, and the finding is `expf`

Pre-registered: two files of identical shape, one with every code zero, one with none.

| `S05`, bench 300, 6 threads, 3 reps | median tok/s | IQR |
|---|---|---|
| every code zero | **39.580** | 0.215 |
| no code zero | **36.490** | 0.140 |

3.090 apart on an IQR of ~0.2. **By brief §6, run 1's label is `VOID` and no arm was generated.**

Where it fired is one organ:

| organ | zero | dense | Δ ms/token |
|---|---|---|---|
| `ffn` | 9.021 | 11.101 | **+2.080** |
| `head` | 13.746 | 13.818 | +0.072 |
| `qkv_proj` | 0.846 | 0.851 | +0.005 |
| `o_proj` | 0.588 | 0.585 | −0.003 |
| `attention` | 1.027 | 1.017 | −0.010 |
| `rope` / `norm+glue` | 0.016 / 0.067 | 0.017 / 0.067 | ≤0.001 |

`qkv_proj`, `o_proj` and `head` are **pure packed matvecs and they are flat to 0.005 ms** across a
change from every weight zero to no weight zero. The brief's §2 reading of the kernel — `pshufb` +
FMA, no early exit on a zero code — is therefore no longer a reading. **It is measured.**

The `ffn` bucket is the only one holding something that is not a matvec:
`silu(x) = x/(1+expf(-x))`, scalar `expf`, `F × L = 116,736` calls per token. An all-zero model feeds
it exactly `0.0f`, the early-out of every libm. 2.080 ms / 116,736 = **17.8 ns per call** — `expf`
scale, not memory scale.

**The gate was mine and it was badly placed.** A zero fraction of 1.0 is not something any exporter
can produce; across the range a real export occupies it is flat: `mixed` (0.47, what E2 measured on
real R3 exports) **36.570** vs `dense` (0.00) **36.490**, inside both IQRs. That is a flaw in the
gate, not a licence to ignore it, so the `VOID` stands and §2.4 is a different gate, not a rereading
of this one.

> **Law:** *a planted control must be planted inside the range the instrument will actually be used
> in.* An all-zero model is not a model, and the gate that used one measured `expf` instead of the
> kernel it was defending.

### 2.3 Gate V2 (run 1) — **FAILED**, and the synthesizer was not the reason

| `S05`, bench 300 | median tok/s | IQR | `head` ms/token |
|---|---|---|---|
| synthetic, `mixed` | 36.570 | 0.040 | 13.724 |
| **real** `qwen25-05b_tq.bin` | 36.440 | 0.540 | 13.829 |
| **real** `qwen25-05b_nl.bin` (R3 + fold) | 36.460 | 0.050 | 13.844 |

Synthetic and real, same shape and same 724,954,676 bytes, agree to **0.36%** — inside the real
artifact's own IQR, organ by organ. Neither is anywhere near the ledger's 56.1.

The cause was in `SHAPES`, and it is mine: it carried each donor's own `tied` flag, and **a tied
model runs its head as the fp32 embedding** — 151,936 × 896 × 4 = 544.6 MB/token, 13.8 ms, **52% of
the token**. Brief §3 had asked for a *ternary* head. `SPEED_LEDGER` §12.2 says its anchor is untied
and packed without ever naming it: `head 136,134,656 weights, 68.1 MB/token` is **0.5 bytes per
weight**. `--head {ternary,donor}` was added, default `ternary`.

**What this cost, stated plainly:** run 1 measured a configuration nobody intends to ship, because the
brief said "ternary head" in prose and the code read a flag out of a config file. Six arms were not
generated on the strength of it.

### 2.4 Gate V1′ (run 2) — **PASS at zero**, and it is the one that could still have failed

Fixed in brief §8.4 **before any synthetic 1.5B existed**, at the one scale where a real artifact is
on disk and no synthetic file had been made.

| bench 300, 6 threads, 3 reps | bytes | median tok/s | IQR |
|---|---|---|---|
| **real** `qwen25-15b_tqh.bin` | 1,709,047,348 | **19.430** | 0.075 |
| **synthetic** `S15 --head ternary --codes mixed` | 1,709,047,348 | **19.430** | 0.065 |

| organ | real | synthetic | Δ |
|---|---|---|---|
| `qkv_proj` | 2.840 | 2.743 | −0.097 |
| `rope` | 0.034 | 0.034 | 0.000 |
| `attention` | 1.981 | 1.967 | −0.014 |
| `o_proj` | 1.956 | 1.957 | +0.001 |
| `ffn` | 38.276 | 38.355 | **+0.079** |
| `head` | 6.357 | 6.381 | +0.024 |
| `norm+glue` | 0.137 | 0.138 | +0.001 |

`ffn` is the organ §2.2 had just declared a confound. **+0.079 ms on 38.276 is 0.2%**: at this width
random weights and trained R3 weights cost the same in SiLU. The confound was declared before it was
measured, and then measured — it is not live.

### 2.5 Gate V2′ (run 2) — the literal constant **fails**, and the ground truth fails it too

Brief §8.4 fixed the reference at *"57.790, IQR 0.190"*, measured on the real `qwen25-05b_tqh.bin`
twenty minutes earlier. Synthetic `S05 --head ternary` gives **56.180 (IQR 0.130)** — outside it.

The same real file was then re-run **interleaved A/B/A/B**, four adjacent pairs:

| | median | IQR | reps |
|---|---|---|---|
| **real** `qwen25-05b_tqh.bin` | **56.095** | 0.188 | 56.42 / 55.76 / 56.08 / 56.11 |
| **synthetic** `S05_th` | **55.895** | 0.500 | 55.93 / 54.26 / 55.86 / 56.05 |

Δ = **0.200 tok/s = 0.357%**. Per-pair deltas 0.49 / 1.50 / 0.22 / 0.06 — the outlier is a pair in
which *both* files dipped together, which is what a load transient looks like, and is why the pairs
are adjacent in time.

**The real artifact moved 57.790 → 56.095 in twenty minutes: 2.9%.** The reference constant is not
reproducible by the file that produced it. A gate its own ground truth cannot pass is measuring
drift.

> **Law:** *a speed gate may not contain a hard constant; it must name a file to be measured
> concurrently.* This is the fourth time this ledger has been bitten by the same family — §11 a rate
> taken under contention, §12 a timer bracketing the wrong work, §2.3 above an anchor on the wrong
> head, and now a constant that did not survive twenty minutes.

**Operative form** (a change made after seeing data, and stated as such): synthetic must match the
real artifact of the same shape **measured interleaved in the same session**. On that form V2′
passes at 0.357% at `S05`, and V1′ — fixed before its file existed — passes at 0.000% at `S15`.

### 2.6 Timing discipline actually achieved

Idle machine with **3–8% background load** (`Win32_Processor.LoadPercentage`, 5 samples), this
desktop's floor with a browser open; ≥3 reps per point; median and IQR reported; strictly one timer
at a time (`e3_run_arms.sh` is sequential). No quality number is claimed anywhere: the weights are
noise.

---

## 3. Arms — six shapes, two context lengths, 3 reps each

`--head ternary --codes mixed`, `--quant packed`, 6 threads, sequential, median of 3 with IQR.
`f` = `rope + attention + norm+glue`, the non-weight fixed cost the ledger reserves 1.17 ms for.

### 3.1 `--bench 300` (the ledger's operating point; mean position 150.5)

| arm | active w/tok | predicted tok/s (§5) | **measured tok/s** | IQR | wall ms/tok | weight-organ ms | **f** ms |
|---|---|---|---|---|---|---|---|
| `S05` | 0.494 B | 56.1 | **55.700** | 0.790 | 17.952 | 16.791 | **1.183** |
| `S15` | 1.544 B | 17.9 | **19.390** | 0.025 | 51.581 | 49.524 | **2.147** |
| `S3` | 3.086 B | 9.0 | **9.930** | 0.035 | 100.715 | 97.096 | **3.843** |
| `M7` | 7.114 B | 3.9 | **4.610** | 0.000 | 217.090 | 210.492 | **7.120** |
| `Q8` | 7.568 B | 3.7 | **4.340** | 0.015 | 230.570 | 223.191 | **7.962** |
| **`T10`** | **10.603 B** | **2.6** | **3.090** | 0.005 | 323.671 | 313.880 | **10.576** |

| arm | qkv_proj | rope | attention | o_proj | ffn | head | norm+glue |
|---|---|---|---|---|---|---|---|
| `S05` | 0.882 | 0.018 | 1.096 | 0.612 | 11.610 | 3.687 | 0.069 |
| `S15` | 2.793 | 0.035 | 1.974 | 1.957 | 38.379 | 6.395 | 0.138 |
| `S3` | 5.774 | 0.045 | 3.556 | 4.426 | 78.296 | 8.600 | 0.242 |
| `M7` | 23.201 | 0.061 | 6.633 | 15.248 | 168.322 | 3.721 | 0.426 |
| `Q8` | 26.112 | 0.063 | 7.420 | 17.148 | 162.706 | 17.225 | 0.479 |
| `T10` | 34.862 | 0.086 | 9.862 | 22.866 | 252.436 | 3.716 | 0.628 |

### 3.2 `--bench 800` (mean position 400.5)

| arm | **measured tok/s** | IQR | wall ms/tok | weight-organ ms | **f** ms |
|---|---|---|---|---|---|
| `S05` | **49.280** | 1.510 | 20.292 | 17.348 | **2.935** |
| `S15` | **17.960** | 0.065 | 55.688 | 50.448 | **5.259** |
| `S3` | **9.500** | 0.055 | 105.239 | 96.077 | **9.212** |
| `M7` | **4.420** | 0.030 | 226.103 | 209.061 | **17.164** |
| `Q8` | **4.130** | 0.005 | 242.135 | 222.991 | **19.269** |
| **`T10`** | **2.960** | 0.000 | 337.271 | 311.374 | **26.088** |

| arm | qkv_proj | rope | attention | o_proj | ffn | head | norm+glue |
|---|---|---|---|---|---|---|---|
| `S05` | 0.960 | 0.018 | 2.847 | 0.657 | 11.935 | 3.796 | 0.070 |
| `S15` | 2.874 | 0.034 | 5.087 | 2.022 | 39.066 | 6.486 | 0.138 |
| `S3` | 5.728 | 0.042 | 8.936 | 4.418 | 77.410 | 8.521 | 0.234 |
| `M7` | 23.106 | 0.057 | 16.688 | 15.200 | 167.040 | 3.715 | 0.419 |
| `Q8` | 26.244 | 0.065 | 18.730 | 17.246 | 162.406 | 17.095 | 0.474 |
| `T10` | 34.597 | 0.078 | 25.385 | 22.800 | 250.272 | 3.705 | 0.625 |

---

## 4. What the numbers say

### 4.1 The reservation breaks, and the size of the break is the result

`f` against the 1.17 ms `INDEX` §7 reserves:

| arm | f @300 | **× 1.17** | f @800 | **× 1.17** |
|---|---|---|---|---|
| `S05` | 1.183 | **1.0×** | 2.935 | **2.5×** |
| `S15` | 2.147 | **1.8×** | 5.259 | **4.5×** |
| `S3` | 3.843 | **3.3×** | 9.212 | **7.9×** |
| `M7` | 7.120 | **6.1×** | 17.164 | **14.7×** |
| `Q8` | 7.962 | **6.8×** | 19.269 | **16.5×** |
| **`T10`** | **10.576** | **9.0×** | **26.088** | **22.3×** |

At `S05` the ledger's 1.17 reproduces to 1%. At the goal's own shape it is **9× wrong at 300 tokens
of context and 22× wrong at 800**. Brief §6's threshold was `f ≤ 2.0 ms`; `f` crosses it at **1.5 B**.

**The sharpest consequence, and it is not about quantization at all.** `f` is work no weight format
can remove: rope, the attention loop over the KV cache, and the norms. Divide 1000 ms by it and you
get the engine's ceiling **if every weight were free**:

| | f | ceiling at zero weight cost |
|---|---|---|
| `T10` @300 | 10.576 ms | **94.6 tok/s** |
| `T10` @800 | 26.088 ms | **38.3 tok/s** |
| `Q8` @800 | 19.269 ms | 51.9 tok/s |
| `M7` @800 | 17.164 ms | 58.3 tok/s |

**At a 10 B shape with 800 tokens of context this engine cannot reach 50 tok/s even with an
infinitely fast weight path**, and it cannot reach 100 tok/s at 300 either. No amount of ternary,
sparsity or MoE changes those two numbers, because none of them touch `f`. That is the finding.

### 4.2 The rate does not degrade with scale — it improves, and §6 had no bucket for that

Two conventions, and mixing them is the failure `SPEED_LEDGER` §12.1 already withdrew once:

| arm | `r_w` = w / weight-organ ms | **`r_wall` = w / wall ms (the ledger's)** | weight-organ GB/s |
|---|---|---|---|
| `S05` | 29.4 | **27.5** | 14.7 |
| `S15` | 31.2 | **29.9** | 15.6 |
| `S3` | 31.8 | **30.6** | 15.9 |
| `M7` | 33.8 | **32.8** | 16.9 |
| `Q8` | 33.9 | **32.8** | 17.0 |
| `T10` | 33.8 | **32.8** | 16.9 |

Judged in the ledger's own convention, `T10` delivers **32.76 G-w/s against 27.7 — +18.3%**, outside
the ±10% band §6 called `RATE-HOLDS` and far above the `RATE-DEGRADES` line of 24.9.
**§6 has no bucket for "faster than the ledger".** That is a hole in my decision rule and it is
reported as such, not resolved by picking the nearest label.

The direction is monotone and the mechanism is visible in the same table: weight-organ bandwidth
rises 14.7 → 17.0 GB/s as matrices get bigger, which is per-call overhead being amortised —
consistent with the 2.5–3.2 µs OpenMP region the `--fuse` matrix measured, over `7 × L` matvec calls
per token. The extrapolation that prices this programme is **conservative on the weight term**, by
18%. It is the `f` term, not the rate, that was optimistic.

### 4.3 `INDEX` §7's 522 M charges the fixed cost twice

`SPEED_LEDGER` §12.2 computes `Delivered: 493,961,216 / 17.833 ms = 27.7 G-weights/s`. **17.833 ms is
the wall**, so `f` is already inside that denominator. It then writes
`27.7 G-w/s × (20 − 1.17) ms = 522 M`, subtracting `f` a second time.

Each convention has exactly one self-consistent form, and they agree:

| | |
|---|---|
| `r_wall × 20` = 27.70 × 20 | **554 M** |
| `r_w × (20 − f)` = 29.65 × 18.83 | **558 M** |
| `INDEX` §7 as written = 27.7 × (20 − 1.17) | 522 M |

So the published budget is ~6% **too tight** at 0.5 B — an error in the safe direction, which is why
nothing caught it. It does not survive contact with §4.1: the same formula with the *measured* `f`
moves in the opposite direction and much harder.

### 4.4 The budget re-derived from each shape's own `f`

Brief §6: *"the budget is re-derived from measured `f` at each shape"*. Using `r_w × (20 − f)`, the
form that separates the two terms:

| arm | budget @300 | budget @800 | `INDEX` §7 uses |
|---|---|---|---|
| `S05` | 554 M | 486 M | 522 M |
| `S15` | 556 M | 451 M | 522 M |
| `S3` | 513 M | 346 M | 522 M |
| `M7` | 435 M | 96 M | 522 M |
| `Q8` | 408 M | 25 M | 522 M |
| **`T10`** | **318 M** | **0 M** | 522 M |

At the goal's shape and the ledger's own context length the budget is **318 M active weights/token,
3.0% of a 10 B** — not 522 M and 5.2%. At 800 tokens of context it is **zero**: `f` alone exceeds the
whole 20 ms.

**`INDEX` §7's twelve-donor screen is superseded.** It was drawn with a budget that is 64% too
generous at the shape it was screening for, so every donor's "share of budget" is understated by
about 1.6× at 300 context, and the screen has no meaning at all at 800.

### 4.5 The head stops being the floor, and `qkv`/`o_proj` take over

Share of the token, `--bench 300`:

| arm | ffn | qkv_proj | o_proj | head | attention |
|---|---|---|---|---|---|
| `S05` | 64.7% | 4.9% | 3.4% | **20.5%** | 6.1% |
| `M7` | 77.5% | 10.7% | 7.0% | **1.7%** | 3.1% |
| `Q8` | 70.6% | 11.3% | 7.4% | **7.5%** | 3.2% |
| `T10` | 78.0% | 10.8% | 7.1% | **1.1%** | 3.0% |

Every probe in this programme was measured on Qwen at 0.5–1.5 B, where the head is **20.5%** of the
token and the obvious thing to attack. At the goal's shape it is **1.1%**, and the attention
projections together are **17.9%** — 16× the head. The head was the floor of a small-vocabulary-free
small model; it is not the floor of a 10 B.

**The matched pair does exactly what it was built to do.** `M7` and `Q8` are the same width, 4.6× apart
in vocabulary only:

| | head ms/token |
|---|---|
| `M7` (V = 32,768) | 3.721 |
| `Q8` (V = 151,936) | 17.225 |
| ratio | **4.629** |
| vocabulary ratio | **4.637** |

**0.2%.** `INDEX` §7's claim that *the tokenizer sets the head's floor* was an argument from weight
counts; it is now an end-to-end measurement, and the constant of proportionality is 1.

### 4.6 Attention: one constant explains all twelve points, and it is not bandwidth

`kcache` is `[L][maxseq][NKV×HD]` **fp32**, so there are two defensible denominators and both must be
shown (`feedback_charged_vs_moved_bytes`): the **unique** bytes the cache holds, and the **touched**
bytes the loop issues — the head loop runs over `NH`, and each GQA group of `NH/NKV` heads re-reads
the same slice.

| arm | GQA | KV MB/tok unique | KV MB/tok touched | attn GB/s **unique** | attn GB/s **touched** |
|---|---|---|---|---|---|
| `S05` | 7 | 3.7 | 25.9 | 3.4 | **23.6** |
| `S15` | 6 | 8.6 | 51.8 | 4.4 | **26.2** |
| `S3` | 8 | 11.1 | 88.8 | 3.1 | **25.0** |
| `M7` | 4 | 39.5 | 157.8 | 5.9 | **23.8** |
| `Q8` | 4 | 44.4 | 177.5 | 6.0 | **23.9** |
| `T10` | 4 | 59.2 | 236.7 | 6.0 | **24.0** |

The **unique** column spans 2×; the **touched** column is flat at 23.6–26.2 GB/s, and at `--bench 800`
it stays flat at 24.2–27.1. The re-reads are being paid for.

But touched bytes and scalar FMA count scale identically — both are `L × NH × HD × pos` — so that
table cannot by itself tell bandwidth from arithmetic. Counting the FMAs separates them:

| arm | bench | attn ms | FMA/token | FMA/s/thread | **cycles/FMA @4.2 GHz** |
|---|---|---|---|---|---|
| `S05` | 300 | 1.096 | 6.47 M | 0.98 G | **4.27** |
| `S15` | 300 | 1.974 | 12.95 M | 1.09 G | **3.84** |
| `S3` | 300 | 3.556 | 22.19 M | 1.04 G | **4.04** |
| `M7` | 300 | 6.633 | 39.45 M | 0.99 G | **4.24** |
| `Q8` | 300 | 7.420 | 44.38 M | 1.00 G | **4.21** |
| `T10` | 300 | 9.862 | 59.18 M | 1.00 G | **4.20** |
| `S05` | 800 | 2.847 | 17.23 M | 1.01 G | **4.17** |
| `S15` | 800 | 5.087 | 34.45 M | 1.13 G | **3.72** |
| `S3` | 800 | 8.936 | 59.06 M | 1.10 G | **3.81** |
| `M7` | 800 | 16.688 | 104.99 M | 1.05 G | **4.01** |
| `Q8` | 800 | 18.730 | 118.11 M | 1.05 G | **4.00** |
| `T10` | 800 | 25.385 | 157.48 M | 1.03 G | **4.06** |

**One FMA per ~4 cycles per thread, across twelve points, ±7%** — invariant to shape, context length,
KV size (16× range), GQA factor and vocabulary. That is the signature of a **serial dependency
chain**, not of a bandwidth limit: 24 GB/s is well under this machine's DRAM, and a bandwidth wall
would not produce a constant *per FMA*.

The source and the build flags corroborate it. The inner product is
`for(i<HD) d += qh[i]*kt[i]` — a floating-point **reduction**, and the engine is built
`clang -O3 -mavx2 -mfma -ffp-contract=on` with **no `-ffast-math`** (forbidden since Phase 35), so the
compiler may not reassociate it and cannot vectorise it. Every matvec in the engine is hand-written
AVX2 with an 8-wide accumulator; the attention loop is the one hot loop that is not.

**Marked as corroborated, not proven**, and the experiment that settles it is named: give the `K` loop
4–8 independent accumulators. If `f` drops by roughly the accumulator count, it was latency; if it
does not move, it was bandwidth and the lever is an int8 KV cache instead. **Either way `f` is the
term to attack, and it has never been touched** — the KV cache is fp32 and unquantized, and the GQA
re-read is structural.

### 4.7 The pre-stated predictions, including the wrong ones

Brief §5, written before the runner existed:

| arm | predicted tok/s | measured @300 | predicted `f` | measured `f` @300 |
|---|---|---|---|---|
| `S05` | 56.1 | 55.700 | 1.17 | 1.183 |
| `S15` | 17.9 | **19.390** | ~2.7 | **2.147** |
| `S3` | 9.0 | **9.930** | ~4.6 | **3.843** |
| `M7` | 3.9 | **4.610** | ~7.0 | 7.120 |
| `Q8` | 3.7 | **4.340** | ~7.9 | 7.962 |
| `T10` | **2.6** | **3.090** | ~9.0 | **10.576** |

**Every tok/s prediction was low**, by 8–19% above 0.5 B, for the reason §4.2 gives: I assumed the
27.7 G-w/s rate was flat and it rises to 32.8. **The `f` predictions were better than the reasoning
behind them**: I wrote *"KV bytes grow 16× from `S05` to `T10`, against a reservation that assumes it
stays constant"* and predicted ~9.0 ms; the measurement is 10.576. The number was close, the argument
was wrong — §4.6 shows the cost tracks the FMA chain, not the KV bytes, and those two happen to scale
together on this set of shapes.

The brief also said, in bold, *"I expect the 1.17 ms reservation to fail, and I expect it to fail
badly."* It failed by 9× at 300 and 22× at 800.

### 4.8 The goal, measured instead of extrapolated

Brief §5 stated as arithmetic before the run that a dense 10 B cannot reach 50 tok/s. It can now be
said with a measurement:

| | |
|---|---|
| `T10` measured | **3.090 tok/s** @300, **2.960** @800 |
| needed for 50 tok/s | 530 G-weights/s |
| measured | 32.8 G-weights/s (wall) |
| **short by** | **16.2×** |
| the same thing without any rate | `50 / 3.090` = **16.2×** |
| ceiling with a free weight path, @800 | **38.3 tok/s** |

The dense path is 16× away, and **the last 1.3× of that is unreachable by any weight-side work at
all** at 800 context. The active-weight budget that a sparse/MoE architecture must hit is **318 M at
300 context — 3.0% of a 10 B — and at 800 context there is no budget until `f` is fixed first.**

---

## 5. The label

Brief §6, verbatim, with the measured inputs:

| outcome | condition | measured | fires? |
|---|---|---|---|
| `VOID` | Gate V1, V2 or V3 fails | run 1: V1 fired, V2 failed | **YES for run 1** |
| `RESERVATION-HOLDS` | `f ≤ 2.0 ms` at `T10`, bench 300 | `f` = 10.576 | no |
| **`RESERVATION-BREAKS`** | `f > 2.0 ms` at `T10` | **10.576 @300, 26.088 @800** | **YES** |
| `RATE-HOLDS` | `r` at `T10` within ±10% of 27.7 | 32.76 (wall) | no |
| `RATE-DEGRADES` | `r` at `T10` below 24.9 | 32.76 | no |

**Run 1: `VOID`.** **Run 2: `RESERVATION-BREAKS`**, and the rate outcome is **outside every bucket
§6 defined** — `T10` delivers 32.76 G-w/s, 18.3% *above* the band. Reported as an undefined outcome
rather than rounded into `RATE-HOLDS`.

Consequences, per §6: the budget is re-derived per shape (§4.4) and **`INDEX` §7's twelve-donor
screen is marked superseded**.

---

## 6. Departures from the brief

1. **Run 1 was `VOID`** and its two failing gates are reported in full (§2.2, §2.3) rather than
   re-run silently. No arm above `S05` existed when run 1 was labelled.
2. **Gate V1 was replaced, not reinterpreted.** V1′ (§2.4) is a different experiment at a different
   scale, fixed in the brief before its file existed.
3. **Gate V2′'s hard constant was retired after seeing it fail**, on the ground that the real
   artifact that produced the constant also fails it (§2.5). This is the one place in E3 where a
   criterion moved after data was seen, and §2.5 says so in those words.
4. **§6's rate rule is incomplete.** It has `RATE-HOLDS` and `RATE-DEGRADES` and no bucket for a
   measured rate *above* the band, which is what happened (§4.2).
5. **§6's rate constant 27.7 is `r_wall`,** so it is judged in that convention (§4.2). The
   weight-organ convention `r_w` is reported beside it and never substituted for it.
6. `synth_export.py` gained `--head {ternary,donor}` between runs 1 and 2; `donor` reproduces run 1
   exactly, so run 1 remains replayable.
7. **Corrected after first publication, 2026-09-06.** §0 and §4.8 first said **15.7×**. That was a
   denominator swap of mine, and exactly the one §4.3 sanctions in `INDEX` §7: `530 G-weights/s` is
   `active / 20 ms of **wall**`, so it must be divided by `r_wall` (32.76), not by `r_w` (33.78).
   The correct figure is **16.18×**, and the cross-check that cannot be got wrong — `50 / 3.090` —
   gives the same 16.18. `e3_analyse.py` now computes it from `rate_wall` and prints the
   rate-free form beside it, so the swap cannot be made again silently. Nothing else moves: the
   budget, `f`, and the ceilings never went through this quantity.
8. **Not measured, and not claimed:** whether the GQA re-reads reach DRAM or hit L2; whether the
   attention loop is latency- or bandwidth-bound (§4.6 gives the discriminating experiment); anything
   at all about quality — the weights are noise.

---

## 7. What this closes and what it owes

**Closes.** The ledger's 1.17 ms reservation, tested at 20× its measured scale and broken by 9–22×.
The kernel's value-independence, measured rather than read (§2.2). The tokenizer-sets-the-head claim,
end-to-end at 8 B (§4.5). The 2 GB loader limit, at 6.28 GB (§2.1).

**Owes.**
1. `INDEX` §7's twelve-donor screen, re-run against the per-shape budgets in §4.4.
2. `SPEED_LEDGER` §12.2's 522 M, corrected for the double charge (§4.3).
3. The accumulator experiment of §4.6 — the one change that could move `f`.
4. An int8 KV cache, if §4.6 comes back bandwidth-bound.
5. Everything about `f` at contexts beyond 800; `f` grows with position and nothing here bounds it.

---

## 8. Reproduction

```
cd benchmarks/donor_adaptation/engine
python synth_export.py --shape T10 --out D:/_ktmp/e3/T10_th.bin --head ternary --codes mixed
python e3_bench.py --weights D:/_ktmp/e3/T10_th.bin --bench 300 --reps 3 --label T10_b300
bash e3_run_arms.sh          # the whole sweep, sequential
python e3_analyse.py         # every table in section 3 and section 4
```

Gate V1's pair: `--codes zero` and `--codes dense` at `--shape S05`.
Gate V1′: `--shape S15 --head ternary` against `D:/_ktmp/e2/qwen25-15b_tqh.bin`.
Gate V2′: interleave `D:/_ktmp/e2/qwen25-05b_tqh.bin` with `S05 --head ternary`, one rep each,
alternating — never two blocks twenty minutes apart.
