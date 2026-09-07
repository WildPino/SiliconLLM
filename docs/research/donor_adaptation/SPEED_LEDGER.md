# The Speed Ledger — what a donor must cost per token for engine.c to hit 50 / 100 tok/s

**Date: 2026-09-04. Author: the Adapter / Principal. Status: ARITHMETIC, NOT MEASUREMENT.**
**AMENDED 2026-09-04 by `probes/P2_EXPERT_PATH_DECOMPOSITION.md`: §4's bracket is CLOSED and §3's
optimistic headline is WITHDRAWN. The amendments are inline below, marked. Nothing has been deleted.**
**AMENDED AGAIN by `probes/P3_DONOR_SHAPE_ON_ENGINE.md`: this ledger prices DENSE FFNs at the
expert path's i.i.d. GATHER rate (17.0 GB/s), which is wrong — a dense donor FFN is read
contiguously. P3 drove the donor's real shapes through the engine's real kernels and measured
Qwen2.5-1.5B at 30.7 tok/s (t6, matvec only), i.e. this ledger is ~18% CONSERVATIVE on dense
donors. A carved MoE donor does pay the gather rate; a dense one does not, and the two must not
share a row here. Every "dense tok/s" figure below is a floor, not an estimate.**
**Generator: `benchmarks/donor_adaptation/speed/donor_speed_budget.py` (`--show-sources` prints the
provenance of every constant). Raw: `speed/donor_speed_budget_ctx4096.json`.**

> **BAND ON EVERY ABSOLUTE tok/s IN THIS FILE: ±5%** (§14.3, the law E4 closed with).
> Between-sweep dispersion on this machine is 5–10%; the within-run IQRs printed in the tables
> below are 0.000–0.015 tok/s and **do not show it**. Every absolute tok/s, G-weights/s and
> active-weight budget here carries that band, wherever it appears and whether or not the band is
> restated at that line. **Ratios taken inside one sweep do not** — that is why the conclusions of
> §12–§14 are all ratios. A number from this file may not be compared to one measured in another
> session unless the reference is re-measured alongside it.

> **Not audited.** Written by the figure that built the generator. A Controller pass is owed.

---

## 0. Why this document exists

The goal names a **speed**: a large model (~10B) at **50 tok/s** to be good, **100 tok/s** to be
excellent. Every donor-adaptation probe this programme has run measures **quality damage** — D0/D0c
the BPB cost of carving, D1 of pruning, D4 of reconstruction, S1 of sparsifying. Not one of them
computes what the speed target actually *demands*, so not one of them can say whether the damage it
measured buys anything.

This ledger supplies the missing half. It converts the target into the only currency that unifies
bandwidth and kernel throughput — **active weights touched per token** — and prices every donor
sitting on this machine's disk against it.

**It is arithmetic over measured rates, not a measurement.** Its rates come from
`docs/PHASE64_BUDGET.md` §1 and §1b (the 64.0 tables at `be5f448`, the 64.1b microbenches at
`f4a53cf`), and its donor shapes are read from the `config.json` files actually present in the local
HF cache. Nothing in it was recalled from memory.

**The single largest source of error, stated up front: `engine.c` has never executed a transformer.**
It is an SSM engine; attention is an organ it does not have. Attention projections here are priced
**by analogy** to the measured proj-GEMV path. That analogy is untested and it is load-bearing.

## 1. The budget

One token at 50 tok/s is 20.0 ms; at 100 tok/s, 10.0 ms. The engine's measured delivery, converted at
the 0.5 B/weight the engine emits today:

| path | measured | in weights | what it is |
|---|---|---|---|
| expert path, **as integrated today** | 4.2 GB/s | 8.4 G-w/s | `PHASE64_BUDGET.md` §1. Overhead-bound, **not** bandwidth-bound |
| expert path, **kernel-pure ceiling** | 17.0 GB/s | 34.0 G-w/s | §1b(b). The same kernel with ~8.4 µs/expert of dispatch/gather/dequant/combine removed. The integrated engine is **3.9× away from it** |
| proj-GEMV, streamed floor | 37.0 GB/s | 74.0 G-w/s | §1b(a), t6 asymptote 34–36, declared floor [34–40] |
| DRAM aggregate ceiling | 42.0 GB/s | — | §1, saturated at 3 threads. Physics |

So the budget, at today's packing and the kernel-pure ceiling:

> **50 tok/s → ≤ 680 M active weights per token. 100 tok/s → ≤ 340 M.**
> At the rate the engine *actually* delivers today, those become **168 M** and **84 M**.

## 2. The four terms, and which ones this programme has attacked

Per generated token, four things move:

| term | scales with | attacked by this programme? |
|---|---|---|
| **FFN / expert weights** | activation fraction | **yes — D0, D0c, D1, D4, S1, probe-2, probe-4.** Everything |
| **attention projections** (q,k,v,o) | nothing; dense every token | **no. Never** |
| **output head** (`d_model × vocab`) | nothing; dense every token | **no. Never** |
| **KV cache** | context length | **no. Never** |

For Qwen2.5-1.5B — the donor every result in this programme is measured on:

    attention projections   154.1 M
    output head             233.4 M      <- LARGER than all attention projections combined
    FFN                    1156.1 M
    KV cache @ ctx 4096     117 MB/token

**The output head is 233.4 M weights, 51% more than the entire attention stack, and nothing in this
programme has ever touched it.** At a 25% FFN carve it becomes **34.5% of the whole per-token
budget**; at a 6.25% carve, **50.8%**.

## 3. The result, on the donor this programme actually studies

Split-path model: FFN on the expert path (17.0 GB/s kernel-pure), attention + head on the proj path
(37.0 GB/s streamed floor), KV on DRAM aggregate (42 GB/s). ctx = 4096.

| packing | attn+head | FFN (dense) | KV | **dense tok/s** | FFN activation for 50 t/s | for 100 t/s |
|---|---|---|---|---|---|---|
| **4-bit (what the engine emits today)** | 5.24 ms | 34.00 ms | 2.80 ms | **23.8** | **≤ 35.2%** | ≤ 5.8% |
| **1.6-bit (queued at E4, never built)** | 2.09 ms | 13.60 ms | 2.80 ms | **54.1** | *already fits dense* | ≤ 37.6% |

~~Read the second row. **With the denser pack and the expert-path overhead fixed, Qwen2.5-1.5B reaches
50 tok/s with no carve at all — no quality damage of any kind.** And 100 tok/s needs only ≤ 37.6% FFN
activation.~~

> ### ⛔ WITHDRAWN 2026-09-04 — P2 measured the assumption and it was wrong
>
> The 1.6-bit row above assumed a denser pack shrinks the FFN term by the full 2.5×.
> **`probes/P2_EXPERT_PATH_DECOMPOSITION.md` measured the expert path's arithmetic floor directly and
> found only ~40% of it is memory** (outcome MIXED, `c/r = 0.593`, five repeats, ordering and
> planted-negative controls both passed). A denser pack therefore buys **1.32× on the FFN, not 2.5×**.
> Corrected table:
>
> | packing | attn+head | FFN dense | KV | **dense tok/s** | FFN act. for 50 t/s |
> |---|---|---|---|---|---|
> | 4-bit | 5.24 ms | 33.60 ms | 2.80 ms | **24.0** | ≤ 35.6% |
> | 1.6-bit | 2.09 ms | 25.39 ms | 2.80 ms | **33.0** | ≤ 59.5% |
> | 1.6-bit, conservative | 5.24 ms | 25.39 ms | 2.80 ms | **29.9** | ≤ 47.1% |
>
> **Qwen2.5-1.5B does NOT reach 50 tok/s dense. It reaches 30-33 tok/s, and the target needs
> sparsity after all** — but at **35.6%-59.5% FFN activation**, not the 25% D0/D0c carved at. What the
> carve costs at 40-60% activation has never been measured, and that is now the obvious next quality
> probe.

Against that, what the carve costs: D0c measured **+0.70611 BPB = 141 σ_seed** at 25% activation, at
the finest *legal* granularity, **with an oracle router** — on a baseline of 0.7676, i.e. nearly
doubling bits per byte. The comparison is not close.

> **The cheapest route to the speed target on this donor is not a better carve** — but after P2 it is
> not the packing alone either. The standing order is: the **~8.4 µs/expert integrated overhead**
> (3.9×, and untouched by P2's decomposition) first; the denser pack second, now priced at **1.32× on
> the FFN** rather than 2.5×; and then a carve at the **35-60% activation the target actually asks
> for**, which is 1.4×-2.4× gentler than the 25% every carve result in this programme was measured at.

## 4. ⚠ The bracket I cannot close, and it is wide

Row 2 above divides a **byte** rate by fewer bytes per weight. That is only valid if the expert path
is **bandwidth-bound** — and `PHASE64_BUDGET.md` §1b(b) says in terms that it is **not**: it is
"gather/kernel-bound", at 2.88 µs/expert. If the cost per expert is set by the gather and not by the
bytes, a denser pack buys **nothing at all** on the FFN.

Both readings, on Qwen2.5-1.5B at 1.6-bit:

| assumption about the expert path | dense tok/s |
|---|---|
| bandwidth-bound (denser pack helps the FFN) | **54.1** |
| gather-bound (denser pack helps attn+head and footprint only) | **25.7** |

**That is a 2.1× bracket and it decides whether the goal is close or far.**

> ### ✅ CLOSED 2026-09-04 — `probes/P2_EXPERT_PATH_DECOMPOSITION.md`
>
> Measured, not argued. Three arms over the identical kernel, varying only which expert each touch
> reads: i.i.d. random (512 MB pool), sequential, and **constant expert 0** — the last one L1-resident,
> so its µs/expert is the arithmetic floor with memory removed.
>
> **At t6: r = 2.857 µs, c = 1.693 µs → the memory term is 1.164 µs = 40.7% of the path.**
> Outcome **MIXED** in all five repeats (`c/r` 0.490-0.632); neither pre-registered edge approached.
> A cross-check nobody arranged: the isolated memory term divided by its bytes gives **42.2 GB/s**,
> landing on the DRAM aggregate ceiling of 40-44 GB/s that §1 registers from a separate bench months
> earlier.
>
> **The bracket collapses from 25.7-54.1 tok/s (2.10×) to 29.9-33.0 (1.10×), toward the pessimistic
> edge.** The 1.6-bit column in the generator remains an **upper bound**, now a quantified one.
>
> **Still owed:** the same decomposition on the **proj** path. P2 split the expert path only, and §3's
> attention/head figures still assume denser packing helps them in full. `PHASE64_BUDGET.md` §1 says
> the fp32 projections run at aggregate bandwidth when threaded, which is evidence — but it was
> measured on fp32, and P2's whole result is that a kernel can have a large compute floor at small
> byte counts.

## 5. The donor menu — and the thing that was hiding in the cache

Split-path, ctx 4096, 4-bit, kernel-pure. `FFN-carve ceiling` = the speedup from deleting the
**entire** feed-forward stack, i.e. the hard limit on everything the FFN probes can ever deliver.

| donor | total | active/token | act% | **tok/s** | FFN-carve ceiling |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 494 M | 494 M | 100% | 77.7 | 3.54× |
| Qwen2.5-1.5B | 1.54 G | 1.54 G | 100% | 23.8 | 5.23× |
| **OLMoE-1B-7B-0924** | **6.92 G** | **1.18 G** | **17.0%** | **24.1** | 2.33× |
| Qwen2.5-Coder-7B | 7.62 G | 7.07 G | 92.8% | 5.2 | 7.97× |
| Qwen3-8B | 8.19 G | 7.57 G | 92.4% | 4.9 | 4.70× |
| **gpt-oss-20b** | **20.9 G** | **3.61 G** | **17.3%** | **11.2** | 4.71× |
| **Qwen3-30B-A3B** | **30.5 G** | **3.04 G** | **10.0%** | **12.6** | 3.03× |
| Mixtral-8x7B | 46.7 G | 12.75 G | 27.3% | 2.7 | 11.14× |

**Three donors already on this disk are pre-trained MoE with a router their own authors trained.**
`Qwen3-30B-A3B` runs at **10.0% activation by construction**; `Qwen3-Next-80B-A3B` at **3.5%**.

This is the finding that most deserves acting on. The programme has spent its effort trying to
**carve** an already-dense FFN into experts — and D0c has now measured that carve at +0.706 BPB with
an oracle router, with the co-activation advantage *shrinking* as granularity improves. Meanwhile a
donor that is **already carved, by its trainer, with a trained router and no reconstruction loss at
all**, was sitting in `~/.cache/huggingface`. Qwen3-30B-A3B delivers 30 B of capacity at 3.04 G active
weights: a **10× capacity-to-active ratio that no carve in this programme has come close to buying.**

It is not free — 3.04 G active is 4.5× over the 680 M budget for 50 tok/s at 4-bit, and its attention
+ head alone (1.22 G) already exceed that budget. But it starts from a place no carve has reached.

## 6. What this ledger says to do, in order

1. **Close the §4 bracket.** One microbench, existing harness, no training. It moves every number in
   this document by up to 2.1× and it decides whether the packing work is worth doing.
2. **Build the 1.6-bit pack** if §4 says the expert path is bandwidth-bound — and build it anyway for
   the attention, head and footprint terms, where the gain is unambiguous. Queued since E4, never
   started, worth 2.5× on every byte.
3. **Attack the ~8.4 µs/expert overhead.** 64.1b already decomposed it (index gather / dispatch /
   dequant / dReLU / combine) and called it "engineering-addressable, not a bandwidth wall". It is
   worth **3.9×** and it is the single largest multiplier in this document.
4. **Price an already-MoE donor end-to-end** — Qwen3-30B-A3B or OLMoE-1B-7B — instead of carving a
   dense one. Its expert width (768 at `d_model` 2048) clears the 48 KiB per-organ floor
   (`49152/(2048×0.5) = 48` neurons) with 16 blocks to spare, so it is engine-legal as it ships.
5. **Open the head and the KV cache**, which together are 34.5% and 12% of the 1.5B carved budget and
   have never been probed. Nothing in the closed negatives touches either.

## 7. What this document does not claim

- It does **not** claim any donor runs on `engine.c`. Nothing here has been executed.
- It does **not** predict quality. Every activation fraction quoted is a **speed** requirement; what
  it costs in BPB is the probes' question, and where the two meet is §3.
- It does **not** model shared experts, first-k-dense layers, or hybrid SSM/attention stacks. Donors
  needing those are **flagged and excluded** by the generator rather than approximated —
  DeepSeek-V2-Lite, granite-4.0-h-small, Qwen3-Next-80B-A3B, Nemotron-H, Zamba2, Falcon-H1.
- It does **not** model prefill, sampling, tokenisation, or the n-gram/block-decode chassis (E5), any
  of which changes the picture.
- Its attention pricing is **by analogy** (§0) and its 17.0 GB/s expert figure is a **ceiling the
  integrated engine is 3.9× short of** — so the "tok/s" columns are what the engine could deliver
  after work that has not been done, not what it delivers.

## 8. The self-check, because a parameter count is easy to get plausibly wrong

The generator carries a planted control: it compares each computed parameter total against the size
**the publisher put in the model's own name** — an independent statement written by someone not doing
this arithmetic. **On its first run the control fired and caught a real bug**: the expert width lives
in `moe_intermediate_size` for Qwen3 but in `intermediate_size` for Mixtral, OLMoE and gpt-oss, and
reading the wrong field had silently undercounted OLMoE as 575 M instead of 6.92 G. It also caught a
second: assuming a 3-matrix gated FFN everywhere overcounted `starcoder2-3b` by 44%, which uses a
2-matrix non-gated MLP.

Current state: **20 agree, 2 disagree, 1 documented exception.** The two failures are `Zamba2-2.7B`
and `Nemotron-H-8B`, both already independently flagged as hybrid SSM — the two controls agree with
each other. The exception is `Mixtral-8x7B`, where the **name** is wrong, not the arithmetic: "8x7B"
would be 56 B, but Mixtral shares attention across experts and is 46.7 B in fact.

A row that fails the control is **excluded from every table**, not merely annotated.

## 9. Reproduce it

```
python benchmarks/donor_adaptation/speed/donor_speed_budget.py --ctx 4096
python benchmarks/donor_adaptation/speed/donor_speed_budget.py --show-sources
```

Reads only `config.json` files from the local HF cache and the constants in §1. No model weights are
loaded, nothing is downloaded, and it runs in under a second.

---

## 10. AMENDED 2026-09-04 — the requirement, from the runtime that exists

Sections 1–9 price the goal from *component* rates. `probes/R1_DONOR_RUNTIME.md` now supplies the
only number that matters: what a **real transformer runtime** actually delivers, end to end,
including everything the component model excluded.

**Measured** (Qwen2.5-0.5B, `donor_engine.c`, t6, packed 2 trits/byte, ternary head, ctx ≤200):
494.0 M active weights/token in 27.72 ms wall → **36.1 tok/s**, of which 26.59 ms is weight-matvec.

*(±5% band, §14.3 — this is an absolute tok/s, and so is the 18.6 G-weights/s below it.)*

> ### **DELIVERED RATE = 18.6 G-weights/s.**

Everything about the goal follows from that one number:

| target | max active weights/token | on a 10B donor that is |
|---|---|---|
| **50 tok/s** | **350 M** | **3.50% activation** |
| **100 tok/s** | **165 M** | **1.65% activation** |

**Qwen2.5-0.5B itself carries 494 M active weights — already over the 50 tok/s budget.** That is
the whole explanation of why the smallest donor available runs at 36 and not 50, and it is not a
statement about 0.5B: it is a statement about the kernel.

### If the LUT kernel lands

`PHASE64_BUDGET.md` §1b(b) measures the kernel-pure expert ceiling at 17.0 GB/s = **34 G-weights/s**
at 0.5 B/weight — **1.83× what this runtime delivers today**. `donor_engine.c` does not use that
path: it converts int8 codes to float and does FMAs, where the LUT path does table lookups and adds.
At that ceiling:

| target | max active weights/token | on a 10B donor |
|---|---|---|
| 50 tok/s | **641 M** | **6.41% activation** |
| 100 tok/s | 301 M | 3.01% activation |

### What that means against the donors that actually exist

| donor | active weights/token | vs the 641 M budget |
|---|---|---|
| Qwen3-Next-80B-A3B | 2.78 G | **4.3× over** |
| Qwen3-30B-A3B | 3.04 G | 4.7× over |
| OLMoE-1B-7B | 1.18 G | 1.8× over |
| Qwen2.5-0.5B | 0.49 G | fits, with margin |

> **The goal as stated — ~10B at 50 tok/s — needs a donor with roughly 640 M active weights per
> token, i.e. ~6.4% activation at 10B. Nothing shipping at that size is that sparse**, and the two
> A3B models that come closest in *fraction* are 3–8× too heavy in absolute active weights because
> they are 30 B and 80 B rather than 10 B.

**Three honest routes out, and they are not exclusive:**
1. **The LUT kernel** — 1.83× measured-ceiling headroom, unclaimed, and the only pure-engineering
   one on this list.
2. **A donor at ~10 B with ≤6.4% activation.** That is a search over what exists, not a technique.
3. **Sparsify a donor further ourselves** — which is what D0/D0c/D1/S1 were doing, and which T1
   says must now be composed with a conversion nobody has composed it with.

**Caveat that applies to every row:** the 3600X is this project's *reference floor*, not its target
(portability law, `ENGINE_PLAN.md`). These are per-machine numbers and a wider machine moves them.

---

# 11. The LUT kernel, BUILT and MEASURED — and §10's rate withdrawn

**Date: 2026-09-04. `donor_engine.c` @ `e02285c`, Qwen2.5-0.5B packed + ternary head, 3600X, t6.**
**This section supersedes §10's rate and closes §10's route 1.**

## 11.1 §10's number does not reproduce

§10 recorded **27.72 ms wall / 26.59 ms matvec → 36.1 tok/s → 18.6 G-weights/s**, and that constant
was propagated into `donor_speed_budget.py` and `INDEX.md`. Re-measured on an idle machine, three
consecutive repetitions of `--bench 300`:

| | rep 1 | rep 2 | rep 3 |
|---|---|---|---|
| packed, tok/s | 48.24 | 48.28 | 48.33 |

and with `--profile`, wall **20.7–21.0 ms/token**, of which weight-matvec (`qkv + o_proj + ffn +
head`) is **19.69 ms** and the fixed remainder (attention, RoPE, softmax, KV, norms, residuals) is
**1.20 ms**.

> **DELIVERED RATE = 493,961,216 / 19.691 ms = 25.1 G-weights/s.**
> §10's 18.6 G-weights/s was almost certainly taken on a contended machine — this session ran a
> 40 GB, 6-thread probe for 48 minutes — and **a contended timing is not a timing.** §10's rate is
> withdrawn; the spread between the two is 35%, which is larger than any lever measured below.

## 11.2 The LUT kernel buys 1.04×, not 1.83×

`--lut` is probe-1's `pshufb`-LUT: tile-major codes, one 16-byte table per input pair, `shuffle`
plus integer add instead of convert plus FMA. It is **bit-exact** against a scalar-integer
reference and its two planted controls fire (`--selftest-lut`). Back to back, same binary:

| config | tok/s | matvec ms | delivered | vs packed |
|---|---|---|---|---|
| packed | 48.13 | 19.69 | 25.1 G-w/s | — |
| **`--lut`** | **51.05** | **18.75** | **26.3 G-w/s** | **1.05×** |
| `--lut --lut-group 32` | 50.96 | 18.81 | 26.3 G-w/s | 1.05× |

Per organ: ffn 12.38 → 11.56 (1.07×), head 3.87 → 3.29 (1.18×), and **qkv got slower**, 3.02 → 3.18.

**§10's "IF the LUT kernel reached PHASE64's kernel-pure ceiling (34 G-weights/s, ×1.83)" is
withdrawn as a measured fact.** The kernel is built; it reaches 26.3, not 34. On an operation
count the LUT path issues ~0.20 vector ops per weight against the packed path's ~0.56 — it should
have been ~2.8× — and it delivers 1.05×, which puts the real bound somewhere neither instruction
count nor DRAM bandwidth explains (247 MB/token at 19.7 ms is 12.6 GB/s, a third of this machine's
measured DRAM aggregate).

**This is the third time this programme has watched a microbenchmark ceiling fail to compose.**
`PHASE61_PROJQUANT` ("microbench compute-bound does not compose to an engine that is memory-bound"),
then P2 → R1 on the packing (predicted ≤1.32×, delivered ~1.0×), now P64's kernel-pure ceiling
(predicted 1.83×, delivered 1.05×). **The law is earning its keep: a kernel ceiling is not a
runtime speedup until it is measured inside the runtime.**

## 11.3 The budget, restated in measured units

Fixed non-matvec cost 1.20 ms/token (measured at 0.5B and short context; it grows with layers and
with context, so these are optimistic for a 10B).

| target | at 25.1 G-w/s (packed) | at 26.3 G-w/s (LUT, built) |
|---|---|---|
| **50 tok/s** | **472 M active weights/token** = 4.7% of a 10B | **495 M** = 5.0% |
| **100 tok/s** | **221 M** = 2.2% of a 10B | **232 M** = 2.3% |

Against the donors on disk, at the 495 M figure: Qwen3-30B-A3B (3.04 G active) is **6.1× over**,
Qwen3-Next-80B-A3B (2.78 G) is 5.6× over, OLMoE-1B-7B (1.18 G) is 2.4× over. Qwen2.5-0.5B, at
0.494 G, is the only thing on this machine that fits — and it is 20× smaller than the target.

**What §10's route 1 was, and what replaces it.** Route 1 was "the LUT kernel, 1.83× unclaimed".
It is now claimed and worth 1.05×, so **the engineering headroom this ledger was counting on does
not exist**. The remaining routes are unchanged and now carry the whole weight: a donor that is
genuinely ~10 B with ≤5% activation, or sparsifying one ourselves — composed, as T1 and T2 insist,
with a conversion that still costs +1.260 BPB on the FFN alone.

**And the term that no route touches:** §7 of `INDEX.md` — the output head is a dense `D × V` GEMV
on every token, and on a Qwen-family 10 B it is **622 M weights, larger than the entire 495 M
budget**. A donor with a 32 K vocabulary carries 134 M for the same width. On this arithmetic
**the tokenizer is a harder speed constraint than the kernel**, and it is chosen, not earned.

## 11.4 Per-organ delivered rate — and a correction to §11.2's commit message

> ## ⚠ §11.4 IS SUPERSEDED BY §12. Read that first.
> The `qkv` row below is an artefact: both `rope()` calls sat inside the `T_QKV` timer, so this
> table charged 1.89 ms of double-precision `pow()` to qkv's weights. **qkv is 14.2 GB/s, not
> 4.1.** The "32 µs per call → threads are being woken" inference and the "1.37×" ceiling that
> follow from that row are **withdrawn**. The table is kept as written because two experiments
> were pointed at it.


The aggregate 25.1 G-weights/s hides a 4× spread. Same profile, same run, weights charged at
the 0.5 B/weight the engine actually emits:

| organ | weights/token | MB/token | ms | **GB/s** | matvec calls/token |
|---|---|---|---|---|---|
| **qkv** | 24,772,608 | 12.4 | 3.022 | **4.1** | 72 (24 × q,k,v) |
| o_proj | 19,267,584 | 9.6 | 0.674 | 14.3 | 24 |
| ffn | 313,786,368 | 156.9 | 12.380 | 12.7 | 72 (24 × gate,up,down) |
| **head** | 136,134,656 | 68.1 | 3.866 | **17.6** | 1 |
| **total** | 493,961,216 | 247.0 | 19.942 | **12.4** | 169 |

> **CORRECTION.** The commit message for `--fuse` (`259e147`) states the head runs at
> **35 GB/s, "at the machine's DRAM streaming rate"**. That is wrong: it charged the head's
> *weight count* as megabytes. At 0.5 B/weight the head moves 68.1 MB, not 136, so the rate is
> **17.6 GB/s**. The conclusion moves in the useful direction — **nothing here is at the
> bandwidth wall**, DRAM aggregate on this machine is 40–44 GB/s and probe-3's post-L3-cliff
> figure is ~28, so the headroom is real rather than closed.

**Three bounds, and the measurement sits far from all of them.** DRAM 40–44 GB/s: we are at
12.4. Instruction issue: the packed kernel spends ~0.56 vector ops per weight, so 494 M weights
is ~277 M ops, and six cores at ~4 ops/cycle × 3.6 GHz would retire that in ~1.2 ms against the
measured 19.9. Denser packing: `ternary_1p6bit` would cut bytes 2.5×, and **on a path that is
nowhere near its bandwidth wall that buys nothing** — which is P2's conclusion, now generalised
from the expert path to every organ in this runtime.

**What the table does say is that the rate tracks how the work is CHOPPED.** The head is one
matvec per token and is the fastest organ. `qkv` is three per layer, two of them 128 output rows
— across six threads that is 21 rows each — and it is 4.3× slower than the head on the same
kernel. The excess is 2.32 ms over 72 calls, **32 µs per call**, which is an order of magnitude
more than an OpenMP barrier costs and points at threads being woken rather than merely
synchronised.

**Ceiling if every organ merely reached the head's own 17.6 GB/s: 14.03 ms of matvec, 15.23 ms
wall, 65.7 tok/s — 1.37×.** That is larger than the LUT kernel delivered, it needs no numeric
change at all, and `--fuse` plus `OMP_WAIT_POLICY` are the two experiments aimed at it
(`bench_matrix.sh`, crossed, three reps, idle machine only).

---

## 12. AMENDED 2026-09-05 — the profiler was hiding 9.6% of every token, and manufacturing an anomaly

`donor_engine.c`'s `rope()` computed `pow()` — double precision — plus `cosf()` and `sinf()` for
**every element of every head of every layer**. The angle depends only on `(pos, j)`: not on the
head, not on the layer. On this donor that is `(14 + 2) heads × 24 layers × 32 elements = 12,288`
double `pow()` calls per token, to produce **32 distinct values**.

Hoisted to one table per token. **Bit-identical**: `--logits` over 64 tokens at positions 0–63,
byte-for-byte equal before and after, 64 × 151,936 floats. `--selftest-lut` still passes all three
cases including both planted controls.

| | ms/token |
|---|---|
| rope, before | **1.890** (9.6% of the whole runtime) |
| rope, after | **0.016** |

**50.9 → 56.1 tok/s, +10.3%, at zero numeric cost.** Commit `4f7b33c`.

*(±5% band, §14.3, on both absolutes; the +10.3% was taken inside one sweep and is not banded.)*

### 12.1 What it invalidates

Both `rope()` calls were **inside the `T_QKV` timer**. §11.4 therefore charged rope's
transcendentals to qkv's weights and reported qkv at **4.1 GB/s against the head's 17.6** — a 4×
anomaly that does not exist. Withdrawn, explicitly:

| §11.4 said | actually |
|---|---|
| qkv delivers **4.1 GB/s** | **14.2 GB/s**, in line with every other organ |
| "the excess is 2.32 ms over 72 calls = **32 µs per call**, an order of magnitude more than an OpenMP barrier → threads are being **woken**" | there is no excess. Independently falsified the same afternoon: `--fuse` removes 48 of those 72 calls and saves 0.146 ms = **3.0 µs per call eliminated**, which is barrier scale |
| "ceiling if every organ hit 17.6 GB/s: **65.7 tok/s (1.37×)**" | **68.8 tok/s (1.23×)** against the head's re-measured 18.5 GB/s |

This is the same failure the ledger has now made twice in two days: **§11 withdrew a rate because
the machine was contended; §12 withdraws a rate because the timer bracketed the wrong work.** A
number is only as good as what its denominator and its bracket actually contain.

### 12.2 Where the runtime is, restated

Qwen2.5-0.5B, 6 threads, `--quant packed`, `--bench 300`, idle machine, median of three
(56.41 / 55.75 / 56.14 tok/s):

| organ | weights/token | MB/token | ms | **GB/s** |
|---|---|---|---|---|
| qkv | 24,772,608 | 12.4 | 0.872 | **14.2** |
| rope | — | — | 0.016 | — |
| attention (KV cache) | — | — | 1.092 | — |
| o_proj | 19,267,584 | 9.6 | 0.601 | 16.0 |
| ffn | 313,786,368 | 156.9 | 11.513 | 13.6 |
| head | 136,134,656 | 68.1 | 3.673 | **18.5** |
| norm + glue | — | — | 0.068 | — |
| **weights only** | **493,961,216** | **247.0** | **16.659** | **14.8** |
| **total** | | | **17.833** | |

**Delivered: 493,961,216 / 17.833 ms = 27.7 G-weights/s** (was 25.1 in §11.3).

| target on a 10B donor | active weights/token | share of 10B |
|---|---|---|
| ~~**50 tok/s**~~ | ~~**522 M**~~ | ~~**5.2%**~~ |
| ~~100 tok/s~~ | ~~245 M~~ | ~~2.4%~~ |

> ⚠ **SUPERSEDED BY §13.** Both rows are wrong twice: the arithmetic charges the fixed cost twice
> (§13.2), and the 1.17 ms reservation below is **10.576 ms** at a 10 B shape (§13.1). Measured:
> **318 M at 300 tokens of context, and no budget at all at 800.** 100 tok/s is unreachable at a 10 B
> shape at any weight cost, because `f` alone is 10.576 ms > 10.

> Those are **weight budgets after reserving the fixed 1.17 ms/token** (rope + attention + norm),
> which is the convention §11.3 used and `donor_speed_budget.py` computes: `27.7 G-w/s ×
> (20 − 1.17) ms = 522 M`, not `27.7 G-w/s / 50 = 554 M`. The 1.17 ms is measured at 0.5 B and
> short context and is **optimistic for a 10 B**, where attention and the KV cache both grow.

**Against the head table (§7 of INDEX):** the budget moving 495 M → 522 M takes **Qwen2.5-Coder-7B
(98%) and Nemotron-H-8B (97%) back under 100%**. Two donors of twelve still cannot reach 50 tok/s
on this machine even if every weight outside their output head were free — Qwen3-8B at 112% and
gpt-oss-20b at 105%. That is the whole distance a 10.3% runtime win buys on the question that
actually gates this programme.

The spread across organs is now **1.3×**, not 4×, and the ordering is by matrix size:
0.5 MB → 14.2, 9.6 MB → 16.0, 6.5 MB per layer → 13.6, 68 MB → 18.5. **Nothing in this runtime
is a per-call-overhead outlier any more**, and nothing is near the 40–44 GB/s DRAM aggregate or
probe-3's ~28 GB/s post-L3 figure.

### 12.3 A convention this ledger did not have

**A tok/s figure is only comparable to another taken at the same `--bench` length.** Attention is
`O(position)`, so a longer bench has a genuinely lower average: 1.09 ms/token at 300 tokens,
**2.85 ms at 800**. That alone is 1.7 ms/token, and it is most of the gap between a 300-token and
an 800-token number on this donor. `bench_matrix.py`'s idleness witness compared an 800-token run
to a 300-token reference and called an idle machine "CONTENDED" for exactly this reason.
**Every rate in this ledger from here on carries its bench length.**


### 12.4 `--fuse` × `OMP_WAIT_POLICY`, finally run — and both hypotheses die

`bench_matrix.py`, 8 cells × 6 rounds × 300 tokens, rotated order, paired within round.
Data: `benchmarks/donor_adaptation/engine/bench_matrix.json`, log alongside it.

**Read the per-organ ms, not the tok/s.** Across four separate sessions the baseline's own tok/s
moved 1.4–9.1% round to round, which swamps every effect here. The per-organ ms are a different
measurement — each is an average over 300 tokens × 24 layers — and the two `OMP_WAIT_POLICY` rows
of a config are an **internal replication** of each other. Where those two rows agree, the number
is real; where they disagree, it is noise. Every figure below is a case where they agree.

| config | qkv | o_proj | ffn | head | **TOTAL** | vs packed |
|---|---|---|---|---|---|---|
| packed | 0.916 | 0.639 | 11.800 | 3.760 | **18.29** | — |
| `--fuse` | **0.795** | 0.641 | 11.738 | 3.755 | **18.09** | **−0.20 (−1.1%)** |
| `--lut --lut-group 32` | **1.221** | 0.701 | **11.398** | **3.433** | **17.88** | −0.41 (−2.2%) |
| `--fuse --lut --lut-group 32` | 0.819 | 0.734 | **11.037** | 3.876 | **17.58** | −0.71 (−3.9%) |

(medians over the two policy rows; `attention`, `rope` and `norm` are flat across all four and
omitted.)

**H1 — OpenMP region count — is real and worth 2.5–3.2 µs per region, not 32.** `--fuse` removes
48 of qkv's 72 matvec calls per token and saves **0.120 ms** (a second, independent 6-round run:
`fuse_focus.log`), and removes 24 of the FFN's 72 and saves 0.062–0.078 ms. Two organs, two
different call counts, **the same per-call figure**. That is barrier scale. §11.4's "32 µs per
call → threads are being woken" was rope's `pow()` divided by qkv's call count.

The whole remaining region overhead is therefore `97 × 2.7 µs ≈ 0.26 ms` of 17.8 — **1.5%**. It
is not the lever, and no further fusion is worth building.

**H2 — `OMP_WAIT_POLICY=active` — does nothing.** `packed/active` vs `packed/default`: 18.218 vs
18.354 ms in one run, 0.994–1.005× in tok/s across every run. Inside the noise in both directions.

**`--fuse` is worth 1.1%, is bit-identical, and costs +111.6 MB** of duplicated weights. On a path
this ledger keeps concluding is memory-shaped, that is a bad trade at 0.5 B and a worse one at 10 B.
**Not adopted as a default.** It stays as a flag and as the instrument that measured the per-region
cost.

**`--lut --lut-group 32` is worth 2.2%, and it is NOT free**: it costs `rel l2 3.10e-02` on the
activations (§11.2 / INDEX §2.1). It helps exactly the two big-matrix organs — ffn −0.40, head
−0.33 — and **hurts the two small ones**, qkv +0.31 and o_proj +0.06. Fusing qkv first removes
that penalty (0.819 vs 1.221), which is why the combination is worth more than either.

### 12.5 What is actually left

| lever | measured | numerically free? |
|---|---|---|
| **rope hoist** | **+10.3%** | **yes, bit-identical** — done, `4f7b33c` |
| `--fuse` | +1.1% | yes, bit-identical — costs 111.6 MB |
| `--lut --lut-group 32` | +2.2% | **no**, `rel l2 3.10e-02` |
| both | +3.9% | no |
| `OMP_WAIT_POLICY=active` | 0 | — |
| every organ at the head's 18.5 GB/s | +23% (ceiling, not a plan) | — |

After the rope hoist the four organs sit in a **1.3× band** (13.6–18.5 GB/s) and the ordering is by
matrix size. There is no outlier left to chase. **The remaining speed on this runtime is in the
weight count, not in the kernel** — which is what §1's budget line has said from the beginning, and
is why the open list leads with the quality side rather than with another kernel.

---

## 13. AMENDED 2026-09-06 by E3 — the last sentence of §12.2 was right, and it was worth 9×

`probes/E3_ENGINE_AT_TARGET_SCALE.md`. §12.2 ends with a hedge about its own weakest constant:

> *"The 1.17 ms is measured at 0.5 B and short context and is **optimistic for a 10 B**, where
> attention and the KV cache both grow."*

E3 measured six shapes from 0.494 B to 10.603 B active weights, at two context lengths, with
synthetic weights gated against real exported artifacts (byte-identical file sizes; **0.000%**
timing difference at 1.5 B, 0.357% at 0.5 B, interleaved). The hedge was correct and understated.

### 13.1 What is withdrawn

**`f`, the non-weight fixed cost.** §12.2 reserves **1.17 ms/token**. At 0.5 B that reproduces to 1%
(measured 1.183). At a 10 B shape:

| | `f` | vs 1.17 |
|---|---|---|
| `T10` @ 300 tokens of context | **10.576 ms** | **9.0×** |
| `T10` @ 800 tokens of context | **26.088 ms** | **22.3×** |

**`f` is work no weight format can remove.** `1000 / f` is therefore the engine's ceiling with an
infinitely fast weight path: **94.6 tok/s** for a 10 B shape at 300 context, and **38.3 tok/s** at
800. **50 tok/s at 800 context is unreachable at a 10 B shape by any weight-side work whatsoever**,
and 100 tok/s is unreachable at 300.

**The budget.** `27.7 G-w/s × (20 − 1.17) ms = 522 M` is superseded by `r_w × (20 − f)` with the
measured `f`: **318 M at 300 context, and 0 at 800.**

### 13.2 An arithmetic error in §12.2, independent of the above

§12.2 states `Delivered: 493,961,216 / 17.833 ms = 27.7 G-weights/s`. **17.833 ms is the wall** — the
same table's own `total` row — so the 1.17 ms of non-weight work is **already inside that
denominator**. Multiplying by `(20 − 1.17)` charges it a second time.

Each convention has exactly one self-consistent form, and the two agree to 0.7%:

| | |
|---|---|
| `r_wall × 20` = 27.70 × 20 | **554 M** |
| `r_w × (20 − f)` = 29.65 × 18.83 | **558 M** |
| as published | 522 M |

**~6% too tight, in the safe direction**, which is why it stood for two weeks. It is not the reason
the budget moves — §13.1 is, and it moves the other way and much harder — but the two must not be
confused, and `speed/e3_budget_by_shape.py` prints both conventions side by side rather than picking.

> **Law:** *a rate and a reservation must be read off the same denominator.* The check is cheap and
> would have caught this: the two self-consistent forms must agree.

### 13.3 What is confirmed, and improved

**The delivered rate does not degrade with scale — it rises.** In §12.2's own convention
(weights / wall):

| shape | active w/token | G-weights/s | weight-organ GB/s |
|---|---|---|---|
| `S05` 0.5 B | 0.494 B | **27.5** | 14.7 |
| `S15` 1.5 B | 1.544 B | 29.9 | 15.6 |
| `S3` 3 B | 3.086 B | 30.6 | 15.9 |
| `M7` 7 B | 7.114 B | 32.8 | 16.9 |
| `Q8` 8 B | 7.568 B | 32.8 | 17.0 |
| `T10` 10 B | 10.603 B | **32.8** | 16.9 |

`27.5` at `S05` reproduces §12.2's `27.7` to 0.7%. The rise to **32.8 (+18.3%)** is per-call overhead
amortising over bigger matrices — consistent with the **2.5–3.2 µs** OpenMP region §12.4 measured,
paid `7 × L` times per token. **§12.2's rate anchor is conservative, not optimistic.** Only `f` was
optimistic, and only `f` mattered.

**§12.2's organ ordering survives and sharpens.** The 1.3× band holds at every scale, and the
ordering by matrix size holds — but *which* organ is largest changes completely:

| share of the token, `--bench 300` | `S05` | `T10` |
|---|---|---|
| `ffn` | 64.7% | 78.0% |
| `qkv_proj` + `o_proj` | 8.3% | **17.9%** |
| **`head`** | **20.5%** | **1.1%** |
| `attention` | 6.1% | 3.0% |

**The head was the floor of a 0.5 B, not of a 10 B.** Every probe this programme owns was measured
where the head was the obvious target.

### 13.4 Where the remaining distance is, measured

The `attention` organ costs **one FMA per ~4 cycles per thread across all twelve measured points
(±7%)**, invariant to shape, context length, KV cache size over a 16× range, GQA factor and
vocabulary. That is the signature of a **serial dependency chain**, not a bandwidth wall — 24 GB/s of
touched bytes is well under this machine's DRAM, and a bandwidth limit would not produce a constant
*per FMA*.

The source agrees: the inner product is `for(i<HD) d += qh[i]*kt[i]`, a floating-point **reduction**,
and the engine is built `clang -O3 -mavx2 -mfma -ffp-contract=on` with **no `-ffast-math`** (forbidden
since Phase 35), so the compiler may neither reassociate nor vectorise it. **Every matvec in this
engine is hand-written AVX2 with an 8-wide accumulator; the attention loop is the one hot loop that
is not.**

Marked **corroborated, not proven.** The experiment that settles it: give the `K` loop 4–8
independent accumulators. If `f` falls by roughly the accumulator count it was latency; if it does
not move, the lever is an **int8 KV cache** instead — the cache is `[L][maxseq][NKV×HD]` **fp32** and
has never been touched. **Parity gate mandatory:** accumulation order is exactly what E1 traced its
`1.5e-05` engine-vs-PyTorch delta to.

### 13.5 The goal, stated from measurement

| | |
|---|---|
| `T10` (10.603 B active, ternary packed head) | **3.090 tok/s** @300, **2.960** @800 |
| needed for 50 tok/s | 530 G-weights/s |
| measured | 32.8 G-weights/s |
| **short by** | **16.2×** |
| the same thing without any rate | `50 / 3.090` = **16.2×** |
| ceiling at zero weight cost, @800 | **38.3 tok/s** — **WITHDRAWN by §14.1** |

> **Corrected 2026-09-06, same day, same law.** This row was first published as **15.7×**,
> which divided a *wall* quantity by `r_w` instead of `r_wall` — the §13.2 error committed a
> second time, by me, in the section that names it. `50 / 3.090 = 16.18` is the form with no
> denominator to swap, and it is the one to quote.

The dense path is 16× away. **The last 1.3× of it is not reachable by any weight-side work at 800
context**, which is the first time this ledger has been able to say where the wall is rather than how
far away it is.

---

## 14. AMENDED 2026-09-06 by E4 — §13.4 was right about the mechanism, and it was worth 2× on `f`

`probes/E4_ATTENTION_ACCUMULATORS.md`. §13.4 marked its reading **corroborated, not proven** and
named the experiment: *"give the `K` loop 4–8 independent accumulators. If `f` falls by roughly the
accumulator count it was latency; if it does not move, the lever is an int8 KV cache instead."*

It was latency. **`LATENCY-CONFIRMED`**, from an interleaved A/B measurement (median ratio 0.4875,
spread 2.98%, all three pairs below the 0.50 line fixed before the run).

### 14.1 What is withdrawn

**§13.5's ceiling.** §13.1 stated: *"50 tok/s at 800 context is unreachable at a 10 B shape by any
weight-side work whatsoever"*, from `1000/f = 38.3 tok/s`. That is withdrawn.

| `T10`, one sweep | `serial` (E3's engine) | **`avx4`** |
|---|---|---|
| `f` @300 | 9.846 ms | **5.216 ms** |
| `f` @800 | 24.678 ms | **12.735 ms** |
| **ceiling `1000/f`** @800 | 40.5 tok/s | **78.5 tok/s** |
| **budget at 50 tok/s** = `r_w × (20 − f)` @800 | **0** | **259 M** |
| same, @300 | 359 M | **523 M** |

**50 tok/s at 800 context is a weight-side problem again.** The absolute budgets carry the ±5%
between-sweep band of §14.3; the ratios do not.

**The cost.** `|ΔBPB| = 3.03e-06` on a real donor over the pinned 24×512 slice — **1650× under
σ_seed**, and smaller than the `1.53e-05` that E1 measured between this engine and PyTorch.

### 14.2 The mechanism, measured instead of inferred

A planted control (`serial2`: the same dot product twice, `(d1+d2)*0.5f`, **bit-identical**) splits
the organ into the loop under test and everything else; a third point (`serial3`) that cannot be
fitted lands within **0.56%** of the prediction. So the split is a measurement:

| `T10` @800 | organ ms | **`Q·K` loop** | speedup | unique K GB/s |
|---|---|---|---|---|
| `serial` | 24.463 | 14.647 | 1.00× | **5.4** |
| `ilp4` — 4 scalar chains, no SIMD | 16.148 | 6.332 | 2.31× | 12.4 |
| `avx1` — 8 lanes, one chain | 13.238 | 3.422 | 4.28× | 23.0 |
| **`avx4`** | 12.058 | **2.242** | **6.53×** | **35.1** |

**2.31× from breaking the dependency chain alone**, moving not one extra byte, settles the
latency-vs-bandwidth question by itself. And the endpoint is informative: 35.1 GB/s of *unique* K
bytes is this machine's DRAM read rate, so the loop that was **6.5× below its own memory limit** now
sits on it. (140 GB/s of *touched* bytes is not achievable from DRAM — so the GQA re-reads are served
by cache, which §13.4 explicitly declined to claim.)

**The int8 KV cache is retired before being built.** §13.4 named it as the alternative lever. It cuts
the dot loop's unique bytes 4×: 2.242 → ~0.6 ms, organ 12.058 → ~10.4 ms = **~1.16×**. It was the
right lever for a loop that no longer exists.

**The floor is now `R`** — the softmax pass plus the `A·V` loop — measured at **9.816 ms** and
**81.4% of the organ**, untouched by every arm here. Its composition is **now measured**.

**§14.2-bis — what `R` is made of (E5, `probes/E5_DECOMPOSE_R.md`, verdict `OVERHEAD-DOMINATED`).**
E5 split `R` with the same doubling trick, ten arms interleaved per token inside one process.
Nine 3× predictions the components were not fitted to all passed, worst error −1.15%. Its `X` =
**2.253 ms** reproduces §14's 2.242 to 0.5% by a method sharing no arm with it. Ratios travel
between sweeps and absolutes do not, so E5's shares are applied to **this ledger's own `R`**:

| term | share of `R` (E5) | ms/token on `R` = 9.816 |
|---|---|---|
| `S`, the softmax pass | 25.3% | **2.485** |
| `Y`, the `A·V` loop | 20.6% | **2.020** |
| `P`, neither loop | **54.0%** | **5.301** |

**More than half the floor is not a loop.** The OpenMP fork — the one candidate that has been
priced — is **1.4% of `P`**, so `P` is not the fork and what it is has never been measured.
E5's own `1000/f` = 68.4 tok/s is **not** a correction to the 78.5 above: its sweep read the
organ 15.5% high, and §14.7 of its brief refuses the absolute for exactly that reason.
One more thing E5 hands over: at `S05` @800 the split **inverts** — 47.7% softmax, 39.2% `P` —
so a softmax fix measured on the small model arrives at target scale worth about half of what it
looked like.

### 14.3 A correction owed to every absolute number in this ledger

E3's own binary was rebuilt from `d7977c8^` and timed in the same session as E4's arms:

| point | published here | **E3's own binary, re-run** | |
|---|---|---|---|
| `T10` @300 | 3.090 | **3.040** | −1.6% |
| `T10` @800 | 2.960 | **2.880** | −2.7% |
| `S05` @300 | 55.700 | **53.370** | −4.2% |
| `S05` @800 | 49.280 | **48.360** | −1.9% |

The same code also read **3.240** and **3.030** at `T10` @300 two hours apart, while the within-run
IQR stayed at **0.005**.

> **Law: a within-run IQR is not a reproducibility interval.** Within-sweep dispersion on this
> machine is 0.000–0.015 tok/s; **between-sweep dispersion is 5–10%**. Every absolute tok/s in this
> ledger carries that band, and the published IQRs do not show it. **The ratios are unaffected** —
> they were taken inside one sweep — and every conclusion in §12–§14 is a ratio.

This is not a re-run request. It is a band to quote: **±5% on any absolute tok/s here**, and a
prohibition on comparing a number in this ledger to one measured in another session without
re-measuring the reference alongside it.


## 15. AMENDED 2026-09-07 by E6 — the ceiling is not the rate, and it had been read as one

**§15.1 — the correction.** `f` is the non-weight cost per token; `1000/f` is the rate the engine
would reach **if the weight path took zero time**. It is a denominator, not a measurement, and two
figures in this ledger have been read as throughput when they are ceilings:

| figure | § | what it is |
|---|---|---|
| `1000/f` = **78.5 tok/s**, `T10` @800 | §14 | ceiling, free weight path |
| `1000/f` = **68.4 tok/s**, same cell | §14.2-bis | the same ceiling, on a sweep that ran 15.5% hot |
| **3.09–3.14 tok/s** | E5 `results/e5/run6.log` | **measured throughput at that cell** |
| 52.4–52.9 tok/s | same log | measured, `S05` @800 (0.5 B) |

**The 10.6 B shape decodes at about 3.1 tok/s — roughly 16× short of the 50 tok/s goal.** Neither
E4 nor E5 ever claimed otherwise; both published `1000/f` as a ceiling and said so. This section
exists because the distance between "ceiling" and "rate" is one word in a table, and the ledger
should not depend on the reader supplying it. **A ceiling is a denominator** — the same law that
killed the 39.7% anomaly, in a new place.

**§15.2 — the first generation rates, and why they are witnesses and not results.** E6 added a
`--generate` mode (greedy argmax) and ran it on the real converted donors:

| arm | weights | decode | prefill |
|---|---|---|---|
| A1 | `qwen25-05b_f32.bin` | 18.80–18.89 tok/s | 17.7–18.0 |
| A2 | `qwen25-05b_tqh.bin` | 59.80–61.58 tok/s | 52.3–55.0 |
| A3 | `qwen25-15b_tqh.bin` | 20.49–20.94 tok/s | 11.7–20.1 |

**These were taken on a loaded machine and a contended timing is not a timing.** They are here to
show that generation costs what teacher-forced decoding costs — the argmax over 151,936 logits and
the token feedback add nothing visible — and for no other purpose. **A2's 61 tok/s is not 50 tok/s
reached**: 0.5 B, a context of 3–8 tokens, and output that `probes/E6_GENERATION.md` §2 shows is
not language.


## 16. AMENDED 2026-09-07 by E7 — the synthetic weights were an honest proxy, and the wall is real

**§16.1 — the check nobody had run.** Every target-scale number in §13–§15 was measured on
`T10`, a **synthetic** shape file: right dimensions, untrained weights. The defence was that a
matvec does not care what the numbers are. E7 exported a **real** donor at a comparable size and
measured the same thing:

| | active weights/token | tok/s @300 | delivered weight rate |
|---|---|---|---|
| `T10`, synthetic 10.603 B (§13) | 10.603 B | 3.090 | **32.8 G-weights/s** |
| **Qwen2.5-Coder-7B, real 7.072 B** | 7.072 B | **4.460** | **31.5 G-weights/s** |

**−4.0%, inside §13's own ±5% band for an absolute rate** — the two are indistinguishable by
the instrument. **The proxy holds and §13–§15 stand.**

**§16.2 — the number, on trained weights.** `--bench` on the packed arm, 3 repetitions per cell:

| cell | median | spread |
|---|---|---|
| 300 context | **4.460 tok/s** | 0.45% |
| 800 context | **4.580 tok/s** | 0.00% |

**11.2× short of 50 tok/s**, on 7.07 B of real weights, on an engine whose attention organ is
already 6.5× faster (§14) and fully decomposed (§14.2-bis). The brief predicted **4.66** from
§13's delivered rate alone, before the donor was exported: **−4.3%**. The weight-rate model
transfers across shape and across trained-vs-synthetic.

**§16.3 — an ordering that inverted, and was noise.** *(This paragraph replaces a two-point fit
that stood here from 2026-09-07 and was refuted the same day; the original is preserved verbatim in
`probes/E7_REAL_LARGE_DONOR.md` §7.)* 800 context reads **faster** than 300 here (4.580 vs 4.460),
the opposite of `T10`. Two-point arithmetic fitted a **fixed ~2.8 s inside the timed region**, which
if real would make every short `--bench` in this ledger read low. A third point settled it:
**`--bench 1600` medians 4.230**, against the fit's prediction of 4.616 (band 4.50–4.68) and against
a threshold of 4.580 fixed before the run. **Refuted.** The inversion is **2.7% — inside §13's own
±5% band.** No cell in this ledger reads low. *Standing law, reinforced: a difference smaller than
the reproducibility band is not a phenomenon, and must not be given a mechanism.*

## 17. AMENDED 2026-09-07 by E7 §8.2 — `f` is linear to 1600 tokens (E4's owed item 3, closed)

Nothing in §§13–16 bounded `f` above 800 tokens of context; every `1000/f` ceiling quoted here was
read at 300 or 800 and silently assumed to extrapolate. It does. Measured on the real Coder-7B,
profiled runs admitted only where the **`ffn`-invariance witness** holds — the FFN organ cannot
depend on context length, so any cell whose `ffn` leaves the uncontended ~175–180 ms plateau was
taken under load and is discarded (six of nine cells survived; the three discarded would have
manufactured a knee):

| context | **attention ms/token** | slope per token of actual context |
|---|---|---|
| 300 | 4.932 | — |
| 800 | 12.90 (median of 13.092, 12.713) | **0.0319 ms** |
| 1600 | 25.64 (median of 25.462, 25.636, 25.951) | **0.0319 ms** |

**Two intervals agreeing to 0.2% over a 5.3× range.** `--bench N` averages the organ over positions
0…N−1, so mean context is N/2 and the slope column is the physical quantity. **`1000/f` may be
extrapolated linearly to at least 1600 tokens**, and the ceilings in §§13–14 are not hiding a knee.

## 18. AMENDED 2026-09-07 by E7 §9 — which attention kernel produced the numbers in this ledger

`donor_engine.c:116` reads `static int g_attn = ATTN_SERIAL;`. **The engine's default attention
kernel is `serial`** — E4's second-slowest arm — not the `avx4` that §14 credits with taking the
`Q·K` dot loop 6.53× and `f` from 24.678 to 12.735 ms. `avx4` is reachable only via an explicit
`--attn avx4`.

**What this does and does not change.** E4's own runners pass `--attn` on every point, so **§14's
table is sound**. `e3_bench.py` defaults `--attn` to the empty string and never passes it, so
**E3's `T10` baseline (§13) and every E7 figure (§16) are `serial` readings.** E4 already published
the rate consequence at `T10`: `avx4` **3.230** vs `serial` **3.110 tok/s, +3.9% — inside §13's own
±5% band.** **No number in §§13–17 requires a numeric correction.** What was missing, and is
supplied here, is the sentence naming the kernel.

The arm transfers to a real donor — profiled, witness-admissible, on Coder-7B:

| context | `serial` | `avx4` | ratio | E4's `T10` ratio |
|---|---|---|---|---|
| 300 | 4.932 | **2.336** | **0.474×** | — |
| 800 | 12.90 | **5.737** | **0.445×** | 0.493× |

**Scope, fixed before the rate was re-measured:** attention is **2.2% of the token at 300 context
and 10.5% at 1600**; the FFN is **73–80%**. Halving a 2.2% organ cannot move an 11.2× gap. **This
is a correctness note about which kernel ran, not a speed result.**

**The default is deliberately NOT being changed.** `serial` is the arm every prior probe's baseline
was taken on; flipping `g_attn` would silently re-base E1–E7. The fix is a runner that passes
`--attn` explicitly and this section — not an edit that makes old numbers unreproducible.


### 18.1 AMENDED same day — the arm's win reaches the wall, and the un-profiled path cannot see it

Two pre-registered attempts to read the `avx4` win as a *rate* on the real donor (E7 §10):

| | |
|---|---|
| **G-Y1** @300 | `serial` 4.460 → `avx4` **4.450**. **The null §18 called in advance** — attention is 2.2% of the token, the arm is 0.474×, the effect is ~1%, and nothing here resolves 1% |
| **G-Y1** @1600 | `serial` 4.230 (spread **6.1%**) → `avx4` 4.290 (1.2%). **Did not decide**: the baseline's dispersion exceeds the effect and its maximum sits above the challenger's median |
| **G-Y1b** @1600, arms interleaved | paired ratios 0.950 / 1.094 / 0.973, **median 0.973** against a pre-registered band of 1.00–1.09. **VOID.** Within-arm dispersion hit **11.5%**, ~2× the largest effect attention can produce there |

**No cause is named for the 11.5%**: one CPU sample after the run read 46%, three minutes later
read 7 / 13 / 14%, nothing of ours running either time, and a single instantaneous sample is not a
load measurement.

**The win does reach the wall**, established instead from four witness-admissible profiled cells at
1600 (`ffn` all inside the 176–181 plateau): **the organs sum to the wall to 0.03% in every cell**,
and **`TOTAL − attention − ffn` is arm-invariant** — `avx4`'s 41.232 sits inside `serial`'s own
40.475–42.159 scatter. On the two cells with closest `ffn`, attention falls 14.32 ms, `ffn` differs
−1.06, the wall *should* fall 15.38 and falls **16.33 — agreement to 0.4% of the token**. That is
worth **~6–7% at 1600 context**, and it is **NOT published as a rate**, because it comes from
profiled walls and §12 forbids exactly that.

> **NEW DEFECT, and it is general. `--bench` has no contention witness.** The `ffn`-invariance
> witness that makes §17 and E7 §8.2 trustworthy **exists only under `--profile`**. On the plain
> `--bench` path nothing can contradict the operator's belief that the machine was idle — and
> **every un-profiled rate in this ledger rests on that belief.** The fix, owed first: time the
> `ffn` organ on the plain `--bench` path and print it on the `BENCH` line. **Gate: the `BENCH`
> rate must be unchanged** — adding a timer to the hot path to detect contention would be an
> excellent way to manufacture some.

## 19. DERIVED 2026-09-07 — how much of the 16.2× is engineering, and how much is not available at all

**This section contains no new measurement.** It is arithmetic on numbers already published in
§1, §12.2, §13 and §14, put together for the first time. Every input is cited; nothing is
estimated. It exists because §13.5 says the dense path is **16.2× short of 50 tok/s** and does not
say how much of that gap an engine could ever close.

### 19.1 The inputs, all published

| quantity | value | where |
|---|---|---|
| `T10` @300, measured | **3.090 tok/s** = 323.6 ms/token | §13.5 |
| `f` at `T10` @300 (non-weight path), best measured | **5.216 ms** (`avx4`) | §14, row `f` @300 |
| therefore the **weight organs** at `T10` @300 | **313.8 ms** (using §13.5's own `f` of 9.846 ms, the `serial` arm 3.090 was taken on) | subtraction |
| weight-organ delivered bandwidth at `T10` | **16.9 GB/s** = 32.8 G-weights/s | §13.3 |
| the engine's byte convention | **0.515 B/weight** (32.8 G-w/s ↔ 16.9 GB/s) | §13.3, self-consistent |

And three *measured* ceilings for a streamed weight read on this machine, from slowest to fastest:

| ceiling | value | what it actually is |
|---|---|---|
| probe-3, post-L3-cliff streaming | **~28 GB/s** | §12.2's correction note. Conservative: a different bench |
| **proj-GEMV streamed floor** | **37.0 GB/s** | §1. **The closest analogue** — the same kernel shape reading streamed weights |
| DRAM aggregate | **42 GB/s** | §1, saturated at 3 threads. Physics |

### 19.2 The ceiling this engine can never pass at a 10 B dense shape

Put the weight organs at each ceiling and keep the best measured `f`:

| weight path at | weight organs | + `f` 5.216 | **ceiling** | headroom over today |
|---|---|---|---|---|
| **16.9 GB/s (today)** | 313.8 ms | 319.0 ms | **3.13 tok/s** | 1.00× |
| 28 GB/s | 189.4 ms | 194.6 ms | **5.14 tok/s** | 1.66× |
| **37.0 GB/s** | 143.3 ms | 148.5 ms | **6.73 tok/s** | 2.19× |
| 42 GB/s | 126.2 ms | 131.4 ms | **7.61 tok/s** | 2.49× |

> **A perfect engine — every weight organ at a bandwidth this machine has actually been measured
> delivering, and the non-weight path already at E4's best `f` — tops out between 5.14 and
> 7.61 tok/s on a dense 10.6 B. 50 tok/s is 6.6× to 9.7× beyond that.**

These are **ceilings, and a ceiling is a denominator** (§12.2's law): they assume every organ
simultaneously at a rate no organ has yet reached, and they still do not get within 6×.

### 19.3 The 16.2×, split

| | factor | what would have to happen |
|---|---|---|
| available from **engine work** | **1.7– 2.5×** | move every weight organ from 16.9 GB/s to a streaming rate this machine is measured at |
| **not available at this shape, by any engine work** | **6.6– 9.7×** | — |
| product | 16.2× | §13.5 |

**This bounds every future optimisation on the weight path at 2.5× and no more, at a 10 B dense
shape on this machine.** It is not an argument against doing that work — 2.5× is large — it is an
argument that the work cannot finish the job, and a number to check any proposal against before
building it.

### 19.4 The same statement as a sparsity budget, which is the actionable form

At 50 tok/s a token is 20.0 ms; `f` takes 5.216 of it, leaving **14.78 ms** for weights.

| weight path at | active weights/token affordable | **as a share of 10.6 B** |
|---|---|---|
| **16.9 GB/s (today)** | **485 M** | **4.6%** |
| 37.0 GB/s (proj-GEMV floor) | **1.06 G** | **10.0%** |
| 42 GB/s (DRAM aggregate) | **1.21 G** | **11.4%** |

> **Even a perfect engine on this machine permits at most ~11% of a 10 B to be active per token at
> 50 tok/s. Today's engine permits 4.6%.** Engine work is worth roughly a **2.5× larger sparsity
> budget** — real, bounded, and not a substitute for the sparsity.

This restates §1's *"50 tok/s → ≤ 680 M active weights per token"* with two differences: it uses
**this engine's own measured `T10` rate** rather than the expert path's kernel-pure ceiling, and it
charges `f`, which §1 did not. **The direction of the correction is unfavourable** — 485 M, not
680 M, at today's rate.

### 19.5 What this does not say

It does **not** say the goal is impossible; it says it is impossible *at a dense 10.6 B active
per token on this machine*, which is a statement about the **model**, not the engine.
`SCALEUP_ARCHITECTURE.md` already prescribes the remedy (thinking/knowing split, MoE, cache
residency) and §1's budget rows already priced it; what was missing until now is the **bound on
the alternative** — that no amount of engine work substitutes for it, and exactly how much engine
work is worth.

It also does **not** cover §19's own weakest assumption: that the weight organs are
bandwidth-limited at all. §12.2 says the opposite for `S05` (*"nothing here is at the bandwidth
wall"*, ~1.2 ms of issue against 19.9 measured) and §13.4 found the attention organ to be
**latency**-bound, not bandwidth-bound, which is why E4 bought 6.53× there. **If the weight organs
are also latency- or overhead-bound rather than bandwidth-bound, §19.2's ceilings are too low and
the engineering share is larger than 2.5×.** That is the experiment §19 argues for, and it is the
open item `P` (§15, INDEX item 0) asked from the other end.


## 20. AMENDED 2026-09-07 by E7 §11 — `--bench` carries a contention witness, and it was gated three ways

§18.1 recorded the defect: the `ffn`-invariance witness existed only under `--profile`, so every
un-profiled rate in this ledger rested on the operator's belief that the machine was idle. Fixed.

**What it is.** `TICW`/`TOCW` at the FFN block and nowhere else, timing **layer 0**, appending
`ffn~ <x> ms/tok` to the `BENCH` line, **on by default** (`--no-witness` restores the old hot path).
The FFN because it is 73–80% of the token and cannot depend on context length; default-on because
a witness a runner must remember to pass is precisely the §18 defect.

**It failed its first gate and was rebuilt.** At `2L` = 48 timestamps per token the paired median
was **0.9877** against a gate of **0.990** — a **1.2%** cost against a **0.008%** prediction, 150×
off. `now_s()` had been calling `QueryPerformanceFrequency` on every timestamp. Rebuilt to **2**
timestamps per token with the frequency cached:

| gate | result |
|---|---|
| **G-W1b** — 10 interleaved pairs, smallest/fastest shape | **PASS**, median **0.9996** (band 0.99–1.01; predicted 0.995–1.000) |
| **G-W2b** — `--logits` sha256, witness vs none | **PASS**, identical `d4960bc7…0424` |
| **G-W3b** — **planted control**, must fire under load | **PASS**, idle `ffn~` 11.631 → loaded **16.225**, **+34.3%** over the idle maximum of ten reps |

> **DECLARED COST, and it touches this ledger. `now_s()` is cheaper as of `02baefa`, so profiled
> walls taken after it are NOT comparable to profiled walls taken before it** — §§12, 14, 17, 18.1
> and E7 §8.2/§10.4 are all pre-commit. **Organ splits and slopes are unaffected**: a uniform
> per-`TIC` cost cancels in a difference, which is all those sections used.

> **It validates nothing retroactively.** Every rate in §§12–19 was taken without it. §18.1's VOID
> stands and G-Y1's 1600 cell stands undecided. The witness makes the **next** measurement
> checkable.
