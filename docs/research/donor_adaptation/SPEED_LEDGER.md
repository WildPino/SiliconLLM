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
| **50 tok/s** | **554 M** | **5.5%** |
| 100 tok/s | 277 M | 2.8% |

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
