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
