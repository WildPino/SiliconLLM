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
the engineering share is larger than 2.5×.** That is the experiment §19 argues for.

> **CORRECTION, same day.** This paragraph first pointed at open item `P` (§15, INDEX item 0) as
> that experiment. **It is not.** `P` lives inside `R`, the *attention* residue — softmax + `A·V`
> + `P` — and §19's ceilings are about the **weight** organs (`ffn`, `qkv`, `o`, `head`).
> Decomposing `P` cannot move them. **The experiment §19.5 actually argues for is a decomposition
> of the FFN organ itself**, on E5's method: is 176 ms at 16.9 GB/s a bandwidth wall, or is it
> chopping and per-call overhead? §12.2 already has the clue — *"the rate tracks how the work is
> CHOPPED"*, with ~32 µs per call unexplained on `qkv` — and E4's whole result was that an organ
> everyone read as bandwidth-bound was latency-bound. **This is now the open item that sets the
> activation budget any MoE design has to hit**, because §19.4's answer moves by up to 2.5×
> depending on it: 4.6% of a 10 B active per token, or 11.4%.


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


## 21. AMENDED 2026-09-07 by E8 — §12.5 was wrong: the kernel was the lever, and it was worth 1.35×

§12.5 closed with *"there is no outlier left to chase; the remaining speed on this runtime is in
the weight count, not in the kernel."* It reached that from the **narrowness** of the organ band
(13.6–18.5 GB/s, ordered by matrix size). **A narrow band is equally the signature of one shared
ceiling**, and every one of those organs runs the same `matvec` inner loop.

It did. `probes/E8_MATVEC_DEPENDENCY_CHAIN.md`, verdict `CHAIN-CONFIRMED`.

**Every `matvec` accumulated into ONE register**, so consecutive FMAs were serially dependent and
the loop ran at Zen 2's 5-cycle FMA **latency** rather than its 2/cycle throughput — verified in
the emitted assembly before the claim was made. `-ffast-math` being forbidden here (Phase 35) is
why the compiler had not removed it.

### 21.1 The measurement

`--mvacc {1,2,4}`; 1 is the old loop byte for byte. Coder-7B packed, `--bench 300`, 10 interleaved
pairs, idle machine, witness on:

| | `--mvacc 1` | `--mvacc 4` | ratio |
|---|---|---|---|
| rate | **4.73 tok/s** (1.9%) | **6.37 tok/s** (2.8%) | **1.3491** |
| `ffn~` | 168.550 ms | 124.469 ms | 1.3529 |
| **weight path** | **16.7 GB/s** | **22.5 GB/s** | |
| delivered | 33.5 G-w/s | **45.0 G-w/s** | |

Ten pairs, ratios 1.3368–1.3615, **1.8% spread, none dissenting.** At 0.5 B: 1.2301.

**All four weight organs moved together** — qkv 1.427×, o_proj 1.431×, head 1.494×, ffn 1.409× —
from one change to one loop, which is what a shared ceiling looks like when it lifts.

### 21.2 The negative control, and an in-run bandwidth demonstration

The same flag on the **fp32** kernel, where the chain has 3.5× slack over DRAM, **must** buy
nothing. Predicted 0.99–1.03 in the pushed brief, ≥1.10 declared as refuting the mechanism.
**Measured 1.0029.**

Unasked for: that arm delivers **33.3 GB/s** through the same `matvec`, on the same machine, in the
same run, while the packed arm managed 16.7. **The packed path was never at a memory limit** — an
in-run demonstration, independent of any cycle counting.

### 21.3 What it does to §19

§19.3 said **1.7–2.5×** was the entire budget available to engine work. **E8 has taken 1.35× of
it.** Remaining to the measured ceilings: **1.24×** (28), **1.64×** (37.0), **1.87×** (42).

§19.4's sparsity budget, restated at the new rate:

| weight path at | active weights/token at 50 tok/s | share of a 10.6 B |
|---|---|---|
| 16.9 GB/s (§19's "today") | 500 M | 4.7% |
| **22.5 GB/s (today, after E8)** | **665 M** | **6.3%** |
| 37.0 GB/s (proj-GEMV floor) | 1094 M | 10.3% |
| 42 GB/s (DRAM) | 1242 M | 11.7% |

**§19.3's conclusion stands and is better supported:** the engine had more in it than §12.5
believed, and it still cannot close a gap that is a property of the model. Coder-7B is now
**7.8× short of 50 tok/s**, where it was 11.2×.

### 21.4 The convention this ledger now needs

**Every rate published before `§21` is a `--mvacc 1` reading.** Not withdrawn, not wrong —
labelled. `--mvacc 1` restores the single-chain loop byte for byte and reproduces them exactly
(verified: `--logits` sha256 identical under the new binary). **The default is now 4**, because a
flag a runner must remember to pass is exactly the §18 defect.

**Profiled walls after E8's commit are not comparable to profiled walls before it** — the FFN
carries four new `--profile`-only sub-timers. Second such declaration in two days; splits and
slopes are unaffected.

### 21.5 The FFN, decomposed — §19.5's owed item, closed

| inside `ffn` at Coder-7B | `--mvacc 1` | `--mvacc 4` |
|---|---|---|
| gate+up | 109.599 | 76.661 |
| down | 55.070 | 37.973 |
| **glue(silu)** | **13.215** | **11.628** |
| residual | 0.038 | 0.035 |
| `sum/ffn` | 1.0000 | 0.9999 |

**It was neither bandwidth nor chopping.** `gate+up` moves exactly **2×** `down`'s bytes while
being chopped **10.6×** differently (37,888 output rows against 3,584); the measured ratio is
**1.990** and **2.019**. A pure-bandwidth ratio is 2.000. **Per-row and per-call cost was never the
binder** — settled by an invariant, not a timing.

**And `glue(silu)` is now 7.4% of the whole token for zero weight traffic**: 530,432 `expf` calls
per token, never charged in this ledger, untouched by E8. The rope `pow()` hoist of §12 was the
same shape and worth +10.3%.

## 22. AMENDED 2026-09-07 by E9 — the SwiGLU glue ran on one thread, and it was worth +5.6%

`probes/E9_GLUE_PARALLEL.md`, verdict `GLUE-CONFIRMED`. Pre-registered and pushed as
`briefs/BRIEF_E9_GLUE_PARALLEL.md` (`82dbb54`) before any measurement. Opened by §21.5's last
paragraph — the item E8's decomposition found by accident.

### 22.1 The measurement

| Coder-7B packed, `--bench 300`, 10 interleaved pairs, idle, witness on | before | after |
|---|---|---|
| rate | 6.45 tok/s (spread 4.0%) | **6.79 tok/s** (spread 4.0%) |
| `ffn~` witness | 123.812 ms | 116.205 ms |
| **paired median ratio** | — | **1.0557** (spread 3.8%) |

Bands fixed before the run: **≥1.035 CONFIRMED**, 1.015–1.035 UNDECIDED, ≤1.015 REFUTED.
**Predicted 1.055–1.066 — it landed inside its own band**, the first prediction in this programme's
last three to do so, and the first derived by E8 §3's corollary (from a *measured* quantity divided
by a *structural* factor) rather than from first principles.

### 22.2 The gate is sha256, not parity

`s->hb[i] = silu(g[i]) * u[i]` is an **elementwise map with no reduction**: each `i` is written once
and reads only its own inputs, so splitting the index range across threads cannot change a bit.
E8 re-partitioned a float sum and could never claim this. **G-G1: `--logits` output byte-identical,
`b94b56d002880d84…548765`, before and after.** +5.6% at zero numeric cost.

### 22.3 The FFN sub-breakdown, updated (supersedes §21.5's `--mvacc 4` column)

| inside `ffn` at Coder-7B, `--profile --bench 100` | before E9 | after E9 |
|---|---|---|
| gate+up | 75.078 | 73.597 |
| **glue(silu)** | **11.346 (9.2% of ffn)** | **2.938 (2.6% of ffn)** |
| down | 37.286 | 36.403 |
| residual | 0.035 | 0.032 |
| ffn | 123.752 | **112.980** |
| TOTAL (organs summed) | 153.219 | **141.901** |
| `sum/ffn` | 0.9999 | 0.9999 |
| profiled rate | 6.59 | **7.12 tok/s** |

**`rmsnorm` was left serial on purpose**, and the brief said so before the run: `norm+glue` is
0.330 ms/token over 56 calls, and at §12.4's measured **2.7 µs per OpenMP region** the remedy would
spend 0.151 ms to save at most 0.275 ms. **A remedy has to clear its own overhead.** Measured
after: 0.327 → 0.329 ms, untouched, as intended.

### 22.4 What §19 looks like after both of today's changes

| Coder-7B, 7.072 B active/token | tok/s | GB/s on the weight path | short of 50 |
|---|---|---|---|
| §12.2 / E7 as published this morning | 4.460 | 16.7 | 11.2× |
| after E8 | 6.37 | 22.5 | 7.8× |
| **after E9** | **6.79** | **24.0** | **7.4×** |

Delivered weight rate **48.0 G-weights/s**. At 0.5 B packed the same two changes take §12.2's
56 tok/s to **79.12**.

**§19.3 is untouched by both.** Engine work had **1.7–2.5×** in it; E8 and E9 have taken **1.52×**,
leaving roughly **1.15–1.65×** to the measured streaming ceilings (28 / 37.0 / 42 GB/s). The
missing **7.4×** remains a property of the model, not of the code.

### 22.5 The witness plateau at 7 B, published as an absolute

E8 §9 item 4 asked for this, because the within-sweep discard rule cannot see a contamination that
moves a whole sweep together. **Coder-7B packed, `--bench 300`, idle, `--mvacc 4`: `ffn~`
122.9–126.8 ms before E9, 114.5–117.0 ms after.** A future sweep reading materially above its
band is contended, whatever its own internal dispersion says. (0.5 B packed, for comparison:
11.552–12.083 ms at `--mvacc 1`, **7.478 ms** on today's canonical binary.)

## 23. AMENDED 2026-09-07 by E10 — §19's load-bearing assumption is MEASURED, and it is false

`probes/E10_PACKED_KERNEL_BINDER.md`, verdict `CORE-BOUND`. Pre-registered and pushed as
`briefs/BRIEF_E10_PACKED_KERNEL_BINDER.md` (`a0366d0`) before the bench existed. Closes E8 §9
owed item 2.

### 23.1 The measurement

Same `matvec` **source**, unmodified (`kbench.c` includes `donor_engine.c` with `main` renamed).
Row length `n_in` held **fixed at 3584**; only the footprint moves. 7 reps/cell, 6 threads,
`--mvacc 4`.

| footprint | packed GB/s | packed G-w/s | fp32 GB/s | fp32 G-w/s |
|---|---|---|---|---|
| 4 MB | 26.04 | 52.07 | **163.45** | 40.86 |
| 12 MB | 25.20 | 50.40 | 161.72 | 40.43 |
| 24 MB | 24.70 | 49.40 | 104.48 | 26.12 |
| 48 MB | 24.55 | 49.11 | 45.10 | 11.28 |
| 512 MB | 25.49 | 50.97 | 37.12 | 9.28 |
| 2048 MB | 26.48 | 52.95 | 38.58 | 9.65 |

**A 512× change in footprint moves the packed kernel by less than 8% and the fp32 kernel by 4.4×.**
Every packed cell is within ±4% of the arm's grand median, 25.49 GB/s.

Gates: **G-K0 planted control PASS at 4.40** (band ≥2.0) — the bench reproduced probe-3's 16 MB L3
cliff unprompted; **G-K1 = 1.022 → CORE-BOUND** (band ≤1.15); **G-K2 PASS 0.977** against the
engine's own 26.1 GB/s, which is also what excludes an elided loop. **G-K3 partly failed as
written** — the 4 MB cell's 30.6% spread is wider than the 15% gap to the boundary, so that ratio
alone cannot decide; the verdict rests on seven flat cells against a 4.4× fp32 contrast.

### 23.2 The engine is already on the ceiling

| | G-weights/s |
|---|---|
| engine `gate+up` (§22.3: 73.597 ms, 3.802 G weights) | **51.66** |
| engine `down` (§22.3: 36.403 ms, 1.901 G weights) | **52.22** |
| kernel bench, **every** footprint 4 MB → 2 GB | **49.11 – 52.95** |

**There is no streaming headroom left in this kernel.**

### 23.3 What is withdrawn from §19

§19 priced the remaining engine budget by assuming the weight organs are **bandwidth-bound** and
then asking what they would deliver at bandwidths this machine has been measured providing.
**§19.5 named that as the weakest link in its own derivation. E10 measures it, and it is false for
the packed kernel** — the kernel stops at ~25.5 GB/s with the data *in L3*, so feeding it faster
changes nothing.

| §19.4 row | status |
|---|---|
| 37.0 GB/s → 1094 M active weights at 50 tok/s | **WITHDRAWN for the packed path** |
| 42 GB/s → 1242 M | **WITHDRAWN for the packed path** |
| 28 GB/s → 665 M-ish | at the edge of, and above, what the kernel does |
| **~25.5 GB/s / ~53 G-w/s** | **the kernel's own ceiling; the engine is at 51.7–52.2** |

**The engine's remaining budget shrinks.** §19.3 had 1.7–2.5×; E8 and E9 took **1.52×**; §22.4 read
the remainder as 1.15–1.65×. **E10 says the weight path's share of that remainder is ~1.03×.**
Anything more requires **replacing the kernel**, not feeding it — and the rest must come from the
non-weight organs.

**§19.3's strategic conclusion is stronger, not weaker.** The missing 7.4× was already a property
of the model; it now has less engine cover than it had this morning.

### 23.4 The format's byte advantage is spent, and the kernel is not FMA-bound

**Cache-resident, packed is only 1.27× faster per weight than fp32** (52.07 vs 40.86 G-w/s) while
reading **8× fewer bytes**. Out of cache it is **5.6×** (52.07 vs 9.28) — which is exactly why the
packed format is right for the engine, and exactly why its *kernel* is now the thing in the way.

At 52 G-weights/s, 8 lanes/FMA, 6 cores, 3793 MHz, the loop issues **0.29 FMA per cycle per core
against a capability of 2 — about 14%**. Arithmetic on measured rates and the published clock; no
port model is asserted, and **E10 offers no mechanism for the remaining 7×.** The candidates named
for E11: each `vpshufb` in `.LBB15_11` produces 16 bytes of which the `vpmovsxbd` consumes 8, four
times per iteration; and the engine's own `--lut` path (`matvec_lut`, probe-1's int8-accumulate
kernel) has **never been measured at donor scale** and may have a different per-weight ceiling
altogether.

## 24. AMENDED 2026-09-07 by E11 — `--lut` is refuted as a speed lever, and §12's 1.05× is corrected

`probes/E11_LUT_KERNEL_CEILING.md`, verdict `NO-LIFT`. Pre-registered and pushed (`00d4446`) before
the arm was written. Closes E10 §6 owed item 1.

### 24.1 The measurement

E10's instrument, `n_in` 3584 fixed, 5 reps, third arm added that sets `m->tm` and lets the engine's
own `matvec` take the `--lut` branch.

| footprint | packed G-w/s | **lut G-w/s** | lut ÷ packed |
|---|---|---|---|
| 4 MB | 48.14 | **85.04** | **1.77** |
| 8 MB | 50.13 | **91.07** | **1.82** |
| 12 MB | 53.41 | **90.48** | **1.69** |
| 24 MB | 51.41 | 44.66 | 0.87 |
| 48 MB | 48.59 | 31.09 | 0.64 |
| **512 MB — the verdict cell, named in the brief** | 55.95 | **21.90** | **0.391** |
| 2048 MB | 60.25 | 21.42 | 0.36 |

**G-L1 = 0.391 against a ≤1.15 NO-LIFT boundary.** At donor-scale footprints the LUT kernel is
**~2.5× slower per weight** than packed.

### 24.2 §12's `--lut` numbers are corrected

§12 recorded `--lut` at **1.05×** over packed and INDEX §2.1 carried "`--fuse --lut` adds ~3.9%".
**Both were measured against a packed arm that still had E8's single accumulator**, while
`matvec_lut` already used four accumulators and was never touched by E8. **Post-E8 the ordering is
reversed and the margin is not small.** The §12 rows are not withdrawn — they were correct for the
engine of the day — but they must not be read as current, and the 3.9% line is superseded.

### 24.3 The kernels swap places at exactly the L3 boundary

**The LUT kernel is the fastest weight kernel this programme has measured — 91 G-weights/s — and
only while L3-resident.** It falls **4.2×** across 16 MB; the packed kernel (E10) does not move
across it at all.

The mechanism is the layout, not the arithmetic. `matvec_lut` reads `codes + t*Mpad + base`, so
consecutive `t` are `Mpad` apart: **2.3 KB at the 4 MB cell, 299 KB at 512 MB** — 1,792 reads of 32
bytes across as many pages. **Tile-major is what makes one `vpshufb` serve 32 rows and it is the
same thing that destroys the stream.**

**Exploratory — not in the pre-registered bands**, reported because it is large, present in both
runs, and specific.

### 24.4 What it does and does not license

**It does not move the goal**: §23.3 stands, the packed kernel remains the weight path with ~1.03×
of headroom, and Coder-7B is 7.4× short of 50 tok/s.

**It does bear on `SCALEUP_ARCHITECTURE`**, which specifies a cache-resident keystone **≤16 MB L3**
with experts streamed from DRAM — the boundary probe-3 measured. **A kernel 1.8× faster inside it
and 2.5× slower outside it is shaped for that split.** Before that becomes a plan it owes an
end-to-end `--lut` rate with the parity gate (Phase 60: a kernel result never composes to a system
claim; Phase 61: a microbench does not compose to an engine) and the activation-quantization cost on
**real** activations — **1.40e-01** whole-vector, **3.10e-02** at G=32, which is not a rounding
error.

### 24.5 Two instrument failures, both recorded

**Run 1 was VOID.** A concurrent worker in a different checkout streamed the 30.46 GB fp32 donor
across six threads; the packed arm acquired a slope it does not have (35.6 → 61.4 where E10 measured
it flat). **A within-run rule could not have caught it — every cell moved together.** What caught it
was a number carried in from a previous session, compared from outside the run: E8 §9 item 4's
absolute plateau, in its first real use.

**G-L2 then failed in run 2 by mis-specification, not contamination.** The 49–53 G-w/s band was
drawn from E10's 7-rep sweep while E10's *own* 3-rep sweep read 57.11 / 60.78 at the two largest
cells; run 2's 55.95 / 60.25 is inside E10's between-run range and outside a band written from half
of it. **A known-positive band must be drawn from every reading of the known-positive.** Fourth
pre-registered rule in this programme aimed at the wrong number — after E8's discard rule, E10's
G-K3, and this.

---

## §25 — E13: the LUT collapse was the layout. `LAYOUT-CONFIRMED`.

Brief `89fd63e` pushed before the arm existed; arm `076381a`; probe
`probes/E13_BLOCKED_TILE_MAJOR.md`. **Closes E11 §7 item 1 and the 1.46× E10 left unclaimed.**

### 25.1 The row E10 had not read in its own table

| E10, 2048 MB, DRAM-resident, 6 threads | moved GB/s |
|---|---|
| fp32 arm | **38.58** |
| packed arm | **26.48** |

Same instrument, same threads, **same moved-byte convention on both sides**. The packed kernel was
at **69%** of a streaming rate that sweep had just demonstrated, so §23's `CORE-BOUND` left
**1.46× measured** rather than nothing. That is what E13 went after.

### 25.2 The defect

`build_tm` stored `tm[t*Mpad + o]` and `matvec_lut` walks `t` with `base` fixed, so consecutive
32-byte reads sat `Mpad` apart: **18.5 KB** at `gate|up`, **3.5 KB** at `down`, **299 KB** at the
512 MB cell — a different page nearly every read, 32 bytes used of every 64-byte line. Total bytes
read were **identical** to the packed arm's. Purely an ordering defect.

`--lutblk` stores each 32-row tile contiguously. Same bytes permuted; `t` order and accumulate tree
untouched, so **bit-identical** and gated on sha256.

### 25.3 The sweep, `n_in` 3584 fixed, 5 reps

| footprint | packed | `lut` | **`lutblk`** | blk ÷ lut | blk ÷ packed |
|---|---|---|---|---|---|
| 4 MB | 51.40 | 83.80 | **87.84** | 1.05 | 1.71 |
| 8 MB | 50.22 | 75.50 | **97.87** | 1.30 | 1.95 |
| 12 MB | 51.59 | 86.55 | **103.05** | 1.19 | **2.00** |
| 24 MB | 55.32 | 63.77 | **101.22** | 1.59 | 1.83 |
| 48 MB | 48.34 | 34.90 | **73.11** | 2.09 | 1.51 |
| **512 MB (verdict cell)** | 56.72 | 22.26 | **67.51** | **3.03** | **1.19** |
| 2048 MB | 59.02 | 20.84 | **71.82** | **3.45** | 1.22 |

G-w/s, medians of 5. **The `lut` 8 MB cell carried 100.8% spread; its median is not quotable** and
is recorded rather than dropped. **G-M1 = 67.51 against a ≥65 boundary → `LAYOUT-CONFIRMED`, and
the brief predicted 65–77.** Second prediction in this programme to land inside its own band, and
derived the same way as E9's: a measured quantity over a structural factor.

**Blocking recovers 3.0–3.5× at the two largest cells. The 4.2× L3 cliff becomes 1.53×.
103.05 G-w/s at 12 MB is the fastest weight kernel this programme has measured.** At 512 MB
`lutblk` moves **33.76 GB/s** against the fp32 arm's **34.75** in the same sweep — **97%**.
**§25.1's 1.46× has been taken; ~3% remains at that footprint.**

### 25.4 It composes — with the right cell

| Coder-7B, `--bench 100`, 5 interleaved reps | median tok/s | spread | `ffn~` |
|---|---|---|---|
| **`--lutblk`** | **9.21** | 3.7% | 85.7–90.1 ms |
| `--lut` | 3.78 | 9.0% | 175.5–184.8 ms |
| packed (default) | 6.78 | 9.1% | 115.8–128.3 ms |

**1.358× at donor scale.** `--lut` as shipped is **0.557×** — 1.8× slower through the engine, which
**confirms E11's `NO-LIFT` end-to-end** rather than softening it.

**Baseline reproduces to 0.15%**: packed 6.78 against E9's 6.79, `ffn~` 115.8/117.3/117.4 on reps
3–5 against E9's published plateau 114.5–117.0 (reps 1–2 at 128.3/125.9 were warm-up).

Coder-7B's fused organs are `gate|up` **67.9 MB** and `down` **33.9 MB** — the 24–48 MB band, where
the sweep gives 1.83 and 1.51. Amdahl on a ~70–80% weight path predicts 1.3–1.4×; measured 1.358×.
**The 512 MB cell's 1.19× was never the engine's number.**

At 0.5 B, 7 interleaved reps: `lutblk` **91.31** vs packed **75.03** vs `lut` **59.72** =
**1.217×**. A first 3-rep pass gave `lutblk` a **180% spread** and is discarded as warm-up —
**three reps were not enough at that shape**, and it is recorded because it is the exact shape the
≥3-rep rule exists to catch.

### 25.5 What it does NOT license

**`--lutblk` is not shippable on this.** The LUT path quantizes activations to int8 at a measured
**1.40e-01** relative L2 whole-vector (**3.10e-02** at G=32) and **that has never been carried to
BPB or greedy parity on the donor**. G-M0's sha256 identity is `--lut` vs `--lutblk` — the two LUT
*layouts* agreeing with each other — and says nothing about either agreeing with the packed path,
which by construction they do not. **E13 makes the LUT path fast enough to be worth the quality
question; it does not answer it. The packed default stays the default.**

**E11's `NO-LIFT` is explained, not overturned.** It was a correct verdict on the layout as
shipped, its §12/INDEX §2.1 corrections stand, and its §3 mechanism paragraph — the one thing it
asserted without testing — is now **tested and holds**.

### 25.6 Where that leaves the goal

| Coder-7B, 7.072 B active/token | tok/s | short of 50 | path |
|---|---|---|---|
| E7 | 4.460 | 11.2× | exact |
| E8 | 6.37 | 7.8× | parity-gated |
| E9 | 6.79 | 7.4× | bit-exact |
| **E13 `--lutblk`** | **9.21** | **5.4×** | **LOSSY, quality unmeasured** |

**The honest statement is 6.79 tok/s exact, with a 1.358× lever available on a lossy path.**
2.065× in a day if the lever holds; 1.52× if it does not.

**No third kernel to write.** The weight path is at 97% of this machine's demonstrated streaming
rate. **§19.3 stands and is better supported than this morning: the remaining 5.4× is a property of
the model.** The levers left are fewer active weights per token and residency — **E12**.

---
## §26 — E12: the exporter's default rule is at chance at every scale tested. `CHANCE-LINE`.

Brief `11a9d83`, runner `87ba957`, §9 amendment + Z-dispersion diagnostic `ae403c2`, probe
`probes/E12_TERNARY_COST_VS_SCALE.md`. Sweep 2161 s, 24×512 heldout, all layers, 6 threads, idle.

**This section replaces an earlier §26 that returned `CONTROL-FAILED`. That verdict is withdrawn:
it rested on a control this programme had already retired. See §26.4.**

**This is the quality counterpart to §23/§25.** E10 and E13 closed the engine side — the weight path
runs at 97% of this machine's demonstrated streaming rate and there is no third kernel to write. So
the remaining 5.4× to 50 tok/s must come from **fewer active weights per token at ternary
precision**, and that road rests on a quality number measured at **one shape** (T2 = Qwen2.5-1.5B).
E12 went to draw the curve. It came back with something more basic.

### 26.1 The chance line

`log2(V) / (bytes per token)` = `log2(151936) / 4.229452` = **4.069819 BPB** — what a model scores by
assigning every token equal probability. The padded config vocab is the conservative choice; on the
151,665 tokens the tokenizer can emit it is 4.069210, and nothing here turns on the difference.

| donor | BPB fp32 | BPB ternary (FA) | **dBPB** | FA vs chance line |
|---|---|---|---|---|
| 0.5 B | 0.871795121 | 4.587452739 | **3.715657618** | **+0.518 above** |
| 1.5 B | 0.767594958 | 5.505834291 | **4.738239332** | **+1.436 above** |
| 3 B | 0.724449797 | 5.734699003 | **5.010249206** | **+1.665 above** |

Each donor against **its own** fp32 baseline; rule **imported** from `t1_ternarize.ternarize`, which
is what `qwen_export.py` calls. **Correction, after this section was first written: `R0` is that
script's flag DEFAULT but NOT the rule this programme ships** — every standing engine artifact was
exported with `--rule R3` (`e1_bpb_through_engine.py:395`). **E12 measured a rule the pipeline does
not use.** No number changes; what changes is what the verdict is about. And E1 already holds two
thirds of the right sweep, on the same slice and so the same chance line: **R3 reads 4.509164 /
4.531234 at 0.5 B (+0.44 / +0.46 ABOVE chance) and 3.484253 / 3.475707 at 1.5 B (−0.59 BELOW it)** —
**the shipped rule crosses the chance line between 0.5 B and 1.5 B, and the crossing is unlocated.**
The missing cell is 3 B, exactly where `R0` inverted.

**Every donor's fp32 baseline is far below the line (0.72–0.87). Every ternarized arm is above it,
at all three scales.** The donors and the slice are fine; the conversion is what puts them past
chance.

### 26.2 Why the pre-registered ratio cannot be read

`r = dBPB(3B)/dBPB(0.5B) = 1.3484` reads `COST-GROWS` mechanically. **It is not reported as a
result.** A model above the chance line assigns the truth *less* probability than knowing nothing
would; differences between two such models measure how confidently wrong each is, and have no reason
to order by damage. The data shows exactly that, **above the line and deterministically**:

| | 0.5 B | 1.5 B | 3 B |
|---|---|---|---|
| `F` (FFN only) | 4.546298 | 4.076694 | **5.928395** |
| `FA` (FFN **and** attention) | 4.587453 | 5.505834 | **5.734699** |
| more organs → more damage? | yes | yes | **no** |

At 3 B, converting **more** organs did **less** damage by 0.194 BPB. Both arms are deterministic.

**`COST-SHRINKS` and `COST-GROWS` both presuppose that dBPB measures cost. At `R0` it does not.**

### 26.3 What E12 does establish

1. **`R0` destroys every donor tested, worse as scale grows**: +0.518 / +1.436 / +1.665 past chance.
   **No donor at any tested scale survives the exporter's default rule.**
2. **The instrument is sound.** `I − base = +0.000e+00` **exactly at all three cells**, over
   168/196/252 substituted tensors; counts structurally correct (7/layer FA, 3/layer F at 24/28/36
   layers); tokenizers **verified identical** (1 fingerprint — the shared disk-cached slice requires
   it); fp32 baselines monotone in scale. **The BPB numbers are sound.**
3. **T2's §3 table re-reads against the line, and its §4(a) mechanism does not survive:**

   | T2 arm | BPB | vs chance line |
   |---|---|---|
   | **R0 — the flag default E12 measured** | 4.076694 | **+0.007 ABOVE** |
   | Z (random signs) | 4.140276 | **+0.070 ABOVE** |
   | R4 (GPTQ on R0's grid) | 4.299819 | **+0.230 ABOVE** |
   | R1 (TWN) | 3.851979 | −0.218 below |
   | R2 (scale search) | 3.390467 | −0.679 below |
   | **R3 (activation-weighted)** | 2.476967 | **−1.593 below** |
   | **R5 (R3 + GPTQ)** | 2.027495 | **−2.042 below** |

   T2 §4(a) read `R0 − Z = −0.064 ± 0.126` as "BitLinear158's choice of which sign carries no
   measurable information about the donor". **That is a contrast between two points both pinned at
   chance, and no such contrast can resolve anything.** The signs are `sign(w)` under every rule in
   that table — a positive per-row scale cannot change them — and R3/R5 carry the identical signs to
   1.6–2.0 BPB *below* the line. **T2's `RULE-HELPS` verdict is untouched; the mechanism §4(a)
   claimed is withdrawn.** And it says what T2's 62% actually bought: not a cheaper conversion of the
   same kind, but **the difference between a model at chance and a model that predicts**.

### 26.4 The control that was retired, and the one that passed

E12 adopted arm `Z` (`random_sign`, same organs as `F`) as its planted control and read it as failing
at 3 B (`Z − F = −1.490`). **Three things were available before the run and I did not carry them in:**

1. **T2 had already retired arm `Z`**, in terms: *"it is why arm Z was the wrong control: the brief
   assumed Z would be far worse than the treatment, and it is not worse at all"* (T2 §4(a)).
2. **The same amendment, dated 2026-09-04, sits in the docstring of `t1_ternarize.ternarize` — the
   function E12 imports** — and names **arm `I`** as its replacement. `t1_ternarize.ARMS` labels its
   own `Z` row `"PLANTED CONTROL (mis-specified, see report)"`.
3. **E12's `Z` does not reproduce T2's `Z` at the shared cell**, while `F` reproduces to six decimals:

   | source | `F` / `R0` | `Z` |
   |---|---|---|
   | T2 (`t2_rules.py`, seeds `1000 + stats["n"]`) | **+3.309099** | **+3.372681** |
   | E12 (`t1_ternarize.apply_arm`, seeds `1000 + rng`) | **+3.309099** | **+4.001257** |

   An **off-by-one in which per-tensor seed lands on which tensor** moves `Z` by **0.628 BPB =
   126 σ_seed**.

**Arm `I` — the control this programme actually sanctions — passed exactly, at every cell.**

**`ternary/e12_zvar.py`, 2351 s.** `dBPB` of arm `Z`, five draws per cell:

| cell | `Z` draws (dBPB), 5 seed bases | spread | mean | `F` | draws with **`Z` < `F`** (control fails) |
|---|---|---|---|---|---|
| 0.5 B | 3.601252 … 4.465219 | 0.863967 | 3.891365 | 3.674503 | **2 / 5** |
| 1.5 B | 3.463229 … 4.042358 | 0.579129 | 3.793314 | 3.309099 | 0 / 5 |
| **3 B** | 3.714379 … 5.410317 | **1.695938** | 4.524840 | 5.203945 | **4 / 5** |

**G-Z0 passed at all three cells** — `seed_base = 1000` reproduced `e12_scale.json`'s `dBPB_Z` to the
last digit (3.714379009 at 3 B), so the `seed_base` parameter left the default path untouched.

**G-Z1: `spread(3B) = 1.695938 ≥ 1.490` → `Z-UNSTABLE`. The §9.3 prediction landed** — the third
band in this programme to contain its own result, after E9 and E13, and derived the same way.

Three readings, in order of what they license:

1. **The 3 B spread (1.696) is larger than the entire `Z − F` gap (1.490) that `CONTROL-FAILED` was
   read off.** A single draw of arm `Z` could not have decided that cell in either direction.
2. **At 0.5 B the control's outcome flips with the seed** — 2 of 5 draws put `Z` below `F`. The cell
   the first write-up recorded as "fires, +0.019" fires on three seeds and fails on two.
3. **At 1.5 B it fires 5/5**, which is why T2's shape looked healthy. T2's own `Z` (`+3.372681`)
   sits 0.091 below the five-draw minimum — consistent with being a sixth draw, so §4's off-by-one
   explanation holds and no further implementation difference needs to be posited.

**What this does NOT say.** At 3 B, 4 of 5 draws beat `F`, and the draw mean (4.524840) is 0.679
below `F`. **That is not noise**: under this comparator the real rule really does tend to score worse
than random signs at 3 B. **It still licenses no claim about damage**, because `F` and every `Z` draw
at that cell sit above the chance line — `F` at 5.928 BPB, the `Z` draws at 4.439–6.135, against
4.070. **The dispersion disqualifies reading any single draw as a gate; the chance line disqualifies
the comparison itself.** §2 is the reason the verdict changed; §4.1 is only the reason the original
gate could not have worked either way.


### 26.5 Rules

**A control must be checked as still sanctioned, not merely labelled.** The planted-control law says
an instrument must fire on a known-positive before its nulls count. It does not say that an arm named
"planted control" *is* one. Here the module being imported said so in three places.

**A stochastic comparator needs its own dispersion before a single reading of it decides anything.**
Same class of error as quoting a contended timing — one draw was published as a verdict.

**A control that passes on a smoke is not thereby a control.** The 0.5 B smoke showed `Z − F =
+0.147`; the full 24×512 sweep showed **+0.019**. Full-scale margin is not predictable from smoke.

**Read a converted model's BPB against `log2(V)/bpt` before concluding anything.** The constant costs
one line and is computable from numbers already in every result file. **Both anomalies that sent me
chasing an instrument bug are explained by it.**

### 26.6 Owed

1. **Finish the R3 sweep — the 3 B cell.** Two thirds exists: E1 has the shipped rule at 0.5 B
   (above chance) and 1.5 B (below it), so **it crosses the line somewhere between, unlocated**.
   The missing cell is 3 B, exactly where `R0` inverted. **Until it runs, this programme has no
   measurement of how the SHIPPED conversion scales — only of how `R0` fails.**
2. **Re-read T2b's organ policy** against the `F` > `FA` inversion and against the chance line: if
   T2b's arms sit above it, its organ ranking faces the same objection.
3. **T3's rotation across scale** (`7cdeca8`), for the same reason as (1).
4. **A diagnostic for the 3 B inversion** — per-tensor scale distributions and activation magnitudes
   through the ternarized 3 B FFN against 1.5 B — but **after** (1), since it may be an artefact of
   `R0` specifically.

**§19.3 is unchanged and unrelieved: the remaining 5.4× is a property of the model. What E12 removes
is the belief that this programme had measured the cost of getting it — it had measured one rule
failing.**

---

## §27 — E14: the lever's quality cost, measured. `CHEAP-BUT-NOT-NEUTRAL`.

**§25/§26 left the 1.358× lever with an explicitly unmeasured quality cost.** `--lutblk` runs the
LUT path's int8 activations at a measured **1.40e-01** relative L2 (E11), and no probe had ever
carried that to BPB or to greedy on a donor. E14 carries it, at two scales, through the engine.

`probes/E14_ACTIVATION_INT8_COST.md`. Pre-registered `a6a6607`, §10 amendment `47184ca` pushed
after reading the fp32 baseline and **before any treatment arm ran**.

### 27.1 The verdict cell — 1.5 B, sitting 0.625503 BELOW the chance line

| arm | activations | BPB | `dBPB` from fp32 | greedy vs fp32-act |
|---|---|---|---|---|
| **A0** | fp32 (shipped) | 3.446375376 | — | — |
| **A1** | int8, one scale/vector | 3.429414114 | **−0.016961** | **45.6%** (73/160) |
| **A2** | int8, one scale/32 ch | 3.440384359 | **−0.005991** | **64.4%** (103/160) |
| **A3** | int8, E13 blocked layout | 3.429414114 | −0.016961 | 45.6%, identical to A1 |

**Int8 activations LOWER bits-per-byte, at a cell where BPB means what it usually means, while
changing the top-1 token on 54% of positions.** BPB scores; greedy ranks; noise that softens an
over-confident logit vector improves the first without improving the second.

**The registered band returns `ACTIVATION-CHEAP` and the label does not survive.** §4's bands are
one-sided — drawn for a *cost* — so a negative delta falls into `CHEAP` by construction. Under
`|dBPB|` the whole-vector arm reads `MARGINAL` (0.016961 = 3.4 σ_seed) instead.

### 27.2 What this does to the lever — it does NOT release it

**The composition is crossed.** E13 measured **1.358×** with `--lutblk` alone, which G-N0 and G-N3
both show is arm **A3 = A1** — the arm with the *larger* quality move (greedy 45.6%). The arm that
is quality-cheap, **A2 (group-32), has no speed number at all**.

| arm | quality (E14, 1.5 B) | speed (E13) |
|---|---|---|
| A1 / A3, whole-vector | `\|dBPB\|` 0.016961, greedy **45.6%** | **1.358× measured** |
| A2, group-32 | `\|dBPB\|` 0.005991, greedy **64.4%** | **never measured** |

`--lutblk` and `--lut-group` are orthogonal flags in `donor_engine.c`, so `--lutblk --lut-group 32`
is reachable — but **Phase 61's law applies exactly here**: a kernel that gains 1.358× applying one
scale per vector does not thereby gain it applying one per 32 channels, and the extra scales land
in the inner loop. **Nothing licenses carrying 1.358× across to A2.**

**So the quoted rate is unchanged: 6.79 tok/s exact on the real 7.072 B, packed remains the
default, and the 1.358× lever remains unspent** — now for a *measured* reason rather than an
unmeasured one.

### 27.3 What it confirms, and what it costs the pre-registration

- **E13's bit-identity now holds three independent ways**: sha256 over 311 MB of logits (E13),
  `A3 − A1 = +0.000e+00` at both cells (G-N0), and **token-for-token identical greedy trajectories
  across 160 tokens** (G-N3). `--lutblk` is a permutation of a permutation.
- **G-N1 does not fire at either cell** (`A2 − A1` = +0.300872 at 0.5 B, +0.010970 at 1.5 B) and
  fires at both in repaired form, `|A2−A0| < |A1−A0|`, separating **24×** and **2.83×** with the
  more accurate arm nearer the reference — the direction E11's 4.5× input-error ratio predicts.
  **The harness was never at fault; the direction written into the gate was.**
- **Two of three gates assumed the answer's shape.** G-N1 assumed a *direction* that holds only
  while the model predicts; G-N2 assumed a *sign* that holds only while quantization damages.
  **Fifth pre-registered rule in this programme aimed at the wrong number**, and the second inside
  one experiment.
- **A protocol defect, recorded not smoothed.** E14 passed no `--seqlen`, and
  `donor_engine.c:1249` defaults `SL = n`, so every arm was scored as **one 12,288-token
  sequence**, not 24 documents of 512 — which is why A0 reads 4.629292 at 0.5 B where E1 reads
  4.531234 on the identical file. The arm-to-arm gates are unaffected; **a `--seqlen 512` re-run is
  owed** and until it lands these are 12 k-context numbers.

### 27.4 Owed

1. **A speed number for `--lutblk --lut-group 32`** at E13's 24–48 MB organ band. Until it exists
   the quality-cheap arm and the fast arm are different arms.
2. **The `--seqlen 512` re-run**, both cells, ~3 h.
3. **A properly banded ranking gate.** Brief §6 forbade promoting G-N3 to a gate, correctly — no
   band for it could be justified from anything measured. The band is still missing, and it is the
   prerequisite for any future verdict that claims a conversion is harmless.

**§19.3 is unchanged: the remaining gap is a property of the model. What E14 removes is the
possibility of quietly buying 1.358× with a metric that was moving for the wrong reason.**

---

## §28 — E15: the artifact every donor rate is quoted on does not predict. `DOES-NOT-PREDICT`.

Pre-registered `7c4243f`, amended `c92ffd4`, both pushed before any arm ran.
Probe: `probes/E15_DOES_THE_7B_PREDICT.md`. Runner: `engine/e15_donor7b_bpb.py`.

**No timing was taken in E15. No rate in §§12–27 moves. What moves is what the rate may be called.**

### 28.1 The measurement

Density `heldout`, 24×512, `ids_sha256 a1a48dc9fc5a6dc1`, 51,870 scored bytes, 4.229452 bytes/token,
`--seqlen 512` passed explicitly, `--threads 6`. Chance line `log2(152064)/4.229452` = **4.070106**,
band **0.000896** (the padded-vs-emittable vocab ambiguity, derived per E8 §3).

| arm | what it is | BPB | vs chance | gate |
|---|---|---|---|---|
| **B0** | `qwen25-coder7b_f32.bin`, fp32 — **planted control** | **0.674026555** | **−3.396080** | **G-Q0 FIRES** (< 1.000) |
| **B1** | `qwen25-coder7b_p.bin`, packed `R0` + `fold layers` + `--head-ternary` | **5.299200075** | **+1.229094** | **G-Q1 `DOES-NOT-PREDICT`** |
| G-Q2 | `B1 − B0` | +4.625173520 | — | descriptive only (E12 §2) |

**+1.229094 against a band of 0.000896 is 1,372× the band.** No vocabulary convention, slice choice
or protocol detail comes within three orders of magnitude of closing it.

### 28.2 What this does to the ledger's headline rate

**The two halves of the E7 headline are about different artifacts.**

| artifact | greedy vs PyTorch | BPB | rate |
|---|---|---|---|
| `qwen25-coder7b_f32.bin` | **160/160** (E7 §2) | **0.674027** — predicts | fp32 path |
| `qwen25-coder7b_p.bin` | **0/160**, diverges at token 0, 5/5 prompts (E7 §5) | **5.299200** — worse than uniform | **6.79 tok/s** (§§12–19) |

E7 §5 already said so — "R0 at 7 B collapses completely, and how much of that is the rule versus the
scale is a separate experiment nobody has run" — and reported no BPB deliberately. **E15 ran it, and
the answer is that the artifact is past the chance line.** What happened in between is that 4.46 and
then 6.79 tok/s went on being quoted as *the rate on the real donor* while that sentence stayed true
and unmeasured for a month.

**Every engine number stands, and the reason is structural: a ternary weight costs the same
bandwidth whatever scale multiplies it.** The kernel reads the same bytes in the same order and does
the same work; the per-row scale is one multiply at the end. So **6.79 tok/s exact, 24.0 GB/s,
48.0 G-weights/s, E8's 1.349×, E9's 1.056×, E10's `CORE-BOUND`, E11's `NO-LIFT`, E13's 1.358× lever
and E14's `CHEAP-BUT-NOT-NEUTRAL` are all unaffected.**

**§19.3 is unchanged. What §28 removes is the phrase "on a working 7 B".**

### 28.3 The donor is fine, and it is the best one here

| donor | fp32 BPB | chance | vs chance | source |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.871795 | 4.069819 | −3.198 | E12 §1 (PyTorch) |
| Qwen2.5-1.5B | 0.767595 | 4.069819 | −3.302 | E12 §1 (PyTorch) |
| Qwen2.5-3B | 0.724450 | 4.069819 | −3.345 | E12 §1 (PyTorch) |
| **Qwen2.5-Coder-7B** | **0.674027** | 4.070106 | **−3.396** | **E15 (engine)** |

**Monotone in scale, and the 7 B is the best model this programme has ever run** — on a
prose-majority corpus (40% pg19 / 25% markdown / 25% python / 10% wikitext), with a code-specialised
donor. The two conventions in the last column are interchangeable here: E1 §2.1 measured engine
against PyTorch on the identical 0.5 B arm at `+1.5347e-05`. **This is also the first evidence
bearing on the open "pull a real ~10 B" question: bigger donors keep paying, in fp32.**

### 28.4 The format's speed value, stated as what it is

E7's `--stage generate` ran both arms in the same run under the same conditions: fp32
**1.382–1.392 tok/s**, packed **4.853–4.888** — a **3.51× ratio**. Those are **generate-stage
figures, not witnessed `--bench` rates**; E7 §11's contention witness postdates them and validates
nothing retroactively, so the **ratio** is the durable part and the absolutes are not quoted.

**This programme has a 7 B that predicts and a 7 B that is 3.51× faster, and they are not the same
7 B. Making them the same artifact is now the binding problem, and it is not an engine problem.**

### 28.5 Predictions, scored — and the amendment made it worse

| prediction | registered | measured | outcome |
|---|---|---|---|
| B0 fp32 control | **0.55 – 0.85** | 0.674027 | **INSIDE** |
| B1, first draft (withdrawn) | above chance, 4.6 – 6.5 | 5.299200 | would have been **INSIDE** |
| B1, amended (the one that stood) | 3.4 – 5.2, direction **not** called | 5.299200 | **MISSED by 0.099** |

The first draft reasoned from E12's unfolded `R0` arms and landed. The amendment composed E2's fold
credit (−0.529, measured at **0.5 B under R3**), T2's rule penalty (+1.600, measured at **1.5 B,
FFN-only, fold none**) and a scale term — **three different configurations**. The brief named that
"the exact move E12 was written to forbid" and did it anyway to set a bound, and **it degraded the
estimate**: the fold credit measured under R3 did not carry to R0 at 7 B.

**Rule, and it is E12's arriving from the other direction: composing measured deltas across
configurations is not conservative just because each delta is measured.** The costliest half was
refusing to call the direction — E12's R0 sweep and E7's 0/160 both supported the call that was
declined.

### 28.6 Owed — B2, and it is now the programme's most consequential open measurement

1. **B2 — re-export the 7 B with `--rule R3 --calib-seqs 32`, keeping `--fold layers` and
   `--head-ternary` so the RULE is the only variable against B1**, and re-measure on this slice.
   **This is E7 §12 owed item 3, promoted from optional to load-bearing.** R3 is what
   `e1_bpb_through_engine.py:395` ships for every other standing artifact, and **it works at 1.5 B**:
   E1 reads `TQH` at **3.475707, 0.594 BELOW** the line. A ternary 7 B that predicts is not ruled out
   by anything measured — it has never been built. Cost: R3 refuses the bf16 loader
   (`qwen_export.py:243`), so the calibration pass runs under a float32 load at ~30 GB resident;
   plus a 5.7 GB write and a ~31 min BPB run. **~2–3 h, one heavy job, run alone.**
2. **B3 — the same export with `--fold none`**, which separates the two axes E15 could not: B1
   differs from the standing 0.5 B/1.5 B artifacts on **both** rule and fold.
3. **E12 §26.6 item 1 (the R3 sweep's 3 B cell) is now the same question one scale down** and should
   be read together with B2.

**If B2 also lands above the line, no rule in this exporter produces a working 7 B**, the binding
constraint moves from the rule to the format, and the case for **training into the format rather
than converting into it** — `SCALEUP_ARCHITECTURE`'s premise — stops being a preference and becomes
the measured conclusion.

---

## §29 — E16: the rule was most of it, and fixing it still does not predict. `SCORE-CROSSES-RANK-DOES-NOT`.

Pre-registered `be65920`, runner `2247923`, control `34e7ba3`, all pushed before the arms ran.
Probe: `probes/E16_R3_AT_7B.md`.

**No timing was taken in E16. No rate in §§12–28 moves.**

### 29.1 The measurement

Same slice, protocol and binary as E15: heldout 24×512, `--seqlen 512`, `--threads 6`, chance
**4.070106** (V=152064) band **0.000896**; Qwen2.5 chance 4.069819.

| arm | rule | fold | BPB | vs chance | greedy vs fp32 |
|---|---|---|---|---|---|
| **C0** Qwen2.5-1.5B, planted control | R3 | none | 3.475706372 | −0.594113 | — |
| **B1** E15's shipped artifact | R0 | layers | 5.299200075 | +1.229094 | **0/160** |
| **B2** | **R3** | layers | **4.017232598** | **−0.052874** | **0/160** |
| **B3** | R3 | none | 4.168325483 | +0.098219 | **0/160** |
| fp32 (E15's B0) | — | — | 0.674026555 | −3.396080 | 160/160 |

- **G-R0 FIRES** — C0 reproduces E1's `3.475706691632780` to **3.20e-07** (tol 0.01, predicted
  `<1e-4`).
- **G-R1 `RULE-FIXES-IT`**, B2 0.052874 below the line = 59× the band.
- **G-R2 = −1.281967477** — the rule's worth at 7 B with everything else fixed, and a damage figure
  rather than a distance because B2 is below the line (E12 §2).
- **G-R3: B3 is +0.098219 ABOVE**; `B3 − C0 = +0.692618791`.
- **G-R4 = −0.151092884** — the fold, and the entire reason B2 is below the line where B3 is above.

### 29.2 The two metrics disagree, and that is the result

B2 is **0.155 nats/token** better than uniform (11.777051 vs `ln(152064)` = 11.932057) — 59× the
band, so it carries real information — **and agrees with its own fp32 donor on 0 of 160 greedy
tokens, diverging at token 0**, identical to E7's planted control.

**`RULE-FIXES-IT` is true of the band and false of the model.** E14's law, written one day earlier
against a different experiment — *pair every scoring metric with a ranking one* — is what caught
it, and **E16's brief registered only scoring metrics**. The ranking run (`e16_greedy_rank.py`) was
added afterwards and **reports rather than decides**, per E14 §6. The registered gate set would have
announced that the rule fixes a model which never once picks the donor's next token. It validates
itself first: B1 reproduces E7's published `0/160` against E7's own stored fp32 continuations.

### 29.3 R3's damage grows with scale too

| donor | R3, fold none | vs chance |
|---|---|---|
| 0.5 B | 4.531234 | +0.461415 |
| 1.5 B | 3.475707 | −0.594112 |
| **7 B** | **4.168325** | **+0.098219** |

**Non-monotone: above, well below, back above.** E12 found `R0`'s damage grows with scale; `R3`'s
does too — it starts from a better place and takes one more octave to show it. The 1.5 B → 7 B step
crosses model families, so `+0.692619` is not a pure scale term and is not reported as one.

**The fold is load-bearing and shrinking**: −0.529 (0.5 B, E2), −0.220 (1.5 B, T3), **−0.151
(7 B, E16)**. E15's B1 was a wrong rule over a *correct* fold — and this is by how much E15's
amended prediction was wrong to import the 0.5 B credit.

### 29.4 Predictions, scored

| gate | registered | measured | outcome |
|---|---|---|---|
| G-R0 | `< 1e-4` | 3.20e-07 | **INSIDE** |
| G-R1 | below the line, 2.3–4.0 | 4.017233 | direction RIGHT, band missed by 0.018 |
| G-R2 | −3.0 to −1.3 | −1.281967 | **the same constraint as G-R1**, one miss not two |
| G-R3 | below the line, 2.5–4.0 | 4.168325 | **DIRECTION WRONG** |
| export | 1–3 h | 0.51 h, 0.46 h | below the band |

**G-R3 is the real failure and the reasoning failed before the band did**: a trend extrapolated
from two points in the same configuration. The brief defended that as legitimate against E15's
cross-configuration composition, and noted in the same paragraph that two points cannot establish
curvature — then relied on them. **Rule: a trend read from two points is a direction, not a law.**
What worked was calling directions at all; E15's post-mortem said refusing to was the costlier half.

### 29.5 Where it leaves the goal

**§19.3 is unchanged and 6.79 tok/s exact stands.** The engine is at 97% of this machine's
demonstrated streaming rate on the weight path with the kernel on its own ceiling at every
footprint; the donor is excellent (0.674027 fp32, monotone in scale); **the conversion destroys it,
the rule was most of the destruction, the fold is load-bearing, and what remains after fixing both
is still at chance.**

E15's conditional — *"if B2 also lands above the line, the binding constraint moves from the rule to
the format"* — did not fire as written, because B2 landed 0.053 **below**. **The greedy result
reaches the same place by another route.** With the honest qualifiers that only two of four rules
were tested and no clean scale axis was available, **the binding constraint at this scale is the
format, not the rule**, and *training into the format rather than converting into it* becomes a
measured conclusion rather than a preference.

### 29.6 Owed

1. **A ranking band** — E14 §5 item 3, now twice load-bearing. Without it `0/160` and `45.6%` are
   observations that cannot convict.
2. **`R1`/`R2` at 7 B**, if a cheap 1.5 B screen suggests either beats R3 there.
3. **Qwen2.5-7B** for a clean scale axis (~15 GB), so `+0.692619` can be attributed.
4. **The 3 B cell of the R3 sweep** (E12 §26.6 item 1) — the point between R3's minimum at 1.5 B and
   its return to the line at 7 B.
5. **A fold sweep at 3 B**, to test whether −0.529 / −0.220 / −0.151 is smooth.

---

## §30 — E17: the head is not the mechanism, and BPB has no resolution where we ship. `HEAD-IS-NOT-THE-MECHANISM`.

Pre-registered `cf33251`, pushed before any arm ran. Probe: `probes/E17_DOES_THE_HEAD_RANK.md`.
Runner: `e17_head_rank.py`; follow-up: `e17_head_vs_twin.py`.

**No timing was taken in E17. No rate in §§12–29 moves. 6.79 tok/s exact stands, §19.3 unchanged.**
Decode rates were recorded by the engine and are discarded — the machine's idleness was not
controlled, and a contended timing is not a timing.

### 30.1 Why the head was the suspect

Every E16 arm was exported `--head-ternary` (`e16_r3_at_7b.py:167`). `Qwen2.5-Coder-7B` has
`tie_word_embeddings=False`, so that flag ternarizes a standalone **152064 × 3584 = 545 M-parameter**
output projection — the one tensor whose entire job is ranking, and a candidate mechanism for E16's
"0.155 nats/token better than uniform, 0/160 greedy". E1 §4.3 had already measured that tensor as
nearly free **in BPB** (`−0.008546` at 1.5 B, `+0.022070` at 0.5 B); no probe had ever run a
head-fp32 arm through `--generate` at any scale.

Nothing was exported for E17. Every artifact was on disk from E1, every reference from E6.

### 30.2 The gates, all four fired

| gate | reading | label |
|---|---|---|
| `G-H0a` 0.5 B fp32 | **160/160** | FIRES |
| `G-H0b` 1.5 B `TQH` (= E6 `A3` = E16 `C0`) | **10/160**, div at token 0 | REPLICATED |
| `G-H0c` 1.5 B fp32 — new cell | **160/160** | FIRES |
| `G-H0d` E13 build vs E6 build on `H0b` | token-identical | COMPARABLE |

`G-H0c` is a third independent known-positive at a third scale, and it closes E1 §4.4's open
question in the generate path: the engine loads a **6,174,857,268-byte** file and reproduces
PyTorch exactly on it. `G-H0d` was registered rather than assumed because E6 used
`engine/donor_engine.exe` (315,904 B) and E16 used `D:\_ktmp\e13\donor_engine.exe` (317,952 B);
without it, §30.4's table could not be written at all.

### 30.3 The head axis, and the distinction it forces

| donor | head ternary | head fp32 | Δ | tokens identical to its own twin |
|---|---|---|---|---|
| **1.5 B** | `10/160` | **`12/160`** | **+2** | **81/160** — 49% of the output changes |
| **0.5 B** | `3/160` | **`3/160`** | **0** | **28/160** — **82.5%** of the output changes |

Both head-fp32 arms still diverge from the donor at **token 0**. Bars (brief §5, derived from the
measured known-negative ceiling `10/160` and the known-positive band `160/160`): mechanism at
`≥ 85/160`, not-mechanism at `≤ 20/160`. Both are far below the lower bar, so **stage 2 — the 7 B
head-fp32 export — does not run**, exactly as the brief registered in advance.

**The head is not inert; it is irrelevant.** It rewrites half the output at 1.5 B and five sixths of
it at 0.5 B while changing the number of *correct* tokens by two and by zero. The two arms are not
similar models — they are two differently-wrong ones.

**REGOLA NUOVA: "questo componente cambia l'output" e "questo componente cambia la risposta" sono
due misure diverse, e un nullo sulla seconda non autorizza a chiamare il componente inerte.**
E17 nearly made that error itself: `G-H2`'s `3/160` against `A2`'s `3/160` reads as "the head does
nothing", and the twin comparison shows 82.5% of the tokens moved underneath it.

### 30.4 The population — every arm this programme has ever generated with

Licensed by `G-H0d`. Chance `4.069819` at `V = 151936`, `4.070106` at `V = 152064`.

| arm | scale | rule / fold / head | BPB | vs chance | greedy |
|---|---|---|---|---|---|
| fp32 | 0.5 B | — | `0.871810` | `−3.198` | **160/160** |
| fp32 | 1.5 B | — | `0.767595` (E12 §1) | `−3.302` | **160/160** |
| fp32 | 7 B | — | `0.674027` | `−3.396` | **160/160** |
| `TQ` | 0.5 B | R3 / none / fp32 | `4.509164` | `+0.439345` | `3/160` |
| `TQH` | 0.5 B | R3 / none / ternary | `4.531234` | `+0.461415` | `3/160` |
| **`TQ`** | 1.5 B | R3 / none / fp32 | `3.484253` | `−0.585566` | **`12/160`** |
| `TQH` | 1.5 B | R3 / none / ternary | `3.475707` | `−0.594112` | `10/160` |
| `B1` | 7 B | R0 / layers / ternary | `5.299200` | `+1.229094` | `0/160` |
| `B2` | 7 B | R3 / layers / ternary | `4.017233` | `−0.052874` | `0/160` |
| `B3` | 7 B | R3 / none / ternary | `4.168325` | `+0.098219` | `0/160` |

**Three known-positives at exactly 160/160 across three scales and two model families. Seven ternary
arms across three scales, two rules, two folds and both head settings: best `12/160`, every one
diverging at token 0.**

### 30.5 What that does to the instrument this ledger is written in

Across those seven ternary arms **BPB spans `1.823493`** — `+1.229094` above the chance line to
`−0.594112` below it — while greedy agreement spans **0 to 12 out of 160**.

**BPB is not broken.** Over the full range including fp32 it tracks perfectly: `0.67`–`0.87` goes
with 160/160, `3.48`–`5.30` goes with 0–12/160. The problem is narrower and worse: **inside the
ternary regime — the entire operating range in which this programme ships and in which every rate in
§§12–29 was measured — BPB has resolution and ranking has none, because every ternary arm is already
on the floor.** E16's `RULE-FIXES-IT` was read off `0.052874` of BPB movement in a regime where
`1.8` BPB of movement buys nothing.

### 30.6 Predictions, scored

`G-H0a`/`G-H0b` held; `G-H0d` held; `G-H1` **inside the called 2–15% band** at `7.50%`, and its
*reasoning* (the `TQ` arm retains ≈18% of the donor's information over uniform, so it should not
reproduce the argmax) held as a direction and is **not** promoted for having landed; `G-H2` held
under the bar at `1.88%`; `|Δagreement| < 10` points held at `1.25` and `0.00`; stage 2 did not run.

**One called direction missed: `H2 > A2` at 0.5 B.** Predicted because the head costs `+0.022` BPB
there — measured **exactly equal, `3/160` vs `3/160`**. §30.3's twin comparison shows the equality is
a coincidence of counting, not similarity: 82.5% of the tokens differ.

### 30.7 Where it leaves the goal

By elimination, each step measured rather than argued: the **rule** is most of the BPB damage and
does not restore ranking (§29, `G-R2 = −1.281967`); the **fold** is load-bearing in BPB and does not
restore ranking (§29, `G-R4 = −0.151093`); the **head** is not the mechanism at either scale (here);
and **scale does not rescue it** — the 1.5 B arm E16's brief called working because it sits `0.594`
below the chance line is E6's *planted control*, emitting `" the\n\n the\n the the\n the"` since
2026-09-05.

**No post-hoc conversion of these donors has ever produced a model that can choose a token, at any
scale, under any setting tried.** §29 concluded the binding constraint is the format rather than the
rule; §30 removes the last cheap alternative to that reading.

### 30.8 Owed

1. **The frequency-coincidence floor** — new, and now the cheapest open item in the programme.
   There is no measured baseline for agreement by luck: these arms emit ` the` and `\n` repeatedly
   and PyTorch sometimes does too, so **`12/160` cannot be claimed to be above zero information.**
   Every "best is 12/160" statement is an upper bound on a quantity whose lower bound is unknown.
2. **The ranking band for intermediate values** — E14 §5 item 3, still owed, still needed for E14's
   `45.6%`, and **not** supplied by E17.
3. **`R1`/`R2` at 1.5 B** — only `R0` and `R3` have ever been generated with.
4. Everything §29.6 still owes: the 3 B cell of the R3 sweep, a clean scale axis, a fold sweep at 3 B.

---

## §31 — E18: the ladder is a cliff, and 50 tok/s is a 10% budget. `CLIFF-NOT-SLOPE`.

Brief `briefs/BRIEF_E18_THE_RANKING_LADDER.md`, part B **pre-registered and pushed before any arm
ran** (`2ac74f5`). Probe `probes/E18_THE_RANKING_LADDER.md`. Runners `e18_agreement_floor.py`,
`ternary/e18_ranking_ladder.py`, `e18_ladder_bandwidth.py`.

**No timing was taken. `6.79 tok/s` on the real 7.072 B donor stands exactly; §19.3 unchanged.**
§31.4 is arithmetic over §§23/26 and is labelled as such everywhere it appears.

### 31.1 The floor §30.8 item 1 asked for

Computed from the reference continuations in `results/e6/ref.json` alone — no engine, no weights, so
it cannot be contaminated by the arms it judges. The floor is the score of the best possible
**constant** predictor: the strongest zero-information model.

| reference | positions | distinct tokens | best CONSTANT predictor | token |
|---|---|---|---|---|
| Qwen2.5-0.5B | 160 | 91 | **`11/160` = 6.88%** | `'\n'` |
| Qwen2.5-1.5B | 160 | 82 | **`12/160` = 7.50%** | `'\n'` |

**§30.4's best ternary arm scored exactly `12/160`.** Every "best is 12/160" in §30 and in E17 must
be read as **"best is exactly the floor"**; the `TQH` arm at `10/160` is *below* it and emits `'\n'`
**77 times of 160**. Nothing in §30 is withdrawn — §30.8 had already refused the claim this settles
— but §30.5's finding is stronger than it was written: BPB does not merely have poor resolution in
the ternary regime, it spans `1.82` across models that are, to the ranking instrument, uniformly
indistinguishable from a constant `'\n'` emitter.

### 31.2 The controls, which are why the empty rungs count

`G-L0`: `base` reproduces `results/e6/ref.json` at **`160/160`** — the PyTorch harness fires on the
known-positive first. `G-L1`: the identity arm, which walks the substitution path and substitutes
nothing, is **token-identical** to `base` (T2b gated this in BPB at `+0.000e+00`; it had never been
gated in generation).

**`G-L2` — the harness reproduces the engine EXACTLY**, not "within the margin":

| arm | E18 (PyTorch) | engine | Δ |
|---|---|---|---|
| `FA` (= E1 `TQ`) | **12/160** | 12/160 (E17 `G-H1`) | **0** |
| `FAH` (= `TQH`) | **10/160** | 10/160 (E6 `A3`) | **0** |

Two independent known values reproduced across two instruments before one empty rung was read.
That is the entire licence for §31.3 (Phase 60's law, taken in the direction it actually runs).

### 31.3 The cliff

Bands fixed in the brief before the run: floor `12` (§31.1), margin `2` (E17's own — a change that
rewrote 49% of the output moved 2 tokens), so `AT-FLOOR ≤ 14`, `RANKS ≥ 80`.

| arm | organs ternarized | BPB | vs chance `4.069819` | greedy | label |
|---|---|---|---|---|---|
| `base` | — | `0.767595` | `−3.302224` | **160/160** | `G-L0` FIRES |
| `I` | identity path | `0.767595` | `−3.302224` | **160/160** | `G-L1` FIRES |
| **`H`** | `lm_head` only | **`1.106584`** | `−2.963235` | **9/160** | **`AT-FLOOR`** |
| **`A`** | `q,k,v,o` only | `1.903569` | `−2.166250` | **4/160** | **`AT-FLOOR`** |
| **`F`** | `gate,up,down` only | `2.476967` | `−1.592852` | **5/160** | **`AT-FLOOR`** |
| `FA` | `F+A` | `3.484251` | `−0.585568` | `12/160` | `AT-FLOOR` |
| `FAH` | `FA+H` | `3.475706` | `−0.594113` | `10/160` | `AT-FLOOR` |

Five of the seven rungs had never been generated with. **All five are at the floor. There is no rung
between "exact" and "broken".**

**The sharpest reading is `H`.** It ternarizes **one tensor**, leaves the whole body in fp32, costs
**`+0.338989` BPB** against the untouched donor and lands `2.963` *below* the chance line — by the
metric this ledger is written in, a better model than most arms the programme has ever shipped.
**It agrees with its own donor on 9 of 160 greedy tokens, below the constant-`'\n'` floor of 12.**

### 31.4 The same ladder on the speed axis — DERIVATION, no timing

Every input measured and cited; **bytes-per-weight is recomputed from `qwen25-15b_tqh.bin` rather
than assumed and comes out to exactly `0.500000` B (`4.0000` bits)**. Moved-byte convention
throughout (§23's `25.5 GB/s` packed, §26's `34.75 GB/s` fp32, from the same sweeps): **an fp32
weight costs `5.87×` a ternary one on this machine.**

| arm | ternary G-w | fp32 G-w | tok/s (weight path) |
|---|---|---|---|
| `base` | 0.000 | 7.070 | **1.23** |
| `H` | 0.545 | 6.525 | 1.31 |
| `A` | 0.822 | 6.248 | 1.36 |
| `F` | 5.703 | 1.367 | 3.71 |
| `FA` | 6.525 | 0.545 | 5.24 |
| **`FAH`** | 7.070 | 0.000 | **7.21** |

Self-check: the derived `FAH` weight path `7.21` sits just **above** the engine's measured `6.79`
(ratio `1.06`) — the correct side, since the weight path excludes attention math, norms, softmax and
glue. **The fastest rung is the one that converts everything, and it is `6.9×` short of 50 tok/s.
Quality improves down that table and speed improves up it.**

**The budget the goal implies**, at the packed kernel's measured rate:

| target | active ternary weights / token | share of a 10 B model |
|---|---|---|
| **50 tok/s** | **0.982 – 1.060 G** | **9.8 – 10.6%** |
| 100 tok/s | 0.491 – 0.530 G | 4.9 – 5.3% |

Reported as a span, not a point: **§23.3 pairs "~25.5 GB/s" with "~53 G-w/s" in one row, and
`25.5 / 0.5 = 51.0`** — its own two companion numbers disagree by **4%**. Recorded here rather than
resolved by quietly picking one.

### 31.5 BPB and ranking run backwards inside the converted regime

Across the five converted arms, spanning **`2.377667` BPB**: **`r(BPB, agreement) = +0.4989`** — a
*positive* correlation. The best-scoring converted arm, `H` at `1.106584`, ranks **worse** (`9/160`)
than the worst-scoring one, `FA` at `3.484251` (`12/160`).

Honest reading: n = 5 and all five are at the floor, so the ordering *inside* the floor is noise and
`+0.4989` is not a mechanism — it is what "no signal" looks like when a correlation is computed on
it anyway. It is recorded because the **absence of the expected negative correlation across 2.4 BPB**
is the finding. §30.5 said BPB has resolution in the ternary regime and ranking has none; §31 adds
the regime §30 never sampled — **`H` is not "in the ternary regime" by any BPB reading and still
cannot rank. The decoupling is a property of CONVERTED models, not of bad ones.**

### 31.6 Predictions, scored — three called directions, three misses

`G-L0`/`G-L1` held. `G-L2` held **exactly, zero error on both**. **`H` → `RANKS`: WRONG** (`9/160`).
**`A` → `RANKS`: WRONG** (`4/160`; the "below `H`" half held). **`F` →
`ABOVE-FLOOR-DOES-NOT-RANK`: WRONG** (`5/160`). Prediction 3 was flagged in the brief as "the least
confident call" and was still wrong by 71 tokens.

**What preserved the reading was not the forecasting but brief §6 prediction 6, written before the
data existed:** *"If instead `H`, `A` and `F` all come back at the floor, the collapse is at the very
first rung and the readable conclusion is far stronger — that ternarizing any single organ of this
donor destroys ranking."*

**Law: registering the ALTERNATIVE outcome is worth more than getting the direction right, because
the alternative is what protects the reading when the direction fails.** Missed directions now in
E16 (`G-R3`), E17 (`H2 > A2`) and E18 (three of three); all three verdicts survived because the
brief had said in advance what each outcome would mean.

### 31.7 Where it leaves the goal

Both axes now fail independently, each measured rather than argued.

**Quality**: there is no partial-conversion operating point. Ternarizing one organ of a frozen donor
— even the one that costs `0.34` BPB — takes it from `160/160` to below a constant-`'\n'` emitter.
With §§28–30 the elimination is complete across rule, fold, head, organ coverage and scale.

**Speed**: 50 tok/s permits **0.98–1.06 G active ternary weights per token, 9.8–10.6% of a 10 B
model**. The 7 B donor activates `7.07 G`. The fastest possible point on the conversion ladder is
**6.9× short**, and every rung that would improve quality is slower still.

The target is therefore not reachable by converting a dense donor, at any conversion quality, for
two independent reasons. **It requires a model trained INTO the format that activates ~10% of itself
per token** — `SCALEUP_ARCHITECTURE`'s premise and Phase 64's actual programme. §29 called the
constraint the format; §30 removed the last cheap alternative; **§31 gives it a number on both axes.**

### 31.8 Scope, and what is owed

**E18 tests conversion WITHOUT healing, fine-tuning or QAT.** That a *converted* donor cannot rank
says nothing about a donor *healed* into the format, and the literature's ternary results are all
trained-in. **This is the largest scope limit and it names the only branch of the conversion route
left standing.** Part B is PyTorch, not the engine (`G-L2` bridges them exactly at the two rungs
where both exist); `H`/`A`/`F` cannot be built as engine artifacts without changing `QWENDON1`,
which carries one global `quant` field. One donor, one scale (1.5 B), one rule, 160 positions.

1. **The same ladder WITH healing** — convert one organ, fine-tune briefly, re-measure ranking.
   If it recovers `160/160` the route reopens; if not, it closes.
2. **The intermediate ranking band** — E14 §5 item 3, owed since E14, **not** supplied by §31.1,
   which bounds a *degenerate* model only: **E14's `45.6%` remains unbanded.**
3. **The cliff at 7 B** — measured here at 1.5 B; E16's `0/160` is consistent but a different family.
4. **`R1`/`R2`**, deprioritized in brief §7 with the reason recorded: both are worse than `R3` at
   equal coverage and `R3` at *any* coverage is now at the floor, so neither can open a route.
5. Everything §29.6/§30.8 still owes: the 3 B cell of the R3 sweep, a clean scale axis, a fold sweep.

---

## §32 — E19: the carve does not rank, and FFN-only carving cannot reach the target. `CARVE-DOES-NOT-RANK`.

Brief `briefs/BRIEF_E19_DOES_THE_CARVE_RANK.md`, **pushed before part B ran** (`3173edc`); runner
`bdd87a4`. Probe `probes/E19_DOES_THE_CARVE_RANK.md`. Run 2280 s, `VOID: none`.
**No timing taken. `6.79 tok/s` stays exact; §19.3 unchanged.**

§31 closed conversion on both axes and left one structural escape: 50 tok/s is a ~10%-activation
budget, and the only lever that reaches an activation budget is conditional activation — carving.

### 32.1 Part A — FFN-only carving cannot reach the target, at any depth

Derivation over measured quantities, importing `qwen_shapes` and the bytes-per-weight verification
from §31's runner rather than restating them. Every weight charged as **ternary** (`0.500000` B) —
the friendliest possible assumption, since it grants full ternarization for free.

| model | active/token | FFN | **attn + head** | tok/s with the **FFN at ZERO** |
|---|---|---|---|---|
| Qwen2.5-Coder-7B | `7.070 G` | `5.703 G` (80.7%) | **`1.367 G`** | **35.9 – 38.8** |
| Qwen2.5-1.5B | `1.544 G` | `1.156 G` (74.9%) | `0.388 G` | 126.7 – 136.8 |

**`attn + head` on the 7 B (`1.367 G`) exceed the ENTIRE 50 tok/s budget (`0.982–1.060 G`), so
deleting the whole FFN still leaves 35.9–38.8 tok/s.** Every carve this programme has built (D0,
D0c, Probe-4) is FFN-only. Projected to the goal's size on each donor's measured attn+head share —
a projection, labelled as one, no config invented — a 10 B carries a **`1.93–2.51 G` uncarvable
floor = `1.82–2.37×` the whole budget**, i.e. **19.6–27.4 tok/s with its FFN carved to zero**.

Published in the brief **before** part B ran, and it fixed part B's arm list: at 1.5 B, 50 tok/s
needs FFN activation `0.5144–0.5817`, so the verdict cell is **51.95%**, and the 25% cells D0/D0c
scored are *more* aggressive than that donor's own budget requires.

### 32.2 The controls, at zero error

`G-C0`: `base` reproduces E6's reference at **`160/160`**. `G-C1`: `FULL` (`k = E`, the hook runs
and removes nothing) is **token-identical** to `base`, BPB difference **`0.000e+00`**.

**`G-C2`** — the hook had to be restated (`d0c_granularity.py` loads a model at import and cannot be
imported), so it is gated **behaviourally**: `base`, `S1`, `A0`, `N0` reproduce D0c's published BPB
to **`0.000e+00` — sixteen decimals, all four**. The partitions are literally D0c's, loaded from the
caches its run wrote, so `CLUSTER_SEED` and the B3 repair are inherited by construction.
`G-C4`: achieved activation equals nominal **exactly** on all six carved arms.

### 32.3 The ladder — every carved arm at the floor

| arm | activation | BPB | vs chance `4.069819` | greedy | label |
|---|---|---|---|---|---|
| `base` | 100% | `0.767595` | `−3.302224` | **160/160** | `G-C0` FIRES |
| `FULL` | 100% | `0.767595` | `−3.302224` | **160/160** | `G-C1` FIRES |
| **`V52`** | **51.95%** | **`0.909441`** | `−3.160378` | **12/160** | **`AT-FLOOR`** |
| `S1` | 25% | `1.383868` | `−2.685951` | 7/160 | `AT-FLOOR` |
| `A0` | 25% | `1.858218` | `−2.211601` | 5/160 | `AT-FLOOR` |
| `N0` | 25% null | `2.578731` | `−1.491088` | 6/160 | `AT-FLOOR` |
| `D10` | 10.16% | `2.806167` | `−1.263652` | 3/160 | `AT-FLOOR` |

**`V52` is the cell that matters and it is worse news than §31's `H`.** Keeping 52% of the FFN — the
depth this donor's own budget requires, selected by an **oracle** reading the true activation mass —
costs **`+0.141846` BPB**, lands `3.160` below the chance line, and **agrees with its own donor on
12 of 160 tokens: exactly the constant-`'\n'` floor**, diverging at token 0.

**Unlike §31, the ordering is not scrambled**: `r(BPB, agreement) = **−0.8562**` across the five
carved arms, the expected sign, over a `1.896726` BPB span. **That sharpens the result rather than
softening it — the degradation is well-behaved and its first usable point is already at zero
information.** No depth is both fast enough to matter and able to choose a token.

### 32.4 Co-activation buys BPB and buys nothing in ranking

D0's headline and D0c's decision rest on the co-activation partition beating a matched random one.
At 25% that gap is large in BPB: `A0 − N0 = **−0.720513**`. In ranking it is **absent**: `5/160` vs
`6/160`. The reading is not that the null ranks better — `5` vs `6` is floor noise by §31's own law
— it is that **`0.72` BPB of partition quality produces no measurable ranking difference**, because
both arms are already degenerate. D0c §3.2's `G32` gap and the granularity decision built on it are
statements about score in a regime where score does not correspond to competence.

### 32.5 Predictions — three called directions, three misses, again

`G-C0`/`G-C1` held; `G-C2` held **exactly**; `D10 → AT-FLOOR` held. **`V52 → RANKS` WRONG**
(`12/160`). **`S1`/`A0` → `PARTIAL` WRONG** (both `AT-FLOOR`; the `S1 > A0` ordering held).
**`N0 < A0` WRONG** (floor noise).

Prediction 3 was argued from a real mechanism — a carve leaves every surviving weight **bit-exact**
and computes a *subset* of the true function, where ternarization perturbs every weight. The
mechanism is true and the call was still wrong by 68 tokens. **A different kind of damage is not a
smaller kind of damage.**

Fourth consecutive experiment (E16, E17, E18, E19) whose directions missed and whose verdict
survived because the brief registered the alternative first — §31.6's law, now paid for a fourth
time.

### 32.6 Where it leaves the goal

**Quality**: the donor's argmax survives neither precision change (§§30–31) nor structural sparsity
(§32), at any depth, under an oracle router, on an otherwise intact donor. The failure is not a
property of ternarization but of **post-hoc modification of a pretrained dense model as such**.

**Speed**: even granting a carve that worked, FFN-only carving cannot reach 50 tok/s at 7 B or
above. A carve that reached the target would have to cut **attention and the output head too** —
never attempted, and prima facie hostile to §30's finding that the head is where ranking lives.

The remaining branch is unchanged and now sole: **train into the format**, ~10% active per token.
Within the donor route only **healing** is untested, and §32 raises its bar — it would have to
repair a model degenerate under *either* kind of modification.

### 32.7 Owed

1. **Healing**, §31.8 item 1, unchanged and sharpened: heal `V52` (cheapest at `+0.141846` BPB) and
   re-measure ranking. If `160/160` does not return there it will not return anywhere on this route.
2. **The intermediate ranking band** — E14 §5 item 3, **still** not supplied: every arm here is
   degenerate too.
3. **Carving attention and the head** — what part A says the target actually requires.
4. A re-read of **D0 §III and D0c §5**, whose decisions rest on a BPB gap §32.4 shows carries no
   ranking signal. Their numbers stand; the decisions built on them need restating.

---

## §33 — E20: the rule axis is exhausted, and the ranking metric was two things.
`RULE-EXHAUSTED-DRIFT-DOMINATES`.

Brief `briefs/BRIEF_E20_RULE_OR_FORMAT.md`, **pushed before any arm ran** (`e5d438c`); run 1 VOID
(`6c7b7a4`); probe `probes/E20_RULE_OR_FORMAT.md`. Part A 3449 s, part B 459 s, `VOID: none`.
**No timing taken. `6.79 tok/s` stays exact; §19.3 unchanged.**

§29 promoted the constraint from the rule to the format while recording that only 2 of 4 rules had
been tested. §§30–32 then eliminated the head, the organ ladder and structural sparsity. E20 pays
off §29's qualifier on the one tensor whose job is ranking.

### 33.1 The gate voided run 1, and the brief's premise with it

`BRIEF_E20` §0 asserts that every ternarization here is `mean|w|` round-to-nearest, citing
`t1_ternarize.py:97-98`. **That is `R0`.** The shipped rule is `qwen_export.quantize` under
`--rule R3` = `t2_rules.r3_actsearch`, an **activation-RMS-weighted** per-row threshold search over
the calibration slice, and `qwen_export.py:148` hooks `lm_head` too — **the shipped head
quantization already looks at tokens.** Run 1's `R3H` was therefore `R0` on the head and `G-Q2`
read `1.319900` / `17/160` against the registered `1.106584` / `9/160`. **VOID as registered.**

GPTQ was also not untried: `t2_rules.r4_gptq` has existed since T2, which ran it on the FFN —
`R4` `4.299819` (**above** the chance line) and `R5` `2.027495`, **the best rule ever measured
here**, `0.449472` better than the shipped `R3` and marked post-hoc under T2's own decision rule.
Its consequence had gone unstated: **`R5` is not implemented in the exporter** (`quantize`
dispatches `R0`/`R1`/`R2`/`R3` only) and had never been read in ranking. The decomposition row
`GPTQ error compensation on a well-placed grid − 0.449` has been sitting in `INDEX.md` §3 the
whole time.

Run 2 imports every rule from `t2_rules` — *one definition each* — and its `R3H` arm is
`t2b_organs.apply_arm(model, "H", act_rms, None)` itself.

### 33.2 Part A — eight ternary heads, one format, none of them ranks

Format held exactly at what `QWENDON1` stores: ternary codes, **one fp32 scale per output row**,
same bytes, same kernel, `0.500000` B/weight. `lm_head` only; every other weight bit-exact.
Bands unchanged from §§30–32: floor `12`, margin `2`, `AT-FLOOR ≤ 14`, `RANKS ≥ 80`.

| arm | knows the data? | BPB | Δ vs base | vs chance `4.069819` | greedy | band |
|---|---|---|---|---|---|---|
| `base` / `ID` | — | `0.767595` | `0.000000` | `−3.302224` | **160/160** | — |
| `R0H` `mean|w|` RTN | no | `1.319900` | `+0.552305` | `−2.749919` | 17/160 | PARTIAL |
| `R1H` TWN | no | `1.280712` | `+0.513117` | `−2.789107` | 13/160 | AT-FLOOR |
| `R2H` unweighted search | no | `1.288465` | `+0.520871` | `−2.781354` | **41/160** | PARTIAL |
| **`R3H` SHIPPED** | yes | `1.106584` | `+0.338989` | `−2.963235` | 9/160 | AT-FLOOR |
| `R4H` GPTQ | yes | `1.031616` | `+0.264021` | `−3.038203` | 15/160 | PARTIAL |
| `R5H` GPTQ + act scale | yes | `0.940203` | `+0.172608` | `−3.129616` | 9/160 | AT-FLOOR |
| `OPTH` scale grid (E20's) | no | `1.293924` | `+0.526329` | `−2.775895` | **42/160** | PARTIAL |
| **`GPTQH`** (E20's) | yes | **`0.938009`** | **`+0.170414`** | `−3.131810` | 11/160 | **AT-FLOOR** |

**The registered verdict arm reads `AT-FLOOR`**, so the brief's registered alternative — *"if
`GPTQH` comes back `RANKS` or `PARTIAL` the conversion route reopens"* — **does not fire.** The
best ternary head anyone here can build costs `+0.170414` BPB, sits `3.131810` below chance, and
agrees with its own donor on 11 of 160 tokens, **below the constant-`'\n'` floor.**

Gates: `G-Q0` `160/160`; `G-Q1` token-identical, BPB diff `0.000e+00`; **`G-Q2` hits BOTH published
anchors — `1.1065835970951252` against T2b's `1.1065836079824596` (`abs diff 1.09e-08`) and `9/160`
against E18's `9`.** `G-Q4` holds: the best-BPB arm is data-aware, and `R5H − R3H = −0.166380`
here against `−0.449472` on T2's FFN.

**`r(BPB, agreement) = +0.6238` over a `0.381891` BPB span — the wrong sign, third occurrence**
(§31 `+0.4989`, §32 `−0.8562`). The two best-ranking heads are both **data-free** and are the
worst-scoring of the search family.

### 33.3 The `PARTIAL` band is one prompt, and that prompt has no near-tie

E14 §5 item 3 has owed an intermediate point since 2026-09-07. E20 produces four — and the split
shows what they are: `R2H` scores `[1, 3, 32, 2, 3]` and `OPTH` `[2, 2, 32, 3, 3]`. **Both
reproduce prompt 2 WHOLE and sit at floor noise on the other four.** Prompt 2 is not degenerate
(18 distinct ids in 32 tokens) and it is **the only prompt with no near-tie anywhere**: its minimum
donor top-2 gap is `1.1630` while prompts 0/1/3/4 dip to `0.0739`/`0.0453`/`0.2740`/`0.0081`.

**160 positions are five trials of thirty-two.** Every ranking number published since §E7 carries
an effective *n* of **5**, and `41/160` differs from `9/160` largely by which prompt survived.

### 33.4 Part B — per-step argmax fidelity is 67-74%, and the rest is drift

**Unregistered; reports, does not decide (E14 §6).** Teacher-forcing the donor's own continuation
makes each of the 160 positions an independent test with the context held identical.

| arm | free-running | **teacher-forced** | drift cost | mean rank of donor token |
|---|---|---|---|---|
| `base` / `ID` | 160/160 | **160/160** | 0 | `1.00` |
| `R0H` | 17/160 | **107/160** | −90 | `16.92` |
| `R1H` | 13/160 | **114/160** | −101 | `12.85` |
| `R2H` | 41/160 | **116/160** | −75 | `13.96` |
| `R3H` | 9/160 | **110/160** | −101 | `2.98` |
| `R4H` | 15/160 | **115/160** | −100 | `2.40` |
| `R5H` | 9/160 | **112/160** | −103 | `2.71` |
| `OPTH` | 42/160 | **119/160** | −77 | `13.26` |
| `GPTQH` | 11/160 | **117/160** | −106 | `2.88` |

**`G-B1`, an exact identity, holds `50/50`**: an arm matching the donor at every position `< k`
under teacher forcing must first diverge at exactly `k` free-running. Both harnesses are therefore
mutually consistent and the drift account is arithmetic.

Three readings. **(1)** Per-step fidelity spans 12 tokens across the eight arms; free-running
spreads them over 33 — most of what the metric measured was *where the first miss fell*.
**(2)** The families separate on **mean rank**: data-aware arms `2.40`-`2.98`, weight-space arms
`12.85`-`16.92`. The Hessian- and activation-weighted objectives do exactly what they promise —
preserve the logit geometry — which buys BPB and neighbourhood, **not the top-1/top-2 boundary
where argmax lives.** That is §33.2's inversion, measured. **(3)** Survival tracks the donor's own
margin: in the top tercile of donor top-2 gap (`4.4439`-`15.9636`) every arm scores **94-98%**; in
the bottom tercile (`0.0081`-`1.7294`), **38-47%**. A drift model `5·(1−p³²)/(1−p)` predicts
`15.1`-`19.5` free-running; six of eight arms land in `9`-`17`.

### 33.5 What moves

**No speed number moves.** Same format, same bytes, same kernel — a rule change moves no byte.
`6.79 tok/s` exact, §19.3 unchanged, §32's FFN-carving arithmetic unchanged.

**The donor route still fails.** 70% per step compounds to nothing over 32, and the best head costs
`+0.170414` BPB for `11/160`. Nothing reopens the conversion route.

**§29's rule qualifier is discharged on this tensor**: six rules, two of them the literature's
strongest post-training quantizers, one format, and the best is still below the floor. **The
constraint is the format.**

**But the standing claim needs re-wording.** "These conversions cannot choose a token" is too
strong. They choose the donor's token about two times in three at fixed context and rank it 2nd-3rd
when they miss; **what they cannot do is survive their own first mistake.**

### 33.6 Owed

1. **Healing / QAT**, unchanged as first since §31.8, and now sharply targeted: repair a head that
   is already right 70% of the time per step. **Needs GPU; the user launches it.**
2. **Re-read every published ranking number as a compound** of per-step fidelity and drift over an
   effective *n* of 5. No verdict is withdrawn — free-running is what a runnable model does — but
   the mechanism attributed to each needs restating.
3. **Adopt teacher-forced top-1 as the standing second metric.** 459 s for ten arms, no drift,
   fires exactly on the known-positive.
4. **`R5` in the exporter** — a branch in `qwen_export.quantize`. It improves every quoted BPB and,
   on this evidence, changes no answer. Recorded so nobody later fixes the exporter and believes
   they fixed the model.
5. **The rule axis on the FFN in ranking** — T2's six rules were read in BPB only.

---

## 34. E21 — rank: attention yes, the head no

**Probe**: `probes/E21_CAN_RANK_BUY_IT.md`. **Brief**: `briefs/BRIEF_E21_CAN_RANK_BUY_IT.md` @
`24b8832`, pushed before any arm ran. **Runner**: `ternary/e21_rank.py`. **Results**:
`engine/results/e21_rank.json`, 3445 s, `VOID: none`.

**No timing taken. Nothing exported. `6.79 tok/s` exact, §19.3 unchanged.**

### 34.1 The axis, and the one number that matters

Rank had never been measured end to end here — D2 computed spectra for three layers and stopped.
E21 replaces donor matrices with the exact rank-`r` minimiser of the **activation-weighted** error
`‖(W − Wᵣ)X‖_F` (not `‖W − Wᵣ‖_F`), `H = XᵀX` over the frozen 32×512 seed-42424 calibration slice,
damping `λ = 0.01·mean(diag H)` as E20.

| arm | BPB | Δ base | free-running | **teacher-forced** | mean rank | band |
|---|---|---|---|---|---|---|
| `base` / `FULL` | `0.767595` | `0` | 160/160 | 160/160 | `1.00` | controls FIRE |
| `H-SVD-256` | `3.826871` | `+3.059276` | 1 | 12 | `39258` | WORSE |
| `H-SVD-512` | `3.480711` | `+2.713116` | 0 | 27 | `32055` | WORSE |
| `H-ACT-256` | `1.330385` | `+0.562790` | 7 | 68 | `22.38` | WORSE |
| `H-ACT-512` | `1.045691` | `+0.278096` | 9 | 101 | `4.83` | WORSE |
| `QO-SVD-256` | `4.686939` | `+3.919344` | 1 | 2 | `28329` | WORSE |
| `QO-ACT-256` | `1.545880` | `+0.778285` | 7 | 93 | `43.31` | WORSE |
| **`QO-ACT-512`** | **`0.820284`** | **`+0.052689`** | **26** | **144** | **`1.16`** | **CHEAPER** |
| `BOTH-ACT-256` | `1.766172` | `+0.998577` | 4 | 48 | `103.96` | WORSE |

**`QO-ACT-512` — activation-weighted rank-512 on all 28 layers' `q_proj` and `o_proj` — costs
`+0.052689` BPB and keeps the donor's token first at 144 of 160 fixed-context positions.** It is
the **least damaging structural modification this programme has measured** (against E20's best
ternary head `+0.170414` and E19's `V52` `+0.141846`), the **only arm in E18/E19/E20/E21 to read
`RANK-IS-CHEAPER`**, and it puts the donor's token in the **top 5 at all 160 positions**. Its
per-prompt teacher-forced split `[30, 26, 30, 29, 29]` is **uniform** — §33.3's one-prompt trap
was checked for and is absent — and it is the first modified donor here that writes correct text
(Paris, 212/32 °F, a working Fibonacci branch).

### 34.2 `G-R2` — the data weighting is doing all the work

| pair | plain SVD | activation-weighted | Δ |
|---|---|---|---|
| head r=256 | `3.826871` | `1.330385` | **`−2.496486`** |
| head r=512 | `3.480711` | `1.045691` | **`−2.435020`** |
| `q/o` r=256 | `4.686939` | `1.545880` | **`−3.141059`** |

Plain SVD on `q/o` at r=256 is **above the chance line** (`4.686939` vs `4.069819`) — worse than
guessing — while the same rank weighted reads `1.545880`. D2's spectra said these matrices are
**not** low-rank in weight space (`o_proj` needs 596–937 of 1536 for 90% Frobenius energy) and
they are right: the weighted construction keeps `0.8937` of the energy **in the directions the
data occupies** where plain SVD keeps `0.3638` of the Frobenius energy. **A matrix that is not
low-rank in weight space can be low-rank where it is used.** D4's Hessian ablation, confirmed on a
second axis.

### 34.3 What moves on the speed side — and what does not

**No measurement moves.** What moves is §32's arithmetic floor. E19 fixed `attn+head` on the
Coder-7B at **`1.367 G`** against a 50 tok/s budget of `0.982–1.060 G`. Charging every weight as
ternary (`0.500000` B) and using **the rank fraction actually validated** (`r/D ≈ 1/3`, not the
brief's `r = 256` which at `D = 3584` is 1/14 and which `BOTH-ACT-256` shows this donor cannot
take):

| Coder-7B | dense | `q/o` at `r/D = 1/3` |
|---|---|---|
| `q + o` | `719.3 M` | `479.5 M` |
| `k + v` | `102.8 M` | `102.8 M` |
| `lm_head` | `545.0 M` | `545.0 M` |
| **`attn + head`** | **`1.367 G`** | **`1.127 G`** |

**`1.127 G` is still 1.06–1.15× the whole budget with the FFN at zero.** E19's floor drops by
`240 M` and the conclusion survives: **the head is the binding constraint.** It is `545 M` =
51–56% of a 7B's entire 50 tok/s budget, it resists rank (`H-ACT-512`: `+0.278096` BPB, 101/160
against `q/o`'s `+0.052689` / 144 at the same rank), and E20 showed its cheapest known treatment
costs `+0.170414`. **No lever measured in this programme makes the output head small.**

**And the kernel is not written.** A factored tensor is two GEMVs with an intermediate of size `r`.
The weight *bytes* fall with the parameter count — which is what a memory-bound engine charges
(§19.3, E13: the weight path runs at 97% of demonstrated stream) — but **`engine.c` has no
factored matvec and `QWENDON1` has no kind for one.** No rank result converts to tok/s until it does.

### 34.4 Three asymmetries worth keeping

1. **Attention tolerates rank; the head does not.** Same construction, same rank, same run:
   `+0.0527`/144 vs `+0.2781`/101. The head maps 1536 dims to 151,936 logits whose top-2 gap *is*
   the answer; attention feeds a residual stream that sums 28 layers.
2. **Composition is sub-additive in BPB and below both parts in ranking.** `BOTH-ACT-256` reads
   BPB `1.766172` against an additive prediction of `1.330385 + 1.545880 − 0.767595 = 2.108670`
   (**excess `−0.342498`** — *better* than adding) and **48/160** teacher-forced against parts of
   68 and 93 (**worse than either**). The two metrics disagree in opposite directions about the
   same pair, so "damage compounds" is true of ranking only. Any plan that stacks cheap cuts must
   measure the stack, on both metrics.
3. **BPB ordering can fail across axes and must be checked.** `H-ACT-256` (BPB `1.330385`) and
   E20's `R0H` (`1.319900`) are within `0.01` BPB and read **68** and **107** teacher-forced.
   **Weakened by §35.6**: E22 measured a second pair (`V52` `+0.141846` vs E20's best head
   `+0.170414`, teacher-forced `117` against `117`) where the ordering held to the token. A
   failure mode to test for, not a rule.

### 34.5 Predictions

Two held (`G-R0`/`G-R1` exact; `G-R2` by `−2.44` to `−3.14`), one held in direction (`QO-ACT-256`
93 > `H-ACT-256` 68), **two missed**: prediction 3 put `H-ACT-256` in `107`–`119` and it read `68`;
prediction 5 put `BOTH-ACT-256` at `RANK-IS-COMPARABLE` and it read `RANK-IS-WORSE`.

**The registered alternative was keyed to `BOTH-ACT-256` and did not fire there.** `QO-ACT-512`
reading `RANK-IS-CHEAPER` is a pre-registered arm against a pre-registered band and stands on its
own, **but it is not the §2 configuration**, and the alternative's consequence — retargeting the
healing — is claimed for **attention only**.

### 34.6 Owed

1. **Compose `QO-ACT-512` with E19's `V52`.** `+0.052689` and `+0.141846`; together they land the
   1.5 B at ≈`0.98 G` active, inside the budget. **§34.4 item 2 says do not assume additive.**
   Cheap, CPU, on-goal.
2. **Rank × precision.** Every E21 arm is fp32; the engine ships ternary. A ternary low-rank `q/o`
   is the artefact that would run, and the two damages have never been composed.
3. **The rank fraction at scale** — `r/D = 1/3` validated at `D = 1536` only; §34.3's 7B row is a
   projection, and E16's non-monotonicity applies.
4. **A factored matvec in `engine.c` and a kind in `QWENDON1`** — §34.3.
5. **The head.** §34.3. Vocabulary-side factorisation and tied output clusters are untouched.
6. **Healing / QAT**, unchanged as first since §31.8 and now retargeted by §34.1: the best-posed
   starting point available is **low-rank attention at 90% per-step fidelity**, not a ternary donor
   at 70%. **Needs GPU; the user launches it, and has offered T4 weeks for exactly this.**

---

## 35. E22 — the budget-feasible object and the working object are not the same object

**Probe**: `probes/E22_DOES_CHEAP_COMPOSE.md`. **Brief**:
`briefs/BRIEF_E22_DOES_CHEAP_COMPOSE.md` @ `3f8465a`, pushed before the runner existed.
**Runner**: `ternary/e22_compose.py`. **Results**: `engine/results/e22_compose.json`, nine arms,
3793 s, `VOID: none`.

**No timing taken. Nothing exported. `6.79 tok/s` exact, §19.3 unchanged.**

### 35.1 Three replications, then the verdict

`G-S0`, `G-S1`, `G-S2` all **FIRE exactly**: `base` at `0.767595`/160/160, `QO512` reproducing
**E21** (`0.8202837636996289`, free `26`, tf `144`) and `V52` reproducing **E19**
(`0.909440994415161`, free `12`, activation `0.51953125`), each to `< 1e-9`. Two prior runs from
two runners written weeks apart, reproduced before any composed arm was read — possible only
because the runner imports `e21_rank.lowrank`, `e19_carve_rank.install` and `t2_rules.r3_actsearch`.

| arm | BPB | Δ base | free | **tf** | mean rank | active | band-tf |
|---|---|---|---|---|---|---|---|
| `base` | `0.767595` | — | 160 | 160 | `1.00` | `1.5436 G` | — |
| `QO512` | `0.820284` | `+0.052689` | 26 | 144 | `1.16` | `1.4995 G` | CHEAPER |
| `V52` | `0.909441` | `+0.141846` | 12 | **117** | `2.49` | `0.9882 G` | COMPARABLE |
| **`QO512+V52`** | **`1.005039`** | **`+0.237444`** | **15** | **126** | **`2.03`** | **`0.9441 G`** | **CHEAPER** |
| `QO512-TB` | `2.812226` | `+2.044631` | 1 | 28 | `1476` | — | WORSE |
| `STACK` | `3.947669` | `+3.180074` | 5 | 4 | `24829` | `0.9441 G` | WORSE |

### 35.2 The half that is good news

**`QO512+V52` is the first configuration this programme has derived inside the 50 tok/s budget**
— `0.9441 G` active against `0.982–1.060 G` (E18 §31), 3.9% under the low end — **and it is the
best-ranking modified donor ever measured here**, 126/160 teacher-forced, above every ternary head
E20 read. Per-prompt `[27, 24, 22, 28, 25]`, uniform; median rank `1.0`; top-5 at 148/160.
**The registered alternative fires.**

### 35.3 The half that takes it back

**That artefact is fp32; the `0.9441 G` was priced in ternary at `0.500000` B/weight.** Convert
the same configuration into the format it would have to ship in and it dies:

| attention, ternary | BPB | organs |
|---|---|---|
| **T2b arm `A`** — dense, `R3` | **`1.903569`** | `q`, `k`, `v`, `o` |
| `QO512-TB` — rank-512 `q/o`, both factors `R3` | `2.812226` | `q`, `o` only (`k/v` fp32) |

**`+0.908657` BPB worse on a strictly easier organ set.** Layer 0 `q_proj`, relative weight error:
dense ternary `0.8084`, fp32 rank-512 `0.3515`, **two ternary factors `0.9874`** — the errors
multiply. `STACK`, the whole runnable model, reads BPB `3.947669` (`0.122` below chance), free
`5/160`, teacher-forced `4/160`.

**So the budget-feasible object and the well-ranking object are two different objects, and what
separates them is the format.** That is E18's conclusion arriving from a new direction, with a
target attached: the structure to *train into* is low-rank attention plus a carved FFN.

### 35.4 The construction defect the smoke caught

Brief §3 registered `A = W H^½ Bᵣ`, `B = Bᵣᵀ H^-½`, both ternarized. That puts the whole
singular-value range into A's **columns** (spread `5.2e2` on layer 0 `q_proj`) while
`r3_actsearch` has only a per-**row** scale: **77.6% of A goes to zero** and the product lands at
relative error `1.5570`, worse than zeroing the matrix. Folding A's column norms into B's rows is
exactly identity-preserving (`2.3e-16`), free (B's format already carries a per-row scale), and
recovers `0.9874`. **Both were run and both are in the table** — repairing a registered
construction quietly is how E20 run 1 went wrong.

### 35.5 Additivity — three data, and E21 was the outlier

`excess = BPB(A+B) − [BPB(A) + BPB(B) − BPB(base)]`, band `±0.020`, registered before the run.

| composition | excess | score band | tf vs min(parts) |
|---|---|---|---|
| `QO512+V52` | **`+0.042909`** | SUPER-ADDITIVE | 126 vs 117 → RANK-**SUB**-ADDITIVE |
| `QO512-T+V52` | `+0.064067` | SUPER-ADDITIVE | 1 vs 1 (both at floor) |
| `QO512-TB+V52` | `+0.146241` | SUPER-ADDITIVE | 12 vs 28 → RANK-SUPER-ADDITIVE |
| *E21 `BOTH-ACT-256`* | *`−0.342498`* | *SUB-ADDITIVE* | *48 vs 68 → RANK-SUPER-ADDITIVE* |

All three of E22's are super-additive; E21's lone datum was sub-additive by seven times the
margin. E21 composed two cuts sharing **one objective** (overlapping errors); E22 composes rank
with sparsity (**different mechanisms**, interacting errors). **Neither is a law. Four
compositions is not a law either** — the operative rule is that *a stack must be measured, on both
metrics, because the two halves can disagree in either direction.*

### 35.6 §34.4 item 3, weakened

E21 wrote "BPB does not order interventions across axes" from one pair. E22 measured a second and
**the ordering held to the token**: `V52` costs `+0.141846` against E20's best ternary head's
`+0.170414`, and reads `117` teacher-forced against that head's `117`. The surviving claim is
weaker — *BPB ordering can fail across axes and must therefore be checked, not assumed.* §34.4
item 3 and `E21_CAN_RANK_BUY_IT.md` §4a are corrected in place.

### 35.7 Predictions

Three held (the three replication gates; `V52` ≥ 107; `STACK` at floor), **two missed** — both
additivity calls, and **both in the favourable direction**: the composition is super-additive in
BPB where sub-additive was registered, and ranks *above* its worse half where below was registered.
Prediction 4 was registered as a conflict between two published laws; BPB ordering won.

### 35.8 Owed

1. **Healing / QAT on `QO512+V52`** — first since §31.8 and now fully specified: a 1.5 B,
   budget-feasible at `0.9441 G`, right 79% per step, whose defect is drift and whose format
   conversion must be **learned**, not applied (§35.3: post-hoc it reads 28/160).
   **Needs GPU; the user launches it, and has offered T4 weeks for exactly this.**
2. **A real router for the carve** — every `V52` number here is an oracle ceiling.
3. **`k/v` in the factored arms**, left fp32 throughout §35.3, which makes that comparison
   favourable to E22; a like-for-like repeat widens the gap.
4. **A factored matvec in `engine.c`, a kind in `QWENDON1`** — unchanged from §34.6.
5. **The head** — the only organ neither cut touches, `545 M` on a 7 B, unchanged from §34.3.
6. Unchanged: the rank fraction at scale, `R5` in the exporter, the E14 and E19 items.

---

## 36. E23 — a real router: the carve survives contact, the COMPOSITION does not

`probes/E23_A_REAL_ROUTER.md`, brief `25bde22` pushed before the runner existed, runner
`7d33921`, results `engine/results/e23_router.json`. Seven arms, 2694 s, CPU only.
**Nothing exported, no timing taken; `6.79 tok/s` remains exact.**

### 36.1 What was replaced

Every carve number in this programme — D0, D0c, E19, §35's `V52` — used an **oracle** router:
it reads the true squared activation mass per expert group and keeps the top `k`. D0c §132–135
says so in its own text. E23 substitutes a per-layer **ridge regression from the block input to
`sqrt(group mass)`**, `λ = 0.01·mean(diag(XᵀX))`, closed form, no gradients, no tuning, fitted on
the frozen calibration slice and read on held-out. Only the SCORE changed: the hook is
`e19_carve_rank.install`'s, so every arm pays the identical top-`k` cost (`G-T4`, no failures).

### 36.2 The table

| arm | router | BPB | free | **tf** | mean rank | band-tf |
|---|---|---|---|---|---|---|
| `base` | — | 0.767595 | 160 | 160 | 1.00 | CHEAPER |
| `V52-ORACLE` | true mass | 0.909441 | 12 | 117 | 2.49 | COMPARABLE |
| `V52-STATIC` | top-`k` by mean calib mass | 1.191953 | 9 | **99** | 6.26 | WORSE |
| `V52-RANDOM` | fixed random | 2.156555 | 4 | 42 | 232.56 | WORSE |
| **`V52-LINEAR`** | ridge | 1.004558 | 7 | **110** | 4.17 | **COMPARABLE** |
| `QO512+V52-ORACLE` | true mass | 1.005039 | 15 | 126 | 2.03 | CHEAPER |
| **`QO512+V52-LINEAR`** | ridge | 1.201477 | 9 | **102** | 5.15 | **WORSE** |

`G-T0`, `G-T1`, `G-T2`, `G-T4` all fire; `VOID: none`. Both oracles reproduce E22 exactly.

### 36.3 The verdict is split, and the half that matters is the weaker one

`retention = (tf(LINEAR) − tf(RANDOM)) / (tf(ORACLE) − tf(RANDOM))`

| configuration | retention | BPB cost of a real router | band |
|---|---|---|---|
| `V52` alone | **0.9067** | `+0.095117` | **`ROUTER-HOLDS`** |
| `QO512+V52` | **0.7143** | `+0.196438` | **`ROUTER-COSTS`** |

The composed configuration is the one §35.8 item 1 proposed for GPU weeks. **It reads
`ROUTER-COSTS` and its teacher-forced count falls below E20's `107` floor.**

### 36.4 Composition reverses sign under a real router

| | `V52` | `QO512+V52` | composition |
|---|---|---|---|
| oracle | 117 | 126 | **+9** |
| ridge | 110 | 102 | **−8** |

§35.5 read the composed arm as RANK-SUB-ADDITIVE (`126 ≥ min(144,117)`). **That was
oracle-conditioned.** The oracle scores from true post-gate activations, so upstream damage
cannot reach it; the ridge router reads the **block input**, which the rank-512 attention cut has
already perturbed. The two cuts are independent under an oracle and **coupled through the
router's input** under a real one — a 17-token swing. Mean rank locates it: `2.03 → 5.15`
composed against `2.49 → 4.17` on the carve alone.

**Consequence for the ledger: §35.5's additivity table is a statement about oracle carves, and
is annotated as such. Five composition measurements now exist and they do not agree; there is
still no additivity law, and §35.5's own warning — "a stack MUST be measured" — is reinforced,
not replaced.**

### 36.5 The static router is most of the story, and Probe-4 does not forbid it

`V52-STATIC` — the same `k` groups for every token, chosen by mean mass on calibration —
reads **99/160**, retention 0.76, from a router that never looks at the token. **The entire
token-dependent routing decision is worth 11 teacher-forced tokens** (110 − 99); the gap to
random is 68. The brief predicted STATIC below 40, arguing from Probe-4's falsified hot-pool
(working sets ≈ i.i.d. across tokens). **That transfer is wrong: Probe-4's finding is about
per-NEURON working sets, and group mass over D0c's 256 co-activation groups is not i.i.d. on
this donor.** A static set is also free — no `11.0 M`, no per-token score — which makes it a
budget question and not only a quality one. Carried into E24 as a registered arm.

### 36.6 The re-derivation, and `k = 133` was never the budget-maximal depth

The router is charged at `1536 × 256 × 28 = 11,010,048` = `11.0 M`, active every token.

| component | active weights/token |
|---|---|
| `q/o` at rank 512 | 88.1 M |
| `k/v` | 22.0 M |
| head | 233.4 M |
| router | 11.0 M |
| **fixed subtotal** | **354.5 M** |
| FFN, full | 1156.1 M |

Against §31's `0.982–1.060 G`, the FFN allowance is **`627.5–705.5 M` = 54.28%–61.03%
activation = `k = 139 … 156` of 256.**

| `k` | activation | total | |
|---|---|---|---|
| **133** (E19, §35, E23) | 0.5195 | **0.9551 G** | **2.8% UNDER the floor** |
| 139 | 0.5430 | 0.9822 G | in budget |
| 148 | 0.5781 | 1.0228 G | in budget |
| 156 | 0.6094 | 1.0590 G | in budget |

**`k = 133` was inherited from E19, where it was fixed before any router was charged to
anything.** So the shallower carve `ROUTER-COSTS` demands is not a retreat — it is the depth the
budget always permitted, and it is free quality. **Whether `k = 139…156` buys the 5 teacher-forced
tokens needed to clear `107` is NOT predicted here; it is what E24 measures.**

### 36.7 Predictions

Three held, two missed. Prediction 3 missed on the arm it named (registered `ROUTER-COSTS`, read
`ROUTER-HOLDS` at 0.9067) and **held on the composed arm** (0.7143, inside the band) — reported
that way rather than as "the band basically held". Prediction 2 is the substantive miss (§36.5).
**The registered alternative fired on `V52-LINEAR` only**: the carve is not an oracle artefact,
but its second half — that `QO512+V52` is "constructible without an oracle" — is refused.

### 36.8 Owed

1. **E24 — the depth sweep the verdict requires**: ridge router at `k ∈ {139, 148, 156}`, plus
   `V52-STATIC` at the same depths and a `log1p`-target router (§36.5 and the declined variant,
   which moved L27 overlap `0.7404 → 0.8197` on the pre-run diagnostic and was **not** adopted
   because it was not registered). **CPU, mine.**
2. **H0 is NOT blocked by any of this** and is deliberately router-free — §36 changes nothing
   about it. See `decisions/T4_HEALING_PROPOSAL.md` §3.
3. Unchanged from §35.8: `k/v` in the factored arms, a factored matvec in `engine.c` and a kind
   in `QWENDON1`, the head, the rank fraction at scale, `R5` in the exporter, the E14/E19 items.


## 37. E25 — the factored matvec exists, and the rank lever pays exactly what it weighs

**Probe**: `probes/E25_WHAT_THE_RANK_COSTS.md`. **Brief**: `briefs/BRIEF_E25_WHAT_THE_RANK_COSTS_IN_THE_ENGINE.md`, pushed at `e506c7b` before the runner ran.
**Verdict `RANK-PAYS-WHAT-IT-WEIGHS`.** Results: `engine/results/e25_rank_cost.json`, `engine/results/e25_parity_factored.json`.

### 37.1 The item that was owed since E21 §8

E21, E22, E23 and the T4 proposal each closed with the same sentence: **`engine.c` has no
factored matvec.** Both E21 and E22 measured the rank cut by computing `A·B` and installing a
DENSE matrix — the right way to price its **quality**, and no way at all to price its **cost**.
Everything the rank axis contributes to tok/s — §31's budget line, E23 §7's `k = 139…156`, the
reason `q/o` at rank 512 is in the plan — rested on `2·D·r < D²`, which nothing had executed.
**It has now been executed and measured.**

`donor_engine.c` gains a factored `mat_t` kind (`y = A·(s ⊙ (B·x)) + b`, two calls to the same
kernels, intermediate of size `r` in its own buffer) and `quant == 3`, a **tagged container**
where every matrix carries its own `int32` kind. The container is what makes the measured
configurations expressible at all: E21/E22/E23 all left `k/v` **fp32** while cutting `q/o`, and a
single global `quant` flag cannot say that. **Until `quant == 3` existed, every rank result in
this ledger was a PyTorch number with no runnable artifact behind it.**

### 37.2 Correctness first — `G-E25P` fires

Worst relative l2 **`6.445e-04`** against a `2e-3` bar, top-1 agreement **`1.0000`** on 10/10
positions, on Qwen2.5-1.5B with `h0_factors.npz` installed on `q/o` — E22's `QO512-TB`, the
state H0 starts from — against a PyTorch reference running `h0_qat.TernaryLowRank`, the same
module the T4 trains. `G-E25a` separately holds the legacy paths still: patched and unpatched
engines produce **bit-identical** logits on `qwen25-15b_tq.bin`, sha256
`e09b30c847f3956142fb3bc670214cb13aba327f`.

### 37.3 The measurement, and the planted control that makes it readable

Idle box, `--threads 6`, 3 repetitions interleaved by rep, dispersion with every rate, `--fuse`
off everywhere. Synthetic shapes (`synth_export.py`): weights are noise, which E3 §4's Gate V1
established is legitimate for timing and illegitimate for anything else.

| arm | active/token | tok/s | spread | vs `R0` | byte prediction |
|---|---|---|---|---|---|
| `S15-PACKED` (untagged) | 1.5436 G | 29.70 | 2.1% | +0.67% | 0 |
| `S15-TAG-R0` | 1.5436 G | 29.50 | 1.3% | — | 0 |
| **`S15-R768`** byte-neutral | 1.5436 G | 28.78 | 6.4% | −2.46% | 0 |
| `S15-R512` | 1.4995 G | 30.40 | 3.2% | +3.03% | +2.94% |
| `S15-R256` | 1.4555 G | 30.51 | 3.3% | +3.42% | +6.05% |
| `T10-TAG-R0` | 10.6032 G | **4.70** | 6.6% | — | 0 |
| **`T10-R2048`** byte-neutral | 10.6032 G | 4.68 | 7.3% | −0.43% | 0 |
| `T10-R512` | 9.3952 G | **5.36** | 3.7% | **+14.12%** | +12.86% |
| `T10-R256` | 9.1939 G | 5.43 | 6.8% | +15.68% | +15.33% |

**The planted control.** At `r = D/2` a square projection is byte-neutral **to the last weight**
(`2·D·(D/2) = D²`), so whatever the `R768`/`R2048` arms lose IS the factored path's own cost —
the second call, the extra OpenMP region, the intermediate. They read `−2.46%` and `−0.43%` at
spreads of 6.4% and 7.3%: **neither resolvable from zero.** The extra 56 / 96 regions per token
cost less than this box can measure. *(Disclosure: `S15-R768`'s `−2.46%` is carried by one rep —
27.55, then 29.39, 29.39; reps 2–3 alone give `−0.74%`. The outlier is kept.)*

**The container is free**: `+0.67%` at spreads of 1.3–2.1%, so no tagged number carries an
offset.

### 37.4 The invariant — charged throughput does not care how the weights are arranged

| shape | `G active weights/s` across arms | band | charged GB/s |
|---|---|---|---|
| `S15` | 44.41 … 45.84 (mean 45.16) | **3.17%** | 22.58 |
| `T10` | 49.59 … 50.36 (mean **49.92**) | **1.54%** | 24.96 |

**Four `T10` arms whose rates differ by 15% deliver charged throughput inside 1.54%.** The rank
axis buys speed through exactly one mechanism — fewer weights — and dense-vs-factored,
one-call-vs-two changes nothing else. **This is the statement worth keeping**, because it
predicts arms that were never run, and it is why `2·D·r` is the right charge.

`GB/s` here is **charged** bytes at the packed format's exactly 0.500000 B/weight, never moved
bytes; the two conventions must not meet in a fraction (§19.3's law).

### 37.5 §31's budget line, checked at the goal's shape for the first time

§31 derived that 50 tok/s needs **`0.982–1.060 G` active ternary weights/token**. At `T10`'s
measured **`49.80 G weights/s`** that band maps to **47.0 – 50.7 tok/s**. The budget was built
from smaller shapes; **it brackets 50 at the goal's dimensions, measured.**

### 37.6 The distance to the goal, measured instead of derived

**`T10` — 10.60 G active weights, the goal's "es 10B" dimensions — runs at 4.70 tok/s on this
box. The target is 50. That is 10.6× away, and rank-512 `q/o` closes it to 9.3×.**

Rank on `q/o` is worth **1.2 G** of the 9.6 G that must come out. **The FFN is 8.45 G of `T10`
and that is nine tenths of the remaining problem** — which is precisely where E19's carve and
E23/E24's router are aimed, and **E24 is still unrun**.

Two things keep this number from being worse than it looks and one keeps it from being better:
`T10` carries **Mistral's 32,768 vocabulary**, so its head is 134 M — a Qwen-vocabulary 10 B
carries 622 M and would be slower (INDEX's head table); the weights are noise, so **`T10` is a
shape and not a model**; and `T10-R512` is `r/D = 1/8`, where E21 validated only `r/D = 1/3` at
`D = 1536`, so **nothing here says a model survives that rank**.

**`6.79 tok/s` is untouched** — that is the real 7.072 B packed donor, a different artifact and
a real one.

---

## 38. E26 — a gathered weight costs more than a streamed one, and §37's invariant does not extend to the carve

**Probe**: `probes/E26_WHAT_AN_ACTIVATED_WEIGHT_COSTS.md`. **Brief**:
`briefs/BRIEF_E26_WHAT_AN_ACTIVATED_WEIGHT_COSTS.md`, pushed before the runner existed.
**Results**: part A `engine/results/e26_parity_carve.json`, part B
`engine/results/e26_carve_cost.json`. The VOID first attempt is kept as
`engine/results/e26_carve_cost_contended.json`.

### 38.1 What §37 established and what §38 takes back

§37.3 is the load-bearing paragraph of this ledger's budget arithmetic: at `T10`, four arms whose
**rates** differ by 15% delivered **charged throughput inside a 1.54% band** (49.59-50.36 G active
weights/s). That is what licenses charging a rank cut `2*D*r` and reading tok/s straight off a
weight count.

**E26 asks whether the same holds when the weights are GATHERED instead of streamed, and it does
not.**

| | E25, rank | E26, carve `T10` | E26, carve `S15` |
|---|---|---|---|
| charged-throughput band | **1.54%** | **12.54%** | **18.00%** |
| range, G active weights/s | 49.59-50.36 | 38.03-43.12 | 34.34-41.12 |

**The charge model `3*D*GSZ*k` is therefore optimistic**, and every `k` in every budget table in
this programme -- 31 (E18), 32 (E19), 36 (E23 s7), E24, E27 -- inherits that.

### 38.2 The planted control, and this time it resolves

At `k = E` the carve keeps every group, so it moves the dense arm's weights plus the router:
byte-neutral to a **registered** `-0.47%` at `T10` and `-0.71%` at `S15`. What it loses below that
offset is the carve machinery itself -- the router matvec, the top-`k`, the row list, the gather,
the transposed kernel's access pattern.

| shape | `k = E` measured | bytes predicted | **machinery's own cost** |
|---|---|---|---|
| `S15` | -4.98% | -0.71% | **-4.30%** |
| `T10` | -6.80% | -0.47% | **-6.36%** |

**Contrast with 37.3.** E25's planted control at `r = D/2` read `-0.43%` / `-2.46%` with
dispersions of 7.3% / 6.4% -- **not resolvable from zero**, i.e. the rank path's own cost is below
what this box can measure. **The carve path's own cost IS resolvable**: 4-6%, paid before a single
group is dropped, and charged to nothing anywhere in this ledger until now.

A second control separates the machinery from the format it lives in: `S15-PACKED` vs `S15-DENSE`
are the same weights in the untagged and the `quant == 4` container with no carving, and read
**1.0056** (paired 1.0079) against a byte prediction of exactly 1.0000. **The container is free;
the carve is not.**

### 38.3 The shape of the loss -- a U, and why the registered trigger missed it

The brief registered: *if the deep arms (`K16`, `K4`) fall below 40 G weights/s, a gathered weight
costs more than a streamed one.* **`T10-K16` reads 41.41 and `T10-K4` 41.67. The trigger does NOT
fire**, and it is not claimed that it did. It was the wrong statistic. Charged throughput relative
to each shape's own dense control:

| `k` | `T10` | `S15` |
|---|---|---|
| 256 | 0.936 | 0.957 |
| 128 | 0.902 | 0.835 |
| 64 | **0.882** | 0.879 |
| 16 | 0.960 | 0.880 |
| 4 | 0.967 | 0.941 |

**At small `k` the FFN is a small share of the token** -- 5.7% of active weights at `T10-K4` -- so
the arm is nearly all dense non-FFN floor and its charged throughput returns to the dense rate
whatever the gathered path costs. **The dense floor masks the effect exactly where the trigger was
looking.** The band was the right statistic and it was already in the brief.

Attributing the floor to the dense rate and solving `1/r = (1-f) + f/x` for the gathered FFN's own
efficiency `x`, with intervals from each arm's own dispersion:

| arm | FFN share | `x` | interval |
|---|---|---|---|
| `S15-PACKED` | 0.749 | **1.008** | [0.969, 1.047] -- the null behaves |
| `S15-K256` | 0.751 | 0.943 | [0.903, 0.985] |
| `S15-K128` | 0.603 | 0.754 | [0.701, 0.808] |
| `S15-K16` | 0.177 | **0.564** | [0.479, 0.673] |
| `T10-K256` | 0.794 | 0.921 | [0.831, 1.015] |
| `T10-K64` | 0.490 | 0.786 | [0.724, 0.852] |
| `T10-K16` | 0.194 | 0.825 | [0.740, 0.925] |
| `T10-K4` | 0.057 | 0.621 | [0.390, 1.339] -- **not resolvable, do not quote** |

**This is a post-hoc model and per E14 s6 it is not a gate.** It also cannot separate the gather's
locality cost from fixed per-token selection overhead; both produce this shape. 38.6 item 1 is the
arm that would.

### 38.4 Row length is the variable -- prediction 5, registered as a direction, lands

A kept row is **768 B at `S15` and 2048 B at `T10`**. The brief predicted the short-row shape would
suffer more, and it does, on three independent comparisons: the band is **18.00% at `S15` vs
12.54% at `T10`**; gathered-FFN efficiency at `k = 16` is **0.564 vs 0.825** with **disjoint**
intervals; and net of machinery the `S15` arms fall 17-24 points short of their byte prediction
while the `T10` deep arms overshoot theirs.

**So it is a locality cost with a lever attached**: a coarser carve -- fewer, longer runs for the
same activated fraction -- should recover part of it, and that is an exporter change, not a kernel
change.

### 38.5 What it costs the tables, and what it does not change

The correction multiplies the **FFN term**, not the token:

- **E24's `K156` at `T10`** (E24 s6: `6.14 G` ~ 8.1 tok/s): `T10`'s ratio interpolates to ~0.91
  between `K128` 0.902 and `K256` 0.936, so **~7.4 tok/s**.
- **E27's `FLOOR-MIN`** at `k = 17.3`, where `T10` measures 0.960: `9.37` becomes **~9.0**.
- **In general**, at the depths the 50 tok/s budget permits at `T10` (`k ~ 2...18`) the correction
  on the token is **3-5%** because the floor dominates, but on the **FFN allowance itself** it is
  **~17%** (`x ~ 0.83`) -- the budget permits about a sixth fewer groups than the charge model says.

**No verdict moves.** 32's "FFN-only carving cannot reach the target", E24's `DEPTH-RECOVERS` and
E27's `FLOOR-IS-NOT-ENOUGH` are negative or quality-conditioned, and a correction that makes the
carve worse strengthens all three. What changes is that carved tok/s figures are now known to be
optimistic and by how much.

### 38.6 Validity, stated as a constraint and not a footnote

The first attempt was run with a game open, 5.82 of 12 cores held, and was **VOIDed**. This run
carries three idleness instruments:

1. **The witness, rebuilt.** The old one gated on the *max* of a short sample at 12% and refused an
   idle desktop three times running. It now gates on the **mean**, and `--selftest` is a planted
   control on the instrument: quiet **12.1%**, six spinning processes **61.5%**, fires on the
   known-positive and accepts the quiet box. The bar moved to 25% **by derivation between those two
   populations**, with the reasoning and the admission written above `IDLE_BAR`. In-run readings:
   10.6 / 5.8 / 6.5 / 8.0% mean.
2. **The E25 anchor, which is the real instrument and which nearly failed.** `S15-PACKED` reads
   **26.79 tok/s against E25's published 29.70 -- `-9.81%` against a `+-10%` bar.** It passed by
   0.19 of a point. Both `S15` arms decline monotonically across reps (28.04/27.53/24.34 and
   27.27/27.32/25.77), which is thermal drift, not contention.
3. **The pairing check.** Because arms are interleaved by rep, a common-mode drift cancels in the
   ratios. Verified: recomputing every ratio paired within each rep changes nothing (`T10-K256`
   0.9312 paired vs 0.9320, largest disagreement anywhere 0.4%).

**Therefore: this record is a RATIO record.** No absolute tok/s from it enters this ledger, and the
registered absolute form of prediction 3 (*every carved arm inside 45-55 G weights/s*) is
**untestable here** -- the dense control itself reads 43.12 where E25 read ~49.9 -- so its failure
is not counted as evidence.

**Owed**: (1) one arm that runs the selection and then uses every group anyway, to split fixed
overhead from the gather; (2) the coarse-granularity arm 38.4 implies; (3) a cold-box re-run with
cooldowns between reps, which would make the absolute form testable; (4) **the E25 anchor should
become standard equipment on every timing probe here** -- a CPU percentage is a pre-filter, a
reproduced published rate is the instrument.

**`6.79 tok/s` is untouched** -- that is the real 7.072 B packed donor, and nothing here is a
measurement of it.

## 39. E28 — the numerator was never a constant: 49.9 becomes 61.64 at the goal's shape

**Probe**: `probes/E28_DOES_THE_KERNEL_TRANSFER.md`. **Brief**:
`briefs/BRIEF_E28_THE_OTHER_FACTOR.md`, pushed at `bb48974` before the runner existed.
**Result**: `engine/results/e28_kernel_transfer.json`. Idle box, witness 5.5% mean / 14% peak
against a 25% bar; 254 s, six arms, three interleaved reps.

### 39.1 The constant this ledger has charged everything against since §10

`THROUGHPUT_G = 49.9` G active weights/s is the number that turns a weight count into a token
rate here, and §37.3 hardened it: four arms differing 15% in rate delivered charged throughput
inside **1.54%**. §38 found the first thing it does not cover (the carve). **§39 finds that it was
never a property of the machine at all -- it is a property of one kernel and one LAYOUT.**

| shape | packed | `--lut` | `--lutblk` | best charged G-w/s |
|---|---|---|---|---|
| S15 | 29.30 tok/s | 22.54 | **36.80** | **56.81** |
| T10 | 4.35 tok/s | 2.49 | **5.81** | **61.64** |

**61.64 against 49.9 is 1.235x**, and the whole of it comes from E13's blocked tile-major layout
running at the goal's shape for the first time.

### 39.2 The lever GROWS with the shape -- the only thing in this programme that has

0.5 B **1.217x** (E13, published) -> S15 **1.258x** -> T10 **1.338x**, all end-to-end
`--lutblk` over the packed default, the last two paired within reps here. Every other lever in
this programme has either failed to transfer upward or transferred as a fraction and lost its
absolute value doing it.

### 39.3 The budget, and the condition that keeps it from being spendable

`50 tok/s` has meant `<= 0.998 G active/token` (`49.9 / 50`) in every table here. On the measured
numerator it is **`1.2328 G/token`, 23.5% more room**: E27's `FLOOR-MIN` goes from `k = 17.3` to
`k = 26.8` of 256, `QO192+KV96` from `8.6` to `15.7`.

**NONE OF THOSE ARMS CAN USE THE KERNEL THAT PRODUCED THE NUMBER.** `donor_engine.c:1437`
refuses `--lut` on any container that is not `quant==2`, and rank is `quant==3`, carve is
`quant==4`. So §39.3 is a budget that WOULD exist if the fast kernel composed with the quality
levers, and today it composes with none of them. That gap is now measured at **1.34x** and is
the first item this ledger owes forward.

### 39.4 Honesty about the band name

The registered verdict cell `T10-LUTBLK / T10-PACKED` reads **1.3354** (means) / **1.3378**
(paired), against a registered boundary at 1.30 -- but the three reps read **1.2022, 1.3640,
1.4471**, so the boundary lies INSIDE the dispersion and rep 1 alone would have named the other
band. The reps were not dropped and will not be. **The existence and rough size of the lever are
solid (every rep >= 1.20, known-positive fires at 1.633); the band name is not.**

### 39.5 What it does to the goal

`4.70 -> 5.81 tok/s` dense at `T10`; E27's best COMPARABLE arm scaled by the lever reads
**10.35 tok/s**. **The gap to 50 tok/s closes from 10.6x to 8.60x** -- the first time since §10
that it has moved from the engine side, and a long way from enough. Prediction 5 of the brief
registered exactly this before the run.

---

## 40. E30 — the wall is measured: 36.3 GB/s, and the engine is already at 0.909 of it

**Probe**: `probes/E30_IS_THE_ENGINE_AT_THE_WALL.md`. **Brief**: `briefs/BRIEF_E30_THE_WALL.md`,
pushed at `63515b4` before the runner existed. **Result**: `engine/results/e30_the_wall.json`.
Idle box, witness 2.8% mean / 7% peak against a 25% bar; 94 s.
**Instrument**: `engine/e30_bandwidth.c`, read-only streaming sum, engine flags, no `-ffast-math`.

### 40.1 The denominator this ledger never had

Every rate in §§10-39 was divided by something implicit. E30 measures it: **`BW-CEIL = 36.30
GB/s`**, the median of four buffer sizes >= 1 GB, 6 threads, on this box. The planted control
`G-E30A` fires at **12.5x** — L3-resident 454.3 GB/s against the DRAM plateau — and the cliff
falls at **16 MB**, exactly where Probe-3 put it.

### 40.2 The engine against it, in MOVED BYTES on both sides

T10 moves **5,312,757,760 bytes/token** = 5.31276 GB, and `G-E30B` reproduces the artifact **to
the byte** (`5,849,628,724`, unaccounted **0**). *The brief's own pre-registered table was short
by the q/k/v fp32 biases, 24,576 B/layer — 5.3116 vs 5.31276, +0.022%; the gate caught it.*

| arm | tok/s | GB/s moved | ÷ `BW-CEIL` |
|---|---|---|---|
| `T10-PACKED` | 4.363 | 23.18 | **0.639** |
| `T10-LUTBLK` | 6.213 | 33.01 | **0.909** |
| *perfect kernel* | *6.832* | *36.30* | *1.000* |
| *the goal* | *50* | *265.64* | ***7.318*** |

**Verdict `AT-THE-WALL`.** E10's "core-bound" was right *for the kernel it tested* — packed still
leaves a third of the machine on the floor — and what moved is the kernel, not the machine.
**Both E28 anchors pass** (`+0.23%`, `+6.88%`), so §39's readings and these come off the same box.

### 40.3 The number that supersedes `THROUGHPUT_G`

`THROUGHPUT_G = 49.9` (§10) became 61.64 in §39. It is now bounded from above, permanently, on
this box:

| budget for 50 tok/s at T10 | G weights/token |
|---|---|
| §10 numerator 49.9 G-w/s | 0.998 |
| §39 numerator 61.64 G-w/s | 1.2328 |
| **§40 — the physical bound, perfect kernel, 100% of measured bandwidth** | **1.452** |

**§39's widened budget is already 85% of the physical bound: the numerator has `1.178x` left in
it in total, forever, here.** E27 measured that everything inside 0.998 G is broken and everything
that holds sits 5-8x away; `1.45x` does not close that.

### 40.4 The cliff is the constructive half

A 16 MB resident core costs **0.035 ms of a 20 ms token — 0.18%, free**. The specification of the
goal on this machine is therefore an inequality in two currencies, priced together for the first
time:

> **~32 M weights resident and free, plus <= 1.45 G streamed, per token, for 50 tok/s.**

Which is Probe-3's "<=16 MB active slice ~ 24-40 M ternary params/token" arriving from the
bandwidth side, and `SCALEUP_ARCHITECTURE.md`'s thinking/knowing split written as a bound.

### 40.5 Honesty, and what it changes in the plan

The cell landed **0.059 above** the 0.85 boundary and the **denominator** is the noisy side: three
extra sweeps (post-hoc, not a gate, the registered cell stands) give a pooled median of 37.81
(ratio 0.873, same band) and a best-ever 40.81 (**ratio 0.809, `PARTIALLY-BOUND`**). **Against the
most generous denominator this box has ever produced the engine still uses 81% of its entire read
bandwidth**, headroom bound **1.24x**, and the brief wrote one disposition for both bands.
Predictions **4 HIT / 1 MISS** — the miss is the band name, by 0.009.

**Plan change recorded here**: §39's owed item 1 (the per-matrix `--lut` guard at
`donor_engine.c:1437`) is **demoted from "the next build" to "worth doing"** — it buys
byte-EFFICIENCY, bounded by 1.24x, not bytes. **Every subsequent probe belongs on the denominator,
against a hard target of 1.45 G moved weights/token.**

---

## 41. E31 — a gathered byte costs 42% of the budget at row granularity, and nothing at 32 KB

**Probe**: `probes/E31_WHAT_A_GATHERED_BYTE_COSTS.md`. **Brief**:
`briefs/BRIEF_E31_THE_GATHERED_BYTE.md`, pushed at `8003fa3` before the runner existed and
before the instrument had ever been timed. **Result**: `engine/results/e31_gathered_byte.json`
(run 4; runs 1-3 VOID and kept). Witness 7.2% mean / 14% peak against a 25% bar; 12 s.
**Instrument**: `engine/e31_gather.c`.

### 41.1 The denominator's own ceiling

Section 40 left one variable -- bytes per token, at most **1.45 G moved weights** for 50 tok/s --
and **that number came off a DENSE stream**. Every road to it (MoE, carve, sparsity) reads a
SUBSET, and a subset is not read the way a matrix is read. 2 GB buffer, one group in eight
(256 MB useful, far outside the 16 MB L3 line of section 40), 6 threads, 9 INTERLEAVED reps.

| group | `sorted` GB/s | `contig` GB/s | **`sorted/contig`** | budget for 50 tok/s | % of T10's dense token |
|---|---|---|---|---|---|
| 64 B (a cache line) | 9.65 | 36.60 | **0.264** | 0.383 G | 3.61% |
| 768 B (an S15 FFN row) | 18.21 | 38.83 | **0.469** | 0.681 G | 6.42% |
| **2048 B (a T10 FFN row)** | 21.50 | 37.14 | **0.579** | **0.840 G** | **7.93%** |
| 8 KB (4 rows) | 31.31 | 37.21 | 0.842 | 1.222 G | 11.52% |
| **32 KB (16 rows)** | 34.37 | 36.41 | **0.944** | **1.371 G** | **12.93%** |
| 128 KB | 34.06 | 34.75 | 0.980 | 1.423 G | 13.42% |
| *dense (section 40)* | | | *1.000* | *1.452 G* | *13.69%* |

**Verdict `GATHER-COSTS`.** Both gates fire: `G-E31A` (same-volume control, worst `-6.5%`, spread
`11.2%` on a `+-15%` bar) and `G-E31B` (E26's known positive -- the 768 B S15 row MUST read below
the 2048 B T10 row, and does).

### 41.2 The design statement

> **Neuron-granular sparsity costs 42% of the budget. Activating in blocks of >= 16 consecutive
> rows costs 6%. The difference between those two designs is `1.63x` on the only axis the goal
> still has** -- against the `1.24x` that was all the numerator had left after section 40.

Crossover to 0.90 of dense: **32 KB**, bracketed in `(8 KB, 32 KB]`. Ordering matters only below
8 KB (`sorted/random` = `1.34x` at 64 B, within 5% from 8 KB up); **a naive unsorted gather at
cache-line granularity delivers `0.197` of the machine, the worst number in this programme.**

With section 40's resident half, the goal is now fully priced in one line:

> **~32 M weights resident and free, plus <= 0.84 G streamed at row granularity -- or <= 1.37 G
> if the architecture activates in >= 32 KB contiguous blocks. Per token, for 50 tok/s.**

### 41.3 The lever this promotes -- CORRECTED the same day

**First reading, and it was wrong**: E26 section 9 had left "a coarser carve, and it is a change
to the EXPORTER, not the engine" unpriced, and E31's first write-up priced it at `2.01x` on the
assumption that the carve reads one FFN ROW at a time. **Reading the code instead of my own
summary says it does not.** `synth_export.py:266` writes `gate`/`up` group-major, so **a carve
group is ALREADY one contiguous run** -- 26.9 KB at S15 with `E=256`, 114.7 KB at T10, both past
this probe's crossover and nearly free. The fine granularity is **`down`**, which
`donor_engine.c:864` REQUIRES to be `MK_PACKED_T` at **`PT_BLK = 64`, exactly one cache line per
selected neuron**: a group contributes `GSZ x 64` bytes -- **2,240 B at S15, 3,584 B at T10**.

Applying 41.1's measured curve to that layout (**desk arithmetic on a measured table, not a
measurement**):

| | `gate`/`up` run | `down` run | relative time |
|---|---|---|---|
| S15 `E=256` (E26's setting) | 26.9 KB, `r=0.929` | **2,240 B, `r=0.596`** | 3.830 |
| S15 `E=16` | 430 KB, `r=1.000` | 35.8 KB, `r=0.946` | 3.057 |
| T10 `E=256` | 114.7 KB, `r=0.976` | **3,584 B, `r=0.685`** | 3.508 |
| T10 `E=16` | 1.84 MB, `r=0.989` | 57.3 KB, `r=0.959` | 3.065 |

**The coarse-carve lever is `1.25x` at S15 and `1.14x` at T10, not `2.01x`** -- comparable to the
`1.24x` section 40 left in the numerator, not dwarfing it. The direction survives; the size does
not, and "the largest single lever left in the programme" is withdrawn.

**What the correction gains is a sharper target**: `gate` and `up` are already effectively free
and **`down` alone carries the whole carve locality cost.** It also exposes a knob the first
reading never saw -- **`PT_BLK`**. At `PT_BLK = 512` a neuron's `down` contribution is 512 B and
an `E=256` group is 17.9 KB (`r ~ 0.90`), worth `1.17x` at S15 **without touching the carve** --
but `PT_BLK` is a compiled kernel constant, so that is an engine change with a kernel rewrite
behind it, not the exporter-only change first advertised.

And E18's "50 tok/s is a ~10%-activation budget", read from the numerator side, meets **7.93% /
12.93%** read from the memory side. The two roads arrive at the same number.

### 41.4 Three VOID runs, and the gate was never widened

Run 1 `-15.4%`, spread `24.4%`: **the control walked the offset array**, so it carried the
gathered arms' own per-group bookkeeping and moved with the variable it controlled for -- while
sitting in the DENOMINATOR of every ratio. Fixed to one unbroken run (the stricter reading);
the old behaviour kept as `contig_ind`, a DIAGNOSTIC. Run 2 spread `17.4%`, run 3 spread `29.5%`
**with reps raised 3 -> 15** -- and that non-response is the diagnosis: `contig` is identical
code over identical volume at every row, so the spread was block-correlated drift, because each
arm's reps sat in one contiguous wall-clock window. **Reps moved OUTERMOST** (section 39's own
interleaving methodology, which this instrument had exactly backwards). Run 4 fires at `11.2%`.
Predictions **5 HIT / 1 PARTIAL** -- the partial is mine: I put the ordering crossover `16x` too
coarse.



---

## 42. E32 -- `--lutblk` does NOT survive its own protocol (`ACTIVATION-COSTLY`, `+0.048991`)

**Probe** `probes/E32_THE_OWED_PROTOCOL.md` | **brief** `briefs/BRIEF_E32_THE_OWED_PROTOCOL.md`
(`1fb4c18`) | **runner** `engine/e32_owed_protocol.py` (`f5e773b`) | **result**
`engine/results/e32_owed_protocol.json`. 3,211 s, eight arms. **Quality only -- no timing.**

### 42.1 The re-run three probes had been carrying as owed

E14 disqualified its own verdict in its own section 1: with no `--seqlen` the engine scores the
whole ids file as ONE sequence (`donor_engine.c:1556`), so E14 read a single 12,288-token
sequence against bands drawn from 512-context regimes. E14 section 5, E28 section 8 and E30
section 8 all carried the re-run. **Meanwhile the whole engine-side story -- 61.64 G-w/s, 33.01
GB/s, `0.909` of the ceiling -- was measured on the arm that verdict licensed.**

All three gates fire: E1's own published anchor (`A0(0.5B)` 4.531236572 vs `e1_05b.log:31`'s
4.531233734, **2.8e-6** on a 1e-3 bar), the known positive (`A3 - A1 = +0.000e+00` at BOTH
cells), and the readability gate (`A0(1.5B)` sits `-0.594113` below the 4.069819 chance line).

### 42.2 The sign flipped

| cell | `dBPB(A1)` = `dBPB(A3)` | `dBPB(A2)` | E14 at 12 k |
|---|---|---|---|
| 0.5 B (instrument-only, `A0` is `+0.461` ABOVE chance) | `-0.113201139` | `+0.001180774` | `-0.313878918` |
| **1.5 B (the verdict cell)** | **`+0.048990745`** | `+0.011739023` | `-0.016961262` |

`+0.049` is **2.4x** E14's own `ACTIVATION-COSTLY` line of `0.020` and about **10x** the
`sigma_seed ~ 0.005` constant -- and that flatters it, because the arms share weights and inputs,
so the difference is DETERMINISTIC, not a draw. `A0(1.5B)` rose `+0.029331` going from 12 k to
512, the sane direction for a harder prediction problem.

### 42.3 The operative numerator falls back to packed

| | `--lutblk` as published | packed, operative |
|---|---|---|
| T10 rate (E30's session) | 6.213 tok/s | **4.363 tok/s** |
| of the measured 36.30 GB/s ceiling | `0.909` | **`0.639`** |
| charged numerator (E28's session) | 61.64 G-w/s | **46.16 G-w/s** |
| gap to 50 tok/s at T10 | `8.05x` | **`11.46x`** |

Rows pair arms measured in ONE session. **No architectural conclusion moves** -- E32 prediction 6
registered that before the number existed: a perfect kernel at T10 still reads 6.83 tok/s, 50
tok/s still needs `7.32x` the machine's whole read bandwidth. What changes is that the engine is
**`1.57x`** from the wall, not `1.10x`: MORE kernel headroom than E30 reported, and it has to be
bought without `--lutblk`'s quality cost. Correction banners added at the TOP of E28 and E30, as
the brief required.

### 42.4 The constructive half: the cost is the SCALE GRANULARITY, not int8

`--lut --lut-group 32` costs `+0.011739` where `--lutblk` costs `+0.048991`: **finer activation
scaling recovers 76% of the loss.** The same shape as E31's finding on the read side -- the
expensive thing was never "sparsity" or "int8", it is the GRANULARITY. **And `A2` has never been
timed** (E14 section 0, still owed): E28 timed `--lut` (S15 22.54, slower than packed's 29.30)
and `--lutblk` (36.80), never `--lut --lut-group 32`. The fast kernel is the costly one and the
cheap kernel is unmeasured -- now the highest-value measurement left in the engine branch.

### 42.5 Predictions 3 HIT / 2 MISS, and the misses are mine on the cell that matters

I registered that the verdict would NOT flip and that `dBPB(A3)` would land in `[-0.015, +0.005]`.
It flipped, and landed at `+0.049`. The load-bearing error is quoted in the probe: *"E14's
arm-vs-arm comparisons were protocol-invariant by construction."* **They are not** -- the
activation distribution is a function of context length, so the quantiser sees different vectors
at 512 than at 12 k, and `dBPB` was never a property of the kernel alone.

The RANK partner (E14 section 3's law) is **verified at source rather than assumed**:
`e14_gn3_greedy.py` drives `--generate`, and that branch returns at `donor_engine.c:1513`, BEFORE
`SL` is computed at `:1556`, so `--seqlen` is unreachable from the generation path and E14's
`45.6%` top-1 agreement stands at this protocol. **Both halves now agree: the score says costly
and the rank says broken.** The score/rank divergence that made E14 hard to read was itself an
artifact of the contaminated protocol.


---

## 43. E33 -- the coarse-carve lever is `1.12x` at S15 and NOT RESOLVABLE at the goal's shape

**Probe** `probes/E33_IS_DOWN_THE_WHOLE_COST.md` | **brief** `briefs/BRIEF_E33_IS_DOWN_THE_WHOLE_COST.md`
(`2fa83e9`) + **addendum** `briefs/BRIEF_E33_ADDENDUM_TWO_GATES_CANNOT_FIRE.md` (`c6090f4`, pushed
before any artifact was exported) | **runner** `engine/e33_down_locality.py` (`f5e773b`) |
**result** `engine/results/e33_down_locality.json`. 144 s, six arms, five interleaved reps.

### 43.1 Verdict `LOCALITY-PARTIAL`

| cell | raw | normalised | band | desk model | fraction realised |
|---|---|---|---|---|---|
| **`S15-E16 / S15-E256`** | **1.1334** | **1.1164** | `LOCALITY-PARTIAL` | 1.253 | **46%** |
| `T10-E16 / T10-E256` | 1.0548 | **1.0432** | below the `1.06` line | 1.145 | 30% |

Jackknife (leave one rep out, five ways, normalised): S15 `1.089 / 1.115 / 1.125 / 1.108 / 1.145`
-- **`LOCALITY-PARTIAL` every time, and under the median too**. T10 `1.040 / 1.049 / 1.029 /
1.053 / 1.045` -- **below `1.06` every time**, with the RAW reading flipping band across
jackknives. **At the goal's own shape the exporter knob is worth nothing measurable.**

### 43.2 Third cut to the same claim, and the second from a measurement

`2.01x` claimed -> `1.25x` after reading the engine (E31 correction banner) -> **`1.12x` measured
at S15, `1.04x` unresolvable at T10.** The brief bound the disposition in advance:
`LOCALITY-PARTIAL` means the knob is still worth setting, **but E31 section 6.1's arithmetic may
not be used to price anything else.** In force from here.

**Why it over-credited**: the desk model charged the whole FFN read at the rate E31 measured for
the `down` run's granularity, when `gate` and `up` -- two of three matrices, two thirds of the
bytes -- were ALREADY in the flat part of the curve at `E=256` and had nothing to gain. Same
class of error as the `2.01x` it replaced: **a measured curve applied to the wrong denominator.**

### 43.3 Gates, and the two that could not fire as written

`G-E33C` as written demanded IDENTICAL charged weights across `E`; the addendum computed, before
exporting, that this is unsatisfiable because the router is charged (`synth_export.py:66`) and
shrinks with `E` (`-1.201% / -1.501%` at S15, `-1.094%` at T10; exact equalisation needs
`k = 4.1429`). Recorded FAILED as written, `G-E33C'` decides at 2.0% and fires. `G-E33A` was
anchored to an E26 ABSOLUTE that E26 disowned, so `G-E33A'` (same-session `S15-E256 / S15-DENSE`
against E26's `ratio_vs_dense 1.9731`) decides: reads `1.9247`, `-2.5%`. `G-E33B`: `S15-DENSE`
`27.74` vs E28's `29.30`, `-5.3%`. Six exports, six `GATE V3` passes.

### 43.4 Run 1 VOID, and the contention was the operator's own documentation work

Pre-run witness `14.8%`, post-rep-2 witness **`35.5%`** -- caused by me running the E32 doc patch
scripts during the timing. Not cosmetic: all three reps read `1.081` (`PARTIAL`), reps 1+3 read
`1.046` (`ABSENT`). **A band that depends on a contended rep is not a band.** Kept at
`results/e33_down_locality_void_run1.json`. Run 2 used five reps with the operator idle, and its
band survives every jackknife. **The law already said a contended timing is not a timing; what
E33 adds is that MY OWN documentation work is contention.**

Predictions **2 HIT / 2 MISS / 1 conditional**: the size (`1.18-1.32`) missed, `E64`'s position
missed (`50.4%` of the way, not `77%`), T10-smaller-than-S15 hit, and the no-effect-on-E30/E31
registration holds.


---

## 44. E34 -- `FLOOR-IS-THE-WALL`: with the FFN GONE, T10 reads 20.03 tok/s

**Probe** `probes/E34_THE_FLOOR_UNDER_THE_FLOOR.md` | **brief**
`briefs/BRIEF_E34_THE_FLOOR_UNDER_THE_FLOOR.md` (`684e677`) | **runner**
`engine/e34_floor_under_floor.py` | **result** `engine/results/e34_floor_under_floor.json`.
205 s, four arms, five interleaved reps, idle box.

### 44.1 The question nobody had asked

Every probe from E18 to E33 attacked the FFN, because at T10 the FFN is nine tenths of the
weight. **Nobody had measured what is left when it is gone.**

| arm | `k` | charged/token | mean tok/s | charged G-w/s |
|---|---|---|---|---|
| `T10-DENSE` (E26's own dense file) | -- | 10.6032 G | **4.408** | 46.74 |
| `T10-K16` | 16 | 2.7263 G | 15.872 | 43.27 |
| `T10-K4` | 4 | 2.3299 G | 19.026 | 44.33 |
| **`T10-FLOOR`** | **1** | **2.2308 G** | **20.030** | 44.68 |

### 44.2 The wall, in both conventions, kept apart

| charged | | moved | |
|---|---|---|---|
| numerator this session | **46.74 G-w/s** | ceiling (E30) | **36.30 GB/s** |
| budget for 50 tok/s | 0.9348 G | budget for 50 tok/s | 0.7260 GB |
| attention+head floor | **2.1475 G** | attention+head floor | **1.0786 GB** |
| **floor / budget** | **2.297x** | **perfect-kernel floor rate** | **33.65 tok/s** |
| measured floor rate | **20.03** (2.50x short) | **floor / budget** | **1.486x** |

**At T10's LITERAL shape, 50 tok/s is not reachable on this box by any amount of FFN work** --
the current engine is 2.50x short on the floor alone, and a PERFECT kernel is still 1.49x short.
The measured floor arm charges 3.882% more than the pure floor (the charged router plus one
surviving group), so the verdict is conservative.

**Residency cannot close it**: 16 MB of L3 is `0.016 GB` of the floor's `1.0786 GB` while the gap
is `0.353 GB` -- **22x the whole L3**. Registered in the brief before the run.

### 44.3 The gates, and why `G-E34C` is the load-bearing one

`G-E34A` uses E26's RATIO not its absolute (E33 addendum s2): `T10-K16/T10-DENSE` reads `3.601`
against E26's `3.735`, `-3.6%`. `G-E34B`: the runner recomputed both floors independently and the
brief's hand table **agreed to the byte** on charged (`2,147,483,648`) and to `+0.002%` on moved.
`G-E34C`: this session's numerator **46.74 G-w/s** predicts `T10-K4` at `20.06` against `19.03`
measured, `+5.4%` -- **and it lands within 1.3% of E28's independently measured packed numerator
(46.16)**, a different session and a different file, after E32 made packed operative. The model
behind every prediction here is not fitted to the points it predicts.

### 44.4 What the budget buys in DEPTH -- desk arithmetic on a measured ceiling

At T10's width, FFN at zero, perfect kernel: attention is `21,069,824 B/layer` moved, the head
`67,239,936 B`, the 50 tok/s budget `726,000,000 B` -> **31 of the 48 attention layers fit, 65% of
the depth**, before the FFN gets any budget at all. E27 measured that depth cuts cheaply IF the
right layers go (`L21-MINRES` 113/160 vs `L21-LAST` 57/160 at identical cost) and E29 confirmed
the ordering against a proper control; **E34 is the first time the engine side has said how much
depth the budget actually buys.** Predictions **5 HIT / 0 MISS** -- which is a sign this probe was
confirmatory, not a virtue: the two probes today where I was WRONG moved the programme further.

### 44.5 Run 1 VOID (the controls caught me) and a unit slip (the law caught me)

`ARMS` ran `T10-DENSE` with no `--carve-k` on the CARVED file, and an un-flagged carved file uses
the `k` stored in it -- 1 -- so the dense arm was **the floor arm wearing the dense arm's label
and its 10.6 G charge**. `G-E34A` read `0.783` against `3.735`; `G-E34C`'s numerator came out at
`217.62 G-w/s`, 4.7x the machine's entire throughput. **The void run's floor arm landed at 21.27,
close to my prediction, and is not quoted** -- a number that agrees with me is the one I should be
least willing to keep from a void run. And the runner printed the charged ratio as
`2297319028.69x`: `floor_charged` in WEIGHTS over a budget in G-WEIGHTS, **the unit hiding in the
denominator**, which is precisely what this programme's byte-convention law is about. Corrected
on recorded values; the JSON carries the erratum.


---

## 45. E35 -- the 50 tok/s envelope, bracketed by measurement on BOTH sides

**Probe** `probes/E35_THE_FIFTY_TOK_S_ENVELOPE.md` | **brief**
`briefs/BRIEF_E35_THE_FIFTY_TOK_S_ENVELOPE.md` (`17e7fc8`) | **runner** `engine/e35_envelope.py`
| **result** `engine/results/e35_envelope.json`. 75 s of timing, five arms, five interleaved
reps, idle box (witness 5.2% before, 2.5/0.5/0/0/0 between reps).

### 45.1 The first arms in this programme to CLEAR 50 tok/s at a 4096-wide shape

| arm | `L` | charged | measured | registered model | dev | room to 50 |
|---|---|---|---|---|---|---|
| `T10-L48` | 48 | 2.2308 G | **19.88** | 20.95 | **-5.1%** | `-1.2858 G` |
| `T10-L32` | 32 | 1.5320 G | **30.24** | 30.51 | -0.9% | `-0.5869 G` |
| `T10-L24` | 24 | 1.1825 G | **40.35** | 39.52 | +2.1% | `-0.2375 G` |
| **`T10-L16`** | 16 | 0.8331 G | **56.16** | 56.10 | +0.1% | **`+0.1120 G`** |
| **`T10-L12`** | 12 | 0.6584 G | **71.35** | 70.99 | +0.5% | **`+0.2867 G`** |

**THE ENVELOPE: at 50 tok/s this box affords `0.9451 G` ACTIVE CHARGED weights per token**,
interpolated between two arms that bracket the goal. Implied numerator **47.254 G-w/s** against
E34's independent dense **46.74** (+1.1%). **And it is gather-inclusive**: every arm is a
`quant==4` carved artifact reading one group per layer, so E31's penalty at `GSZ=56`
(`down` runs of 3,584 B) is already inside the number.

At `L=16` the room is **111,980,220 weights/token = 7.0 M per layer = ~10 carve groups = 560 of
14,336 neurons = 3.9% activation.**

### 45.2 Verdict `DEPTH-IS-HALVED`, and the UNIT decides one of my predictions

`L* = 19.33` under the model as registered (brief predicted 19). But the brief's model is
labelled "FFN at carve k=1" while its per-layer coefficient is the **pure floor**
(`41,943,040`); the arms measured carry the residue and read **`43,679,744`** per layer, 4.14%
more. Same crossing, two conventions: **19.33 as registered (prediction 3 MISS), 18.56
residue-inclusive (would have been a HIT).** Scored as registered -- **the reading that flatters
the prediction is not taken** -- and both are reported, because 19.33 is the depth of a TRUE
zero-FFN model and 18.56 the depth of an arm like the ones on the box. Same class of defect as
E33's `G-E33C`: a model labelled as one object whose arithmetic describes another.

Prediction 4 (linear in `1/charged` within 5%) **BREAKS at 5.1%** -- on one arm, `L=48`, and
slow. Four of five sit within **2.1%**. A charged-weight model cannot see a PER-LAYER FIXED
COST, which is what makes deep arms read slower than their weight count says; E26 measured the
carve machinery at `-6.36%` before a group falls, and E34's floor arm missed in the same
direction. **The model is excellent where the goal lives (`L <= 32`) and pessimistic at depth.**

### 45.3 What it licenses, and what it does not

`T10-L16` at 56 tok/s is **0.8 G of attention, not a 10 B** -- registered as prediction 5 before
the run. What E35 licenses is a SPEC: **~16 layers at 4096 wide with ~560 FFN neurons activated
per layer per token, in carve groups.** Sizing a 10 B total at that shape needs **~46,000 FFN
neurons per layer**, i.e. **~1.2% activation** -- desk arithmetic on measured inputs, and the
target of the next probe rather than a claim of this one. Gates: `G-E35A` **-0.8%** against
E34's own artifact (the tightest session-to-session agreement in the branch), `G-E35B` exact on
all five arms at zero tolerance, `G-E35C` five of five `GATE V3`.


## 46. E36 -- ten billion parameters at fifty tokens a second

**Probe**: `probes/E36_TEN_BILLION_AT_FIFTY.md`. **Brief** `1738aa2`, pushed before the runner
existed. **Runner** `engine/e36_ten_billion.py`; **results** `engine/results/e36_ten_billion.json`
(run 1, registered) and `..._order_reversed.json` (run 2, the order control).

**SPEED ONLY.** Synthetic weights. This section says nothing about quality and nothing about
whether such a model can be trained.

### 46.1 The artifact, and the ratio that is the point

`A10B`: `D=4096, F=46080, L=16, NH=32, NKV=8, HD=128, V=32768`, `--carve 256` (group = 180
neurons). **9,999,220,736 parameters, 5,485,658,936 bytes.** Charged at `k=3`:
**928,251,904** a token. **The file weighs 10.77x what a token costs** -- the first artifact in
this branch whose SIZE and whose COST are different numbers. Every earlier 10 B-shaped file
(`T10`, 10.74 B) was dense-active at 10.6 G a token.

### 46.2 Verdict `TEN-B-NEAR-FIFTY`: 49.96 against a 50.0 bar

| arm | neurons | charged | run 1 (registered) | run 2 (order control) |
|---|---|---|---|---|
| `T10-L16` control | -- | 0.8331 G | 58.08 (+3.4% vs E35) | 54.85 (-2.3%) |
| `A10B-K1` | 180 (0.39%) | 0.8575 G | **54.86** | **56.67** |
| `A10B-K2` | 360 (0.78%) | 0.8929 G | **51.50** | **52.65** |
| **`A10B-K3`** | **540 (1.17%)** | **0.9283 G** | **49.96** | **51.50** |
| `A10B-K4` | 720 (1.56%) | 0.9636 G | **47.74** | **46.21** |
| `A10B-K6` | 1080 (2.34%) | 1.0344 G | **43.78** | **44.93** |

**The crossing is the robust number, not the verdict cell.** Measured twice under OPPOSITE arm
orders: **`k* = 2.867` and `k* = 2.861`** -- **516 and 515 of 46,080 neurons, 1.12% activation,
agreeing to one neuron in forty-six thousand.** RANK partner (E14 sez.3): both runs strictly
monotone in `k`, 5 of 5, under opposite orders.

### 46.3 Run 2 printed 51.50 on the verdict cell and did NOT promote it

Run 1's arms ran in a FIXED order inside each rep and the box drifted within reps (rep 1 opened
with the fastest control of the run, 62.33, and closed with the slowest `k=6`, 40.60) -- a fixed
order under drift charges the drift to the HIGH-`k` arms, which is the slope the verdict stands
on. **The rule was pushed at `9819226` before run 2 existed**: run 2's only question is whether
the k-slope depends on arm order; agreement leaves the verdict unchanged NO MATTER WHAT NUMBER
IT PRINTS. Slopes **0.8995 vs 0.9553 ms/group, +6.2%**, far inside the reps' own dispersion
(worst per-arm spread 12.4% / 19.3%) -> **agree -> verdict unchanged at 49.96.** And the control
answered against my suspicion: reversing the order made the slope STEEPER, not shallower. The
fixed order was not manufacturing the slope.

### 46.4 The finding: the envelope is TWO numbers, and an FFN weight costs 1.25x an attention weight

Prediction 3 (the measurement beats the model, because `GSZ=180` puts `down` runs at 11,520 B on
the flat part of E31's curve) is a **MISS in both runs** (-2.4%, -0.8%). Splitting the cost
instead of averaging it says why. Fit `time(k) = a + b*k`:

| | run 1 | run 2 | E35's envelope rate |
|---|---|---|---|
| **base** (attention + head + router, 0.8221 G) | **47.214 G-w/s** | **48.548 G-w/s** | **47.254** |
| **marginal carve group** (180 neurons, 0.0354 G) | **39.345 G-w/s** | **37.046 G-w/s** | |
| **gather penalty at `GSZ=180`** | **0.833** | **0.763** | |
| R2 | 0.995 | 0.920 | |

**The base reads at the FULL envelope rate. It is the marginal group that is expensive, at
~0.80 of it.** Not a per-layer fixed cost (the brief's own stated alternative), not a failure of
E35's envelope -- the gather, still, at four times the granularity of any previous arm, and
**E31's ~0.85 at 11,520 B was OPTIMISTIC.**

```
time per token = 17.2 ms (attention + head + router) + 0.93 ms per carve group of 180 neurons
```

**Every budget from here must price an FFN weight at ~0.80 of the attention rate.** Third cut to
the same claim (E26 `2.01x` -> E31 `1.25x` -> E33 `1.12x`), and the first measured PER GROUP
rather than inferred from a curve fitted at another granularity.

### 46.5 What it says about 100 tok/s

The base term alone is **17.2 ms**, so `L=16` at 4096 wide caps at **~58 tok/s with a
zero-cost FFN**. 100 tok/s wants the whole token in 10 ms, which at the measured base rate buys
**~8 layers with no FFN at all**. **100 tok/s at 4096 wide is not an FFN problem, it is an
attention problem** -- E34 said it, this prices it.

Predictions **3 HIT / 1 MISS / 2 held**. Gates: `G-E36A` +3.4%; `G-E36B` 9,999,220,736 from the
exporter == the same recomputed from the FILE'S OWN HEADER, `>= 9.9 G`, with bytes on disk ==
E1's independently written v4 layout exactly; `G-E36C` exact at every `k` at zero tolerance.
