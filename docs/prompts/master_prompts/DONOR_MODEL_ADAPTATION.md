# Master Prompt — Donor-Model Adaptation

**Target audience: an AI agent (hereafter "the Adapter") tasked with an open research problem.**
**Status: mandate, not a plan. Revised 2026-08-20 by the Architect with owner-sealed acceptance gates in §3.**
**This document defines the problem space exhaustively and closes almost none of it. That is deliberate.**

---

## 0. How to read this document

This is a **prompt**, not a specification. Its job is to leave you with **no ambiguity about what the
problem is**, and with **maximum freedom about how to solve it**. Where a decision has already been
made by the owner or by a prior measurement, it is marked **SEALED** and you may not reopen it without
saying so out loud and getting a human answer. Where a decision is open, it is marked **OPEN** and
carries a **closure criterion** — the thing that must be measured before anyone is allowed to have an
opinion. Everything else is yours.

Three reading rules:

1. **Numbers in this document are measured unless labelled otherwise.** Every measured figure carries
   its protocol and its scale. Where a number is a projection, an inference or a literature figure, it
   says so in the same sentence. Preserve that discipline in everything you write. The single fastest
   way to fail here is to quote one of these numbers without its scale caveat.
2. **A closure criterion is a contract.** When you close an OPEN item you must state which criterion
   you applied, what you measured, and — critically — that you fixed the criterion *before* you saw the
   number. This project's central law is anti-Goodhart: **gates are never loosened after the fact.**
   Costs may be re-priced; gates may not.
3. **You do not have permission to spend money or GPU-hours.** You propose; the owner launches. See §9.
4. **A system result is not a physics upper bound.** DRAM peak bandwidth, kernel-only rates and
   arithmetic footprint ceilings are useful lower bounds or feasibility screens, never a claimed engine
   tok/s or deployable fit. System claims use the end-to-end engine rate and the complete resident
   inventory defined in §3/S2.

---

## 1. The mandate

> **Find the best way to make a large existing open-weights model (1B–100B class) run on this project's
> C inference engine — that is, on a mid-range consumer CPU with adequate RAM — by converting it
> mathematically rather than by retraining it.**

The engine is not a generic runtime. It is a **co-designed** engine built on one thesis: *split the
thinking from the knowing*. A small, ternary, cache-resident compute core handles per-token compute; a
large, sparse, DRAM-resident tier holds capacity and is barely touched per token. Every byte path is
sequential and cache-resident or bulk-contiguous; never a per-token random gather.

Adapting a donor to this engine is therefore not a porting exercise. It is the question: **which
existing models can be transformed into this shape, by what transformations, at what cost in quality
— and is there a class of donor for which the answer is good?**

The honest possibility that this is a negative result is not a failure mode. It is one of the two
outcomes, and a well-evidenced "no, and here is precisely why" is worth more than a badly-evidenced
"yes". See §10.

### 1.1 The required decision, not merely an interesting result

The Adapter must end with one of these adjudicable statements for a named donor class and SKU:

1. **GO:** the converted system meets every sealed acceptance gate in §3 and its result is reproducible
   on the declared hardware/protocol.
2. **NO-GO:** a sealed gate is missed, with the mechanism localized and the closest measured result
   reported.
3. **AMENDMENT REQUEST:** the evidence says a sealed constraint, rather than the conversion recipe, is
   what prevents success. State the exact amendment required; do not silently proceed under it.

“Interesting”, “plausible”, or high local layer fidelity alone are not outcomes. They are intermediate
evidence only.

---

## 2. Where you are starting from — the measured ground

Everything in this section is **measured on real hardware** (reference: Ryzen 5 3600X, Zen 2, AVX2/FMA,
**no AVX-512, no VNNI**, DDR4, L3 16 MB per CCX × 2 CCX) unless stated. Per the project's Portability
law, the reference machine is a **floor exemplar**, not the target: every hardware number is a runtime
parameter, and the target *class* is x86-64-v3 old-generation and up.

### 2.1 What exists as committed code

| artefact | what it is | where |
|---|---|---|
| **`engine`** | single-binary C inference engine, AVX2/FMA, ternary pshufb-LUT kernels, selective-SSM scan, one SWA layer, dense or top-8 MoE MLP, parity-gated bit-exact against an fp32 reference | `benchmarks/phase60/engine.c` |
| stage engines e1–e4 | archival parity oracles for each optimization rung | `archive/benchmarks/phase60_stage_engines/` |
| recall tier | IVF-PQ two-stage retrieval slot, C-side smoke measured | `benchmarks/phase56/`, `bin/recall_probe.exe` |
| block-verify chassis | n-gram-drafted speculative decode, lossless, **built but off by default** | `--block K` in the engine |
| threads | OpenMP over independent outputs only; **bit-identical by construction**, invariant to thread count | `--threads N` |
| weight format | versioned binary, magic `E1M1` (dense) / `E4M1` (MoE); ternary parts stored both dequant-fp32 and packed int8+scale | `benchmarks/phase60/e4_export.py` |

### 2.2 The measured constants you must design against

**Bandwidth and cache (this is the physics of the problem):**

- **The L3 cliff is real and sharp: a step at exactly 16 MB** (L3-per-CCX) — compute-bound at
  ~100 GB/s below it, falling toward a ~28 GB/s DRAM floor above it. Single-thread.
- **Threaded, the wall becomes a slope, not a cliff.** Row-partitioned proj-GEMV at 6 threads sees the
  *aggregate* L3 (2 × 16 MB): measured r(size) = 187 / 185 / 134 / 60.5 / 55.7 / 45.5 / 45.3 / 36.5 GB/s
  at 4 / 8 / 16 / 24 / 32 / 48 / 64 / 96 MB. Single-thread the same curve is 30.5 → 23.3 GB/s, breaking
  at 16 MB. **Use the curve, not a single number.**
- **DRAM cold-stream ceiling: 21–26 GB/s single-thread, aggregate 40–44 GB/s, saturated at 3 threads**
  (6 threads add nothing; unpinned 6 threads regress).
- **The ρ-law: random-DRAM access vs sequential-cache-resident ≈ 14× worst case** on Zen 2 (~2× in-cache,
  ~4.3× in DRAM). This law has already killed four candidate designs in this project (SIMVQ,
  vector-codebook weight quantization, ANN graphs, IMI). **It will kill your design too if you put a
  per-token random gather anywhere on the byte path.** Bulk-contiguous chunk loads at ~48 KB granularity
  measure only ~2.5× penalty, not 14× — granularity is the escape, not locality.
- **Expert-pool streaming rate: kernel-pure 7.45 GB/s (1 thread) → 17.0 GB/s (6 threads) = 2.88 µs per
  48 KB expert.** The engine-integrated figure is 4.2 GB/s, and the ~3.9× gap decomposes as **~8.4 µs
  per expert of dispatch overhead around the kernel** (index gather, dequant, dReLU, combine). That is an
  engineering lever, **not** a bandwidth wall — worth knowing before you design around a number that is
  an artefact of unoptimized glue.

**Per-position compute floor (the second wall, and the one people forget):**

At the reference 8.3M config the engine's per-token compute floor is ≈ **1.14 ms**, decomposed
(µs/token, 1 thread → 6 threads): proj-GEMV 662 → 272 (×2.43); LUT-MLP dense 313 → 207 (×1.51), MoE
573 → 543 (**×1.06, thread-flat** — the overhead above); scan-recurrence 169 → 46 (×3.66); SWA 92 → 65;
head 40 → 8; norms and glue ~7. **Ternary shrinks bytes, not this floor.** A design that fixes bandwidth
and ignores the per-position floor has fixed nothing — this is exactly how the block-verify speedup was
scoped out at 8.3M.

**Quality costs of the engine's own properties (all at ~8.3M params, TinyStories or code, single-seed
unless stated — these are small-scale numbers and the literature trend says the ternary gap shrinks
with scale, which is a trend, not a measurement we own):**

- ternary (1.58-bit) QAT MLP: **+0.028 BPB** (an upper bound — the run was still improving)
- gated dReLU vs SiLU: **+0.0006 BPB** at 92% hidden / 79% gate sparsity
- both together, seed-averaged matched calibration: **+0.013 ± 0.005 BPB**
- ternarizing the SSM *projections*: **+0.018 to +0.022 BPB → rejected, they stay fp32** (Phase 61).
  Note carefully what that measured: **weight quantization**, not compute dtype. Confusing the two has
  already cost this project a wrong diagnosis once.
- **σ_seed = 0.005 BPB.** Any single-seed delta smaller than this is not resolvable. This constant is
  how every result in this project is judged; adopt it.

**Speed achieved:** 176 → 848 tok/s dense single-thread across the Phase 60 optimization ladder (4.8×);
1652 tok/s dense at 6 threads, bit-identical. At 8.3M params. **No quality claim exists above ~8.3M
in this project.** Do not import one.

### 2.3 The architectural findings that constrain the shape of a solution

- **Granular MoE beats dense at matched total params.** E32×h128 top-8 measured BPB 0.8589 vs dense-1024
  0.8799 and dense-4096 0.8674. Granular beat coarse (E8×h512 top-2 = 0.8637). Router healthy under
  Switch aux loss (0 dead experts, max/mean ≤ 1.47×).
- **Expert routing locality is FALSIFIED, three independent times.** The expert working set over a block
  matches the i.i.d. baseline `E·(1−(1−k/E)^K)`; expert persistence equals the base rate k/E. **There is
  no hot expert pool.** Consequence: experts are DRAM-streamed and granularity-bounded, not cache-warm.
  If your donor's routing shows real locality at scale, **that is a finding, not a convenience** —
  report it loudly, it reopens a closed question.
- **Active-set predictability is in-place, not ahead-of-time:** a cheap linear probe forecasts 86–92% of
  the active units in the *same* layer; predicting the *next* token's active set drops to 50–63%. The
  durable lever is SKIP (don't stream the inactive), not prefetch.
- **The recall tier works and is cheap:** IVF-Hadamard partition + 4-bit ADC shortlist + exact top-16
  rerank measured **29.05 µs/token at 128K entries at 6 threads** (52.4 µs at 1 thread, scalar ADC),
  footprint 1.69 MB searchable + 64 MB values. The InfoNCE *representation* is load-bearing; the
  *partition* is data-independent Hadamard and therefore drift-proof by construction.
- **Multi-token generation from SSM state is CLOSED at sandbox scale** (the hidden state carries ~1
  reliable lookahead step even when co-trained to carry more). The model is AR-lossless.

### 2.4 The engine's current hard limitation — read this before anything else

**The engine has no runtime dimensions. It has compile-time ones.**

`benchmarks/phase60/engine.c:51-75` declares `V 1024`, `D 256`, `N 96`, `H 8`, `L 6`, `DN 512`,
`DTR 16`, `CONV 4`, `WIN 128`, `SWA_LAYER 5`, `MLP_HID 1024`, `E 32`, `HID_E 128`, `KTOP 8` as
preprocessor constants. The weight file's 16-word header *carries* V, D, N, H, L, Dn, dt_rank, conv,
win, swa_layer, E, hid_e, k — but `load_weights` validates only `E`/`hid_e`/`k` (MoE) or `mlp_hid`
(dense) against its own defines and otherwise **reads according to itself**. A file with different
dimensions is read as garbage or fails on a short read.

There is exactly one architectural layer of the SWA/SSM split too: `is_swa[l] = (l == SWA_LAYER)` —
a single hardcoded attention layer at index 5.

**Therefore: the first obligated work item, prior to any donor question, is to make the engine
dimension-agnostic at runtime.** It has a free and unambiguous gate: **bit-exact parity with the
current binary on the current model.** If the generalized engine does not emit a byte-identical logit
stream on `results/phase60/*.bin`, it is wrong, and no donor result computed on it means anything.
The project's law here is explicit and was learned the hard way: **kernel-bit-exactness does not
compose to system-correctness — always verify end-to-end parity.**

Note also that this generalization is a *precondition*, not a *research result*. Budget it as
engineering, gate it, and move on. Do not let it become the project.

---

## 3. Owner-sealed constraints — SEALED, not negotiable without asking

These four were decided by the owner on 2026-08-20 in response to direct questions. They bound the
search space. If your research finds one of them makes the problem unsolvable, **say so explicitly and
stop for a human decision** — do not quietly relax it.

### S1 — Compute budget: free Kaggle quota only, and it must be SHORT

Up to ~90 GPU-hours/week exist across three accounts (T4 / P100 class, ~9-hour sessions, relay-based
handoff). **But the owner's instruction is stronger than the quota:** *"it must be a short thing, not
7 weeks of compute. There is no need to distill — the point is to find a mathematical way to convert."*

Read this as the design constraint it is:

- **The primary path is a TRANSFORMATION, not an optimization.** Closed-form or near-closed-form
  operations on the donor's weights: orthogonal/rotation transforms, factorizations, clustering,
  re-parameterizations, activation-statistics-driven rescalings, layer-wise analytic solves.
- **Calibration is allowed and expected.** Forward-only passes over a modest calibration corpus to
  collect activation statistics are *not* training and are cheap enough to run on CPU. Quantify what
  you use; hundreds of MB is normal in the PTQ literature, and the *composition* of that corpus is
  itself a variable worth measuring.
- **Layer-wise / block-wise local optimization is a grey zone — define your position explicitly.**
  Solving one layer's output-matching problem in isolation (seconds to minutes per layer, no end-to-end
  backprop) is closer to a transformation than to training, and much of the PTQ literature lives here.
  It is *permitted*, but you must **budget it, declare it, and report it separately** from anything
  closed-form, because it is the boundary that will erode if nobody watches it.
- **End-to-end gradient descent on billions of tokens is OUT as the primary path.** It may be proposed
  as an explicitly-costed *fallback* (§8.M) with a measured quality delta showing what the fallback buys
  over the transformation-only result. It is never the plan.

**Operational definition, SEALED.** “Transformation-only” permits deterministic algebraic transforms,
forward calibration, and a separately declared local solve whose inputs are fixed donor activations and
whose outputs alter one layer/block at a time. It forbids end-to-end gradients through two or more
blocks, optimizer state carried between blocks, and any parameter update chosen from end-to-end task
loss. Before proposing a local solve, state: calibration-token count and hash, fitted parameters,
objective, iterations/wall time per layer, CPU/GPU use, and total wall-clock budget. If any of those
are unavailable in advance, the method is a healing proposal under §8.M, not transformation-only.

### S2 — Two parametric SKUs

| SKU | RAM budget | machine class | *weight-only arithmetic ceiling* at 0.5 B/weight | *weight-only arithmetic ceiling* at 0.2 B/weight (trit-pack, **not built**) |
|---|---|---|---|---|
| **A — mainstream** | ≤ 16 GB | 32 GB desktop | ≲ 32B | ≲ 80B |
| **B — enthusiast** | ≤ 64 GB | 64–128 GB workstation | ≲ 128B | ≲ 320B |

The table is deliberately an **unusable upper envelope**: 32B × 0.5 B already consumes all 16 GB, so it
does *not* mean a 32B donor deploys in SKU-A. A donor fits only when the complete inventory is below the
RAM budget with a declared OS/allocator margin:

```
resident RAM = packed weights + scales + padding + architecture metadata + non-ternary organs
             + embeddings/head + KV cache at the SKU context + activations/scratch/logits
             + mmap/page-cache or heap policy + OS/allocator margin
```

**Any part of the donor you fail to ternarize is charged at fp32 (4 B/param) and eats the budget four
to twenty times faster.** Inventory every O(size) allocation on the path, including the ones that fit.

Treat RAM as a **runtime parameter**, not a compile-time assumption: the same binary must serve both
SKUs by reading the machine, per §8.K.

### S3 — Hybrid operators: attention permitted on a minority of layers

The engine may grow GQA + RoPE (or equivalent) and keep it on a **minority** of layers, with SSM on the
rest. This is not alien to the engine — it already runs one sliding-window attention layer.

**"Minority" is the sealed word and it is doing real work.** It means the design must not silently
become a transformer runtime. But the *fraction* is OPEN and is one of the most important things you
will measure (§8.B). If your measurements say the workable fraction is 60% rather than 40%, **that is
a finding that must go back to the owner as a request to amend S3** — not a number you round down and
proceed on.

**Context contract, SEALED.** SKU-B must provide **128K native context** within its 64 GB whole-process
budget. SKU-A must provide a **128K user-visible context contract** within 16 GB: native retained
attention may use only a measured 8K–32K window, and the engine recall tier must supply the long-range
path for the remainder. The Adapter must not call SKU-A “128K” merely because the prompt is accepted;
it must pass the long-context retention gate in S4/§8.I at 128K.

### S4 — Success is judged general-purpose

The donor brings general knowledge and the entire point is how much survives conversion. So:

- **Primary metric: retention versus the unconverted donor**, measured on standard general benchmarks,
  same harness, same prompts, same decode settings. Report absolute *and* retention-%.
- **Acceptance gates, SEALED:** at least **90% global retention** and at least **80% retention on each
  critical task**: Code, Long-Context Recall, and Logic. No critical task below 80% is admissible.
  The harness must define retention appropriately per metric before seeing any converted score:
  chance-normalized retention for bounded accuracy metrics, and a predeclared monotone normalization for
  likelihood/perplexity metrics. A low-performing donor cannot pass only because it retains a high
  fraction of a low baseline; absolute donor and converted scores are always reported.
- **Performance gate, SEALED:** the complete engine must sustain **≥10 generated tok/s in decode** for
  each SKU, measured one stream, declared thread/pinning policy, warmed weights, and the SKU context
  contract. Report prompt-prefill/TTFT separately; decode tok/s does not stand in for them.
- The project's pinned P62 code-val BPB is **still measured and reported** at every point (it is the one
  metric comparable to everything else this project has ever done, it is decontaminated, and it is free)
  — but under S4 it **does not decide**.
- Building the general-purpose eval harness is therefore **in scope and is a deliverable**, not an
  afterthought. It does not exist in this repo. See §8.N.

---

## 4. The central thesis, and the central tension

**Thesis to test:** there exists a class of open-weights donor models — identifiable by stated,
checkable criteria — that can be transformed into the engine's shape by data-free or
calibration-only mathematics, retaining enough capability to be worth running, on a consumer CPU.

**The tension, stated plainly because everything else depends on it:**

S1 says *do not train*. S3 says *attention on a minority of layers*, which implies the majority of the
donor's attention layers become SSM layers. **Converting attention to a state-space recurrence without
training is the hardest open problem in this entire document.** The published attempts that work
(transformer→Mamba distillation of the Mamba-in-the-Llama / MOHAWK family) do the *initialization*
mathematically — reusing the donor's QKV projections as SSM input projections — and then spend
**billions of tokens** healing it. The mathematical half is cheap; the healing is exactly what S1
forbids.

So one of these must give, and **your first substantial job is to find out which**:

- **(i) The healing is not actually needed** if the matching is done well enough layer-wise. Possible
  — nobody has seriously tried the closed-form/local-solve end of this — and it would be the strongest
  result available here. Treat with suspicion proportional to how much you want it to be true.
- **(ii) The minority fraction is wrong.** Keep attention on most layers and attack the other axes
  (ternarization, sparsification, MoE-ification, cache partitioning) where the mathematics is far more
  developed. This yields a working system sooner but concedes part of the thesis, and requires an S3
  amendment from the owner.
- **(iii) The SSM backbone is not what makes this engine fast**, and the speed comes from ternary
  weights + activation sparsity + cache residency + bulk-contiguous streaming — properties that are
  **operator-agnostic**. If that is true, converting attention→SSM is optional, and the interesting
  claim changes shape.

**(iii) deserves particular attention because it is cheap to test and would restructure the whole
project.** From §2.2's decomposition: at the sandbox config the scan recurrence is ~13-15% of the
compute floor while proj-GEMV is ~52%. The SSM's structural advantage is **O(1) state instead of a
growing KV cache** — which is a *memory* and *long-context* argument, not a per-token *speed* argument.
Quantify that trade explicitly at each SKU's context target before assuming either way.

**Closure criterion for §4 is deliberately two-stage.** **Screening (stage 2a):** a measured,
per-layer comparison of (a) donor attention layer kept as-is, (b) donor attention layer converted to
SSM by closed-form initialization with no healing, and (c) the same with layer-local output matching
against a fixed activation-statistics budget. Score only donor-layer output fidelity, with the threshold
pre-registered before running (b) or (c). This is a cheap filter, not a success claim. **Decision
(stage 5):** only after the generalized engine and evaluation harness exist, score surviving variants on
the sealed end-to-end retention and performance gates in S4. A stage-2a result can kill a route; it
cannot certify it.

---

## 5. The arithmetic that says this is not obviously hopeless

Do not take these as results. They are **desk-model projections** built from §2.2's measured rates, and
their purpose is to tell you where the walls actually are so you do not spend three weeks optimizing
the wrong thing. Every one of them must be replaced by a measurement.

**Physical streaming lower bound for a sparse donor.** For a MoE donor with `P_active` active parameters
per token, all ternary at the engine's real 0.5 B/weight packing:

```
streamed_bytes_per_token ≈ P_active × 0.5 B
physical_time_lower_bound ≈ streamed_bytes_per_token / 42 GB/s
```

| donor shape | active/token | streamed B/token | physical lower-bound µs/token @42 GB/s | physical upper-bound tok/s |
|---|---|---|---|---|
| ~30B total, ~3B active | 3B | 1.5 GB | ~36,000 | **~28** |
| ~120B total, ~5B active | 5B | 2.5 GB | ~60,000 | **~17** |
| ~8B dense | 8B | 4.0 GB | ~95,000 | **~10.5** |
| ~70B dense | 70B | 35 GB | ~833,000 | **~1.2** |

**This table does not predict engine tok/s.** 42 GB/s is the aggregate DRAM ceiling, while the existing
integrated MoE path measures 4.2 GB/s and includes dispatch overhead; neither scales automatically to a
donor layout. It omits scales/padding, per-selected-expert dispatch, the per-position compute floor,
head, attention/KV traffic, and all glue. It may rank candidates, but it cannot establish the 10 tok/s
gate. The stage-1 arithmetic must replace it with an implementation-aware budget:

```
t_token_estimate = Σ(bytes_organ / measured_engine_rate_organ)
                 + Σ(selected_expert_calls × measured_dispatch_overhead)
                 + measured_compute_and_glue + head + attention/KV traffic
```

Measure the relevant kernel and integrated-engine rate at donor-like dimensions before promoting a
candidate. Activation sparsity/MoE remains a plausible eligibility feature, not a pass by itself.

**The footprint wall, which is where the real trouble is.** 120B × 0.5 B/weight = **60 GB** of packed
weight codes alone — already leaving only 4 GB in SKU-B for every other allocation, so it is not yet a
SKU-B fit. This assumes *everything* ternarizes.

**The head is a first-class problem and it is easy to miss.** A general-purpose donor carries a large
vocabulary — commonly 128K–256K. At D = 2048 and V = 152K the embedding/head matrix alone is ~311M
parameters: **1.24 GB in fp32, read every token for the output projection** — a ~30 ms *physical lower
bound* at 42 GB/s before any other work happens. This project has already
banked the same finding at its own scale (a 256K teacher tokenizer was rejected as a "keystone-killer"),
and it explicitly parked the compressed-head probe **only because its own vocabulary is ≤ 4K**. A donor
brings that problem straight back. **Assume the head must be quantized, factorized, clustered or
otherwise compressed, and treat "how, at what quality cost" as a named open question (§8.H).**

---

## 6. The non-negotiable working method

These are project laws. They are not style preferences; every one of them exists because its violation
already cost this project real time.

1. **Pre-register every gate before its number exists.** Write down the criterion, push it, *then* run.
   A gate interpreted after the result is not a gate. Costs may be re-priced; gates may not be loosened.
2. **One variable per stage.** If you change two things and the result moves, you have learned nothing
   and spent the budget learning it.
3. **Planted controls: an instrument must be shown to FIRE on a known-positive before its nulls mean
   anything.** This project's characteristic failure is not the wrong number — it is the *plausible
   artefact*: a measurement tool that silently returns zeros, a checkpoint that loads with mismatched
   architecture and reports nothing, a guard that never fires in either direction. Before you trust any
   detector, verifier or comparator, corrupt something deliberately and prove it screams.
   **The minimal-significant-corruption rule:** plant the *smallest* thing that could be wrong (one ULP,
   one element), not a catastrophe chosen to make the detector look good.
4. **No guard ships without being exercised in both directions, and the exercise goes in the log.**
   Must-pass case and must-refuse case, both recorded.
5. **The artefact is the authority, never a table in your code.** If a checkpoint carries its own
   architecture config, read it from the checkpoint and use any hardcoded expectation only as an
   assertion that must match. This exact defect — a tool's own hardcoded table silently overriding what
   the file said — was found in this repo and it is the general shape of the failure.
6. **`strict=True` on a state-dict does not verify architecture.** Two different modules can share
   parameter names and shapes (a ternary linear and a dense linear both expose `weight`). Verify the
   *config*, and cross-check with a numerical probe that is architecture-sensitive.
7. **NO `-ffast-math`.** Ever. Determinism is a gate in this project.
8. **Parity is end-to-end or it is nothing.** Kernel-level bit-exactness does not compose to
   system-level correctness. Measured, in this repo, more than once.
9. **A tag is not a reproducible specification; a content hash is.** Pin data and artefacts by hash.
10. **Report honestly and deflate your own claims first.** When a result can be read two ways, write
    down both and say which the evidence supports. When you are corrected, correct plainly and continue.
11. **Every hardware figure is a runtime parameter.** Design for the *class* (x86-64-v3, old
    generation), not for one exemplar. A portable design that loses some speed beats a fast one that
    only runs on one machine.
12. **Name the correctness relation before reporting it.** There are four distinct contracts:
    (a) kernel correctness = bit-exact against a scalar integer reference; (b) legacy-engine
    correctness = byte-identical logits on the current model after refactoring; (c) converted-engine
    correctness = C agrees with a declared high-precision reference of the *same transformed graph*,
    at a predeclared numerical tolerance; (d) donor fidelity = the transformed model retains donor
    behaviour/metrics. Never call (c) or (d) “parity with the donor”.

---

## 7. Roles — who does what

You are working inside an existing human-plus-agents structure. Respect its boundaries.

| role | authority |
|---|---|
| **the Owner** ("il Capo") | launches all GPU and long-running jobs; makes every spend decision; owns strategy. **Nothing costing money or a GPU session happens without an explicit go.** |
| **the Architect** | adjudicates results, re-verifies arithmetic independently, writes the briefs and the record, decides direction. Does not write production code and does not commit. |
| **you, the Adapter / Principal** | You are the sole AI coordinator. You must spawn and direct the specialist agents below, integrate their written evidence, decide task decomposition and stop work that lacks its required controls. You write briefs, pre-registrations, decision memos and reports; **you do not write production or experimental code yourself.** Within sealed constraints you may choose the next investigation; the Architect/Owner alone adjudicate sealed-gate GO/NO-GO or amendments. |
| **the Builder** | The sole author of production, conversion, benchmark and test code. Implements one approved brief at a time, smokes it, and STOPs with a measurement table and reproducibility manifest. Does not choose research direction, alter gates, commit, push, or launch GPU work. |
| **the Media Manager** | Owns repository hygiene, commits, branches and pushes. Creates and maintains the isolated research branch `research/donor-adaptation`, records its base SHA, and never touches the concurrently running training branch. Is the only agent allowed to commit or push (without adding AI signatures, attributions, or trailers (e.g., do NOT include 'Co-authored-by:')). Does not alter scientific conclusions or source code logic except repository metadata explicitly assigned to it. |
| **the Researcher** | Performs deep prior-art research from primary papers, authoritative technical documentation and relevant textbook material. Produces citable, implementation-relevant research notes: mechanism, assumptions, cost, counter-evidence, exact source/version and applicability to S1–S4. Does not write engine code, choose conclusions, or promote a literature claim to a project fact. |
| **the Controller** | Independent adversarial reviewer. Its job is to break the proposed method, apparatus and interpretation—not to help them look plausible. It audits gates before a run, code and planted controls before evidence is trusted, then logs BLOCK/PASS/FLAG findings with a reproduction path. It does not implement the fix, commit, push, or adjudicate direction. |

### 7.1 Mandatory multi-agent operating protocol

**This is mandatory, not a suggestion.** The Principal must operate through spawned agents. At the start
of the goal it spawns, at minimum, the Media Manager, Researcher and Controller; it then spawns the
Builder as soon as there is an approved implementation brief. If parallel-slot limits prevent all roles
from being active simultaneously, sequence them, but never omit a role or replace independent review
with the Principal's own confidence.

All specialists report to the Principal in a compact written form. Every assignment names: the question,
owned files or read-only status, expected artefact, gate affected, explicit non-goals, and stopping
condition. The Principal maintains a decision log with `claim → evidence → independent check → open
risk → next authorized action`. No result may be summarized as established merely because an agent said
so.

**Two-key evidence rule.** Any artefact that opens, closes, or materially informs a sealed gate needs:

1. an author report from the Builder or Researcher, including command, revision/hash, inputs and raw
   outputs; and
2. an independent Controller report that tries to falsify the claim, exercises the smallest meaningful
   planted corruption, and verifies the relevant guard.

For a result used to request a sealed-constraint amendment or to make a GO/NO-GO recommendation, the
Principal also spawns a **second independent Controller** for replication or an adversarial arithmetic/
protocol audit. Agreement is not enough; unresolved disagreement is reported to the Architect.

### 7.2 Work ownership, repository and publication protocol

- The Media Manager creates `codex/research/donor-adaptation` from a recorded base SHA before research
  changes begin. It keeps training work isolated, records pre-existing dirty files, never resets,
  deletes, rebases or force-pushes unrelated work, and makes no commit on the training branch.
- The Builder owns all code and tests for an approved brief. The Controller is read-only with respect to
  that code and writes only its review artefacts; the Researcher writes only research artefacts; the
  Media Manager owns Git and repository housekeeping. File ownership is stated before editing so shared
  agents do not overwrite one another.
- **Push-before-run applies to every gate-bearing apparatus:** the Builder's code, pre-registration,
  planted control and exact invocation are committed and pushed by the Media Manager in falsifiable form
  before the run they govern. Corpora, checkpoints and bulky generated data stay uncommitted; manifests,
  scripts, content hashes, raw-log locations and environment versions are committed.
- The Controller reviews every non-trivial Builder diff twice: once **before** a gate-bearing run (wrong
  config, dead guard, invalid control, hidden default, measurement leak), and once **after** it (whether
  the recorded evidence actually supports the claim). A BLOCK stops the pipeline until the Builder fixes
  it and the Controller rechecks; a FLAG may proceed only if the Principal records the residual risk.
- The Principal never asks a specialist to “make it pass”. It asks whether the hypothesis survives. A
  negative result is committed, reported and preserved with the same care as a positive one.

### 7.3 Required reporting cadence

At every stage boundary, the Principal reports only: completed artefacts and commit hashes; measurements
with protocol/scale; Controller verdicts; deviations from pre-registration; unresolved risks; and the
single next decision requiring the Architect or Owner. It must explicitly say “no decision” when evidence
is insufficient. No GPU session, money spend, sealed-constraint amendment, branch merge, or public push
is authorized by a specialist or by the Principal alone.

---

## 8. The problem space, axis by axis

This is the heart of the document. **Each axis states what the problem is, what is already known, what
is open, and what would close it.** None of them is closed here. Where the axes interact — and they
interact heavily — the interaction is named.

### 8.A — Donor eligibility: which models are even candidates

**The problem.** "1B–100B" is not a criterion. The output of this axis is a **checkable eligibility
predicate**: given a model card and a weights file, decide whether it is a candidate, and for which SKU.

**Known / strongly suspected:**
- §5's arithmetic says **activation sparsity in the donor is worth more than small total size**. A 30B
  sparse-MoE donor is a better candidate than an 8B dense one on the streaming term alone.
- Licence is a hard filter, not a preference. Permissive (Apache-2.0 / MIT) donors avoid an entire class
  of downstream problems. Community/research-restricted licences may forbid exactly the redistribution
  a converted artefact implies. **This is the Owner's risk, so it is stated, not decided by you.**
- Architectural family matters enormously: pre-norm vs post-norm, RMSNorm vs LayerNorm, RoPE variant,
  GQA group count, tied vs untied embeddings, SwiGLU vs GeGLU vs ReLU², QK-norm, attention sinks,
  MoE routing style (top-k softmax vs sigmoid, shared experts, aux-loss-free bias). Each of these
  either has a clean transformation or does not.
- A donor is not eligible merely because its tensor shapes can be read. The Adapter must be able to
  write a complete transformed-graph descriptor: residual topology, widths, layer order and types,
  normalization, positional encoding, activation, routing semantics, token IDs, special tokens and
  tied-weight relations. An unsupported semantic operator is a named rejection or an explicit engine
  work item, never a silent approximation.

**OPEN — the questions to answer:**
1. What is the **full eligibility predicate**? Draft it as a checklist with a reason attached to every
   item, then *test it* by applying it to a broad survey of real released models.
2. Does **vocabulary size** belong in the predicate as a hard filter (per §5's head arithmetic), or is
   the head compressible enough (§8.H) that it becomes a cost rather than a gate?
3. Do **MoE donors** carry hidden disqualifiers — shared-expert layers, fine-grained experts too small
   for the ρ-safe 48 KB granularity, routing that requires full-precision logits, load-balancing state?
4. Does a **hybrid donor** (models already mixing attention with SSM/linear-attention layers) short-cut
   §4's central tension by arriving half-converted? This is the highest-leverage question on this axis.
5. What about **already-quantized releases** — do they help (less work) or hurt (information already
   destroyed, transformations no longer valid on the original weight distribution)?
6. What is the **distribution policy** for the converted artefact? Check the exact weight licence,
   model revision, redistribution/derivative restrictions, tokenizer licence and any gated-access or
   acceptable-use terms. “Open weights” alone is not a redistribution grant; record the result without
   giving legal advice.

**Closure criterion:** the predicate is closed when it has been applied to a documented survey of at
least ~15 real candidate models spanning the size range and the architectural families, produces a
ranked shortlist with reasons, and **at least one rejection is justified by a measurement rather than
by inspection**. The last clause is deliberate: a predicate that has never rejected anything for a
measured reason has not been tested.

### 8.B — The operator boundary: attention vs SSM, and the minority fraction

**The problem.** S3 permits attention on a minority of layers. Which layers, chosen how, and what
happens to the rest?

**Known:**
- The engine already implements one SWA layer (window 128) alongside SSM layers, so the hybrid
  execution pattern exists in C and is parity-gated.
- The published transformer→SSM conversions initialize SSM projections from the donor's QKV weights and
  then heal with billions of tokens. **The initialization is the part S1 permits.**
- The SSM's advantage is **O(1) state vs growing KV cache** — a memory/long-context property. Its share
  of the per-token compute floor at sandbox scale is ~13-15%.

**OPEN:**
1. **Which layers must stay attention?** The literature on layer importance suggests the first and last
   layers behave differently from the middle, and that some heads are functionally special (induction,
   retrieval). Is there a *measurable* per-layer criterion — attention entropy, effective rank, distance
   profile, retrieval behaviour — that predicts which layers survive SSM conversion? **This is where the
   project's existing MQAR-style retrieval diagnostic is directly reusable.**
2. **What is the actual conversion map?** Q/K/V/O to SSM in_proj / x_proj / dt_proj / out_proj is
   dimensionally suggestive but not canonical. State it explicitly, derive it, and be honest about which
   parts are principled and which are analogy.
3. **Does closed-form or layer-local matching close the fidelity gap** left by initialization alone
   (§4's (i))? Measure the fidelity, do not assume it.
4. **What happens to RoPE** on layers that become SSM — and to positional consistency between converted
   and unconverted layers in the same stack? This is a correctness question, not an aesthetic one.
5. **The KV cache is a footprint line item that §5's table ignores.** For every layer left as attention,
   compute KV bytes at the target context length and charge it against the SKU budget. At 128K context
   this can dominate everything else, and it is the axis where "minority" pays for itself.
6. **What is the fraction?** Sweep it. Report the retention-vs-fraction curve; that curve *is* the
   deliverable of this axis.

**Closure criterion:** a retention-vs-attention-fraction curve on at least one real donor, with the
layer-selection rule stated in advance and a control arm that selects layers randomly at the same
fraction. **Without the random-selection control the curve proves nothing about your selection rule.**

### 8.C — Ternarization without training

**The problem.** The engine's kernels are ternary. The donor's weights are not. Quantization-aware
training is what makes ternary nearly free, and S1 forbids it.

**Known:**
- This project's ternary numbers (+0.028 BPB; combined +0.013 ± 0.005) are **QAT-from-scratch** at 8.3M.
  They do not transfer to a post-training setting and must not be quoted as if they did.
- Post-training ternarization is measured in the literature as **severely lossy** (roughly ~2× perplexity
  at 7B–70B in at least one published study). **Assume the naive path is broken and design around it.**
  Verify this figure yourself before building on it.
- The literature's escape routes are mathematical and mostly data-free or calibration-light:
  rotation/orthogonal transforms that destroy outlier structure (Hadamard-style, "computationally
  invariant" reparameterizations), activation-weight scale migration, error-compensating sequential
  quantization, salience-weighted scaling. **These are exactly the shape S1 asks for.** Treat this list
  as leads to verify, not as established fact.
- **Mixed precision is legitimate and already precedented here:** Phase 61 measured that ternarizing the
  SSM projections costs +0.018-0.022 BPB and *rejected* it — the projections stay fp32. A donor
  conversion will need its own precision map, derived the same way: by measurement, per organ.
- **Stage −1 is the mandatory cheap kill gate for this axis.** It asks whether a bounded, best-available
  calibration-only ternary PTQ path has enough donor fidelity on a 1–3B control donor to justify any
  downstream ternary-engine work. Passing it is not evidence that a large donor fits S2 or passes S4;
  failing it is a decision point about the ternary-primary thesis before expensive engineering.

**OPEN:**
1. What is the **best achievable ternary quality** on a real donor under a calibration-only budget, and
   how does it decompose by layer type and by depth?
2. **Which organs must stay higher-precision**, and what does the resulting footprint inventory look
   like against S2? This is a direct trade: every fp32 organ is 8× the bytes of a ternary one.
3. Is **ternary even the right target for a donor**, or is the correct answer per-organ mixed
   (ternary MLP / higher-precision attention and head)? Note the engine's kernel advantage is
   specifically ternary-LUT — measured 4.2-5.0× vs fp32, and **faster as bits drop**, which is the
   opposite of the int8 story on this silicon. A design that quietly drifts to int8 has given up the
   project's main kernel advantage and must justify that explicitly.
4. Does the **grouping/packing granularity** (per-row scales today) need to become per-group at donor
   scale? Outlier channels are known to emerge with model size, and per-group scales are the standard
   escalation — but they cost bytes and complicate the kernel.
5. What is the **interaction with 8.B**: does a layer converted to SSM ternarize differently from one
   left as attention? Do not assume independence; measure it.

**Closure criterion:** a per-organ precision map with a measured quality cost for each choice on a real
donor, plus a footprint inventory that adds up to a number checkable against S2. The map is closed only
when the *rejected* alternatives are also measured — this project rejected ternary projections *with a
number*, not by argument.

### 8.D — Activation sparsity when it is not native

**The problem.** The engine's 2.12× active-weight shrink comes from gated **dReLU** sparsity: 92% of
hidden units inactive per token. Donors use SwiGLU/GeGLU, whose activations are dense — nothing is
exactly zero, so nothing is skippable.

**Known:**
- The measured skip mechanism is **predictor-free and exact**: gate-first evaluation, then row-skip on
  up and tile-skip on down. It is a correctness-preserving optimization, not an approximation.
- The 92% figure is **learned behaviour of a trained network, not a structural property.** This project
  explicitly re-measures it at every rung and never assumes it transfers. A donor has no reason to
  exhibit it.
- The literature's conversion path ("ReLUfication" / dReLU-substitution) is well known and reports
  large sparsity gains — **but the published recipes involve continued pretraining**, which is what
  S1 forbids. This is the same tension as §4, on a different axis.
- A **temporal-coherence regularizer** was measured here to make sparsity block-structured (cache-
  friendly) at zero quality cost. That was a training-time lever and is unavailable to you — but it
  tells you what property you need: contiguity, not just count.

**OPEN:**
1. Is there a **training-free** route to exploitable sparsity? Candidates to evaluate: thresholding
   small activations with error compensation (approximate, needs a quality gate); exploiting the natural
   near-zero mass of SwiGLU with a calibrated cutoff; predicting the active set from the input with a
   tiny cheap probe (the in-place predictability measured here at 86-92% is directly relevant, and it
   was measured on *this* architecture — re-measure it on the donor before relying on it).
2. If sparsity becomes **approximate rather than exact**, the engine's parity gate no longer applies as
   written. **You must define what replaces it** — an error-bounded gate, a statistical gate — and
   pre-register it. Do not let a lossless-by-construction system quietly become an approximate one
   without a new contract.
3. Does sparsity even matter for a **sparse-MoE donor**, where the MoE routing already delivers the
   active-set reduction? Possibly the two levers are largely redundant, in which case this axis drops
   in priority for the best donor class. **Quantify the overlap before investing here.**
4. Is the sparsity **contiguous enough to skip cheaply**, or is it a scatter that the ρ-law punishes?
   Sparsity you cannot skip in bulk is worth nothing on this silicon.

**Closure criterion:** measured sparsity fraction *and* measured achieved speedup on the real engine —
**both**, because this project has already learned that a microbenchmark speedup does not compose to an
engine speedup (Phase 61: compute-bound microbench gains vanished in a memory-bound engine).

### 8.E — Making a dense donor sparse: MoE-ification

**The problem.** The engine's capacity tier is a granular ternary MoE. A dense donor has one big MLP
per layer. Can a dense MLP be *partitioned* into experts by mathematics alone?

**Known:**
- This is a real published technique (splitting a dense FFN into expert groups by neuron co-activation
  clustering, with a learned or derived router) and it is **fundamentally a clustering problem, not a
  training problem** — which puts it squarely inside S1's permitted space.
- This project measured that **granular experts beat coarse ones** at matched params, and that routing
  shows **no temporal locality** — so the resulting experts will be DRAM-streamed, and their size must
  respect the ~48 KB bulk-contiguous granularity that keeps the ρ-law penalty at ~2.5× instead of ~14×.
- The router in the engine is a small fp32 Linear + bias per layer, resident.

**OPEN:**
1. Can co-activation clustering produce experts whose **top-k selection preserves the dense output**
   closely enough, on a calibration set, without training the router?
2. **How is the router derived** without training? Options span from a closed-form projection of the
   clustering to a tiny logistic fit on calibration statistics (cheap, arguably not "training", but
   declare it under S1's grey-zone rule).
3. What **expert count and size** optimize the (quality × granularity × ρ-safety) product for a given
   donor? The engine's own answer at sandbox scale was E32×h128 top-8; a donor's dimensions are wildly
   different and the answer will be too.
4. Does MoE-ification **compose with ternarization**, or does clustering on fp32 weights become invalid
   once the weights are ternarized? **Order of operations is a real variable here** — rotate, then
   cluster, then quantize, in some order — and this project's law is one variable per stage.
5. For a donor that is **already MoE**, this axis becomes *expert re-granulation* instead: are the
   donor's experts the right size for the ρ-law, and can they be split or merged safely?

**Closure criterion:** dense-vs-MoE-ified output fidelity on held-out calibration data, plus end-to-end
retention (§S4), plus the measured streaming cost of the resulting expert layout on the real engine.

### 8.F — The thinking/knowing partition: what actually lives in cache

**The problem.** This is the project's founding thesis and the axis the Owner named first. Given a
converted donor, **which parameters are resident and reused every token, and which are streamed?**

**Known:**
- The keystone constraint, parametric form: the active per-token slice must fit the **aggregate LLC at
  the operating thread count** (reference: 2 × 16 MB at 6 threads, ~185 GB/s while resident, sloping to
  ~34-36 GB/s fully streamed). At 0.5 B/weight that is **~24-32M active ternary params** for a
  single-CCX 16 MB budget.
- The current split: resident = SSM backbone + projections + router + head + norms + embeddings;
  streamed = the selected experts, in contiguous per-expert blocks.
- **There is no third option.** Every byte is either resident-and-reused or bulk-streamed. Anything
  else meets the ρ-law.

**OPEN:**
1. For a donor, **what is the natural resident set** and does it fit? A donor's attention projections
   and head are much larger than this project's; the inventory may not fit at all, which would force
   either compression (§8.C, §8.H) or a different split.
2. **Is the split even the same shape for a donor?** The current split follows the architecture
   (backbone vs experts). A donor might partition better by *depth* (early layers resident, late
   streamed), by *organ*, or by measured *reuse frequency*. This is genuinely open and it is one of the
   most interesting questions in the document.
3. **Is there a residency benefit at all**, given that routing locality was falsified here? The measured
   answer for *experts* is no. It might differ for a donor whose routing was trained differently —
   and if it does, **that is a publishable finding**, not a lucky break. Design the measurement so it
   could come out either way.
4. **What is the cost of getting the partition wrong**, and can it be tuned automatically at load time
   rather than decided at export time? This connects directly to §8.K.
5. Where does the **recall tier** fit? It is measured, cheap, and currently unfused with the engine
   core. For a donor with a large KV footprint at long context, an indexed recall tier is a candidate
   *replacement* for part of the attention memory — which is a bigger idea than it first looks and
   deserves its own evaluation.

**Closure criterion:** a residency inventory for a real converted donor, measured tok/s at the chosen
split, and at least one *alternative* split measured for comparison. A partition with no measured
alternative is an assumption wearing a number.

### 8.G — Lookup tables: the kernel contract

**The problem.** The engine's speed comes from ternary weights served through `pshufb` byte-LUTs. Any
converted donor must produce weights in exactly the layout those kernels expect, or produce new kernels.

**Known:**
- Ternary weights are packed at **4 bits/weight today** (base-3, g=2 codes) = 0.5 B/weight. The dense
  trit-pack (5 trits per byte-pair, ~1.6 bits/weight = 0.2 B/weight) is **designed but NOT built** and
  would be a ~2.4-2.5× reduction in streamed bytes. Every streamed-byte figure in this project uses the
  real 4-bit packing.
- Kernels are validated by **synthetic self-tests that check bit-exactness against a scalar integer
  reference** on random ternary weights and int8 activations (`--kselftest`). That harness is the model
  for any new kernel you add.
- Weights are stored transposed/blocked for the kernel (`bc_tm`) with padding to 32-element boundaries;
  per-row scales are separate fp32 arrays.

**OPEN:**
1. Does the current LUT layout **survive donor dimensions**, or do the padding and blocking choices
   degrade at much larger D and hidden sizes? Measure the kernel rate curve at donor dimensions before
   assuming.
2. Is the **trit-pack worth building now**? It is a 2.4-2.5× streamed-byte win, it directly moves the
   S2 footprint ceilings (the second column of the SKU table), and it is pure engineering with a clean
   bit-exactness gate. **This is the highest-value known-quantity engineering item in the document** —
   but it is also the classic trap: a satisfying, well-defined task that is not the research question.
   Cost it, propose it, and let the Architect decide when it is scheduled.
3. Do donor weights need **per-group rather than per-row scales** (§8.C.4), and what does that cost in
   the kernel's inner loop?
4. If §8.B keeps attention layers, do their projections go through the **same LUT path** or a different
   one — and if a different one, what is its parity gate?
5. **Activation quantization**: the LUT path takes int8 activations (per-token scale, validated at
   sandbox scale). Donor activations have known outlier channels at scale. Per-group activation scales
   are the standard escalation and are **not implemented**. This is a likely required work item and it
   interacts with every rotation-based method in §8.C.

**Closure criterion:** for every kernel touched or added, a synthetic bit-exactness self-test against a
scalar reference *and* end-to-end parity on a real model. Both, per law 8 in §6.

### 8.H — Vocabulary, embeddings and the head

**The problem.** §5 showed the donor's head can cap throughput by itself and consume a large fraction of
the SKU budget. This project deliberately avoided the problem by keeping V ≤ 4K. A donor reintroduces it
at full force.

**Tokenizer contract, SEALED.** The primary path preserves the donor tokenizer, token IDs, special-token
semantics and vocabulary exactly. Head compression may change its representation, not which token each
ID denotes. Vocabulary pruning, token merging or a replacement tokenizer is a separate experimental arm:
it invalidates token-for-token comparison, must be evaluated in bytes as well as tokens, and cannot be
combined silently with a primary retention claim.

**Known:**
- The compressed-head probe ("probe-5") was **specified and never engaged**, precisely because this
  project's own vocabulary is small. It is now in scope.
- Candidate mathematics: tied input/output embeddings, low-rank factorization, clustered/adaptive
  softmax, vector-quantized heads, aggressive quantization of the head specifically. Each has a
  different interaction with the LUT kernel and with cache residency.
- The head is currently fp32 and resident; at sandbox scale it costs 40 → 8 µs/token (1 → 6 threads).

**OPEN:**
1. **What is the quality cost of each head-compression method** on a donor, measured, not assumed?
2. Can the vocabulary itself be **reduced** — pruning rarely-used tokens, merging, or re-mapping to a
   smaller domain-specific vocabulary — without retraining the embedding? This is a mathematical
   question (it is a projection problem) and a data question (what is "rarely used" is corpus-dependent).
3. **Does a smaller vocabulary hurt bytes-per-token?** Fewer tokens per byte means more forward passes
   per byte of output — the compute floor is per *position*. This project already noted vocabulary is
   quietly a compute-floor lever in *bytes/s*, not just tok/s. **Whatever you measure, normalize to
   bytes, not tokens** — this project's mantra guard exists exactly because unit choices hide effects.
4. How do embeddings and head interact with **SKU-A's 16 GB budget** specifically? For some donors this
   single organ may decide SKU eligibility.
5. Can exact-token output avoid evaluating every vocabulary row without changing greedy selection or
   likelihood scores? If it cannot, candidate/clustered output is an approximation and needs its own
   transformed-reference and retention contract.

**Closure criterion:** a measured quality-vs-footprint-vs-speed curve for at least three head treatments
on one real donor, reported in bytes-per-byte terms.

### 8.I — Context, the KV cache and long-range memory

**The problem.** The engine's SSM has O(1) state. Attention has a KV cache that grows linearly with
context and can dwarf the weights. S3 keeps some attention. Something has to give at long context.

**SEALED contract, not a measurement:** SKU-B is native 128K; SKU-A reaches 128K through bounded native
attention plus the recall tier. The quality requirement is in S3/S4; it has not yet been demonstrated.

**Known, measured:**
- The recall tier is IVF-Hadamard partition + 4-bit ADC + exact rerank, measured 29.05 µs/token at
  128K entries, 1.69 MB searchable + 64 MB values. Drift is a non-problem on this SSM because the
  state norm is bounded.
- The recall tier has **never been fused with the engine core**. It is measured standalone.
- Sliding-window attention (window 128) is already implemented and bounds KV for the layers that use it.

**OPEN:**
1. **What is the KV footprint of the retained attention layers** at each SKU's target context, and does
   it fit? Compute it explicitly per donor — it is arithmetic, it is free, and it may reorder your
   entire donor shortlist.
2. Can retained attention layers be **converted to sliding-window** (bounding KV) without unacceptable
   loss? This is a cheap, mathematically trivial intervention with a real quality question attached, and
   it is probably the first thing to measure on this axis.
3. Can the **recall tier substitute for global attention** on some layers — retrieval instead of full
   context? This is the project's most original standing idea and a donor is an unusually good test bed
   for it, because the donor supplies trained representations the recall tier can index.
4. **KV quantization** is standard practice and orthogonal to everything else here. It is probably
   necessary; measure its cost rather than inheriting a literature number.
5. **How is recall fused without changing the baseline when disabled?** Specify the engine interface,
   memory ownership, token-position/key contract, deterministic ordering and failure/refusal path. The
   recall-off path must preserve the otherwise identical engine logits; the recall-on path must expose
   its memory and per-token overhead. This is the stage-4b engineering gate, before it is credited with
   any converted-model quality.
6. What context length does the **converted** model actually retain, versus the donor's advertised one?
   Advertised context lengths are frequently optimistic even before conversion. **Measure it; do not
   quote the model card.**

**Closure criterion:** measured retention on a long-context retrieval task (the existing MQAR-style
diagnostic generalizes) at **128K** for the donor and each conversion variant. SKU-B must pass using its
native context path. SKU-A must pass through its bounded-attention-plus-recall path; a run at 8K–32K is
diagnostic only and cannot close this axis.

### 8.J — Engine generalization: dimensions at runtime

**The problem.** §2.4. The engine's dimensions are compile-time constants.

**This axis is engineering, not research.** It is stated as its own axis only because it blocks
everything else and because it has a trap.

**OPEN:**
1. **Runtime dimensions vs per-model compile.** Reading dimensions from the header at runtime costs some
   speed (dynamic bounds, heap allocation, lost constant-folding). Generating a specialized binary per
   model keeps the speed and complicates distribution. **Both are legitimate; measure the cost before
   choosing**, and note that a hybrid — runtime-general with a compile-time specialization path for a
   shipped model — is also available.
2. **The layer-type map must become data.** `is_swa[l] = (l == SWA_LAYER)` becomes a per-layer type
   descriptor in the header. Design that descriptor to accommodate §8.B's outcome, whatever it is.
3. The weight-file **format needs a version bump** and a real architecture descriptor. Design it so a
   file that does not match the engine's capabilities is **refused with a named error**, never read as
   garbage. This is law 5 in §6, applied to the format.
4. **Memory-mapped weights** — for donors at SKU-B scale, loading tens of GB into heap at startup may
   be untenable. `mmap` is the obvious answer and it interacts with the ρ-law and with page-fault
   behaviour on first touch. Measure it; do not assume it is free, and do not assume it is fatal.

**Closure criterion:** bit-exact parity with the current binary on the current model, at every thread
count, plus the both-directions guard exercise on the new format's refusal path (a good file loads; a
mismatched file is refused by name).

### 8.K — Hardware-adaptive residency: sizing the thinking core to the machine

**The problem — the Owner's own question, and a branch this project has never touched.** The engine
today assumes one machine's cache. It should **read the machine and adapt**: detect the cache hierarchy,
compute the available resident budget, and size the resident partition to maximize it.

**Known:**
- The keystone is already stated parametrically (aggregate LLC at the operating thread count), and the
  spill behaviour is a **measured slope**, not a cliff, when threaded — the r(size) curve in §2.2 is
  literally the model to look up against.
- Thread count, pinning and CCX topology all move the effective budget. Measured: spread pinning ≥ close
  pinning on this 2-CCX part (locality traded for aggregate bandwidth), SMT regresses, and unpinned
  6-thread streaming *regresses* against 3 threads.

**OPEN:**
1. **How does the engine learn the machine?** `CPUID` leaf 4 / leaf 0x8000001D gives cache sizes and
   sharing; core topology and CCX/CCD grouping are messier. What is portable across x86-64-v3 vendors
   and generations? What is the fallback when detection fails, and how does the fallback fail *safely*
   (a conservative budget, never an over-optimistic one)?
2. **Is the resident partition adjustable at load time at all?** This is the deep question. It requires
   the conversion (§8.F) to produce not one fixed split but a **family** of splits, or a layout that can
   be re-partitioned cheaply at load. That is a real design constraint on the export format and it must
   be decided *before* the format is frozen, not after.
3. **Should the engine auto-tune by measuring** rather than by reading spec sheets? A short startup
   micro-benchmark (a few hundred ms) measuring the actual r(size) curve on *this* machine is more
   honest than a lookup table and is exactly how this project measured the curve in the first place.
   Cost it against startup time.
4. **What is the gain?** Before building any of it: how much tok/s is actually on the table between a
   badly-sized and a well-sized partition on the same machine? **Measure the spread first.** If it is
   10%, this axis is a nice-to-have; if it is 3×, it is a headline. This ordering is not optional — it
   is the difference between engineering and speculation.
5. Does adaptivity interact with **determinism**? Parity gates are per-configuration; a model that
   partitions differently on different machines produces different *performance*, and must still produce
   **identical logits**. Make that invariant explicit and gate it: **residency is a performance decision
   and must never be a numerical one.**

**Closure criterion:** the spread measurement in (4) first, before any implementation. Then, if it
justifies the work, a measured tok/s comparison of auto-sized versus fixed partitions across at least
two genuinely different cache configurations, with logit-identity verified across both.

### 8.L — Calibration data

**The problem.** Every transformation in §8.C–8.E depends on activation statistics from a calibration
corpus. That corpus is a variable, and it is the kind of variable that quietly decides results.

**Known / project rules that bind you:**
- **Corpora are never committed or redistributed.** Manifest + script + hash only; `data/` is gitignored.
- The project's **P62 code-val is pinned and must never be modified**, and decontamination against it is
  mandatory for any training-adjacent data.
- The project has a **measured law about data ordering**: at rung-1 scale, blocked (non-i.i.d.) sampling
  order cost +0.0339 BPB = 6.8 σ_seed versus a global shuffle, at identical data, tokens and steps.
  Coverage was 100% on both sides — **order alone did it**. If your calibration draws samples in file
  order, you are inheriting a measured hazard.

**OPEN:**
1. **How much calibration data, and of what composition?** For a general-purpose target (S4), a
   code-only calibration set is a plausible way to silently destroy general capability. Measure the
   sensitivity rather than picking a default.
2. **How sensitive are the results to the calibration set?** Run the same conversion with two disjoint
   calibration sets and compare. If the answer moves more than σ, the calibration set is a load-bearing
   variable and must be pinned by hash and reported with every number.
3. **Contamination**: calibration data overlapping the eval sets invalidates §S4's retention numbers.
   Decontaminate and say how.
4. **Licence**: the same rules that govern training corpora govern calibration corpora.

**Closure criterion:** a sensitivity measurement across at least two disjoint calibration sets, with the
chosen set pinned by content hash.

### 8.M — The fallback: micro-budget healing

**The problem.** If transformation-only conversion lands below the usefulness bar, how much healing
buys how much back — and is any of it inside S1?

**This axis exists so the negative result has a next step. It is not the plan.** Under S1 it is
explicitly a fallback, it must be costed before it is proposed, and it must never migrate into the
primary path by accretion.

**OPEN:**
1. What is the **retention-vs-healing-tokens curve**, measured at the smallest budgets that fit a single
   Kaggle session? The shape of the first part of that curve is the whole question: if 90% of the loss
   comes back in a few hundred million tokens, the answer is different than if it needs tens of billions.
2. Which **parameter-efficient** forms (adapters, low-rank updates, norm-and-bias-only, single-layer
   repair) buy the most per GPU-hour?
3. Can healing be **localized** to the layers the diagnostics say are broken, rather than applied
   globally? A targeted repair is a fundamentally different cost class from end-to-end healing.
4. **What is the honest cost table** in GPU-hours and euros, per donor class, for each rung of the curve?

**Closure criterion:** the curve's first three points measured, with the cost of each stated in
GPU-hours, before any proposal to spend beyond them.

### 8.N — Evaluation: the harness that does not exist yet

**The problem.** S4 makes general-purpose retention the deciding metric. This repo has no general-purpose
eval harness. **You must build one, and it must be trustworthy before any conversion result is believed.**

**OPEN:**
1. **Which benchmarks**, and why those? Retention-% against the donor is the primary framing, which
   means the harness must run the *unconverted* donor too — that is a real compute and memory cost on
   the available hardware, and it may itself be the binding constraint on which donors you can evaluate.
2. **How do you evaluate a 60 GB model on the reference machine?** Possibly you cannot, and the donor
   baseline must be established elsewhere (published numbers are **not** an acceptable substitute —
   different harness, different prompts, different decode settings, and this project does not quote
   numbers it did not measure per-protocol).
3. **Decode hygiene is locked** in this project (greedy, repetition penalty 1.2, window 128, no top-p)
   and it must be identical across every *generative* arm. State it, enforce it, and verify it
   programmatically. Multiple-choice and likelihood tasks must instead use raw logits/scoring with no
   generation penalty or sampling policy. The harness must refuse a task run under the wrong mode.
4. **Statistical resolution:** what is the σ of each benchmark under this harness? Without it, a
   retention delta is uninterpretable. The project's σ_seed = 0.005 BPB exists precisely because
   somebody measured it; **do the equivalent for every metric you intend to decide with.**
5. **Planted control (mandatory, per §6.3):** the harness must be shown to *detect* a deliberately
   degraded model. Take a known-good donor, break it in a controlled way (quantize one layer to
   nonsense, shuffle one weight matrix), and confirm the harness's score drops by a margin much larger
   than its own noise. **A harness that has never scored a broken model has not been tested.**
6. **Task contract:** before donor evaluation, publish task membership and weights for global retention,
   the exact critical Code/Logic/Long-Context Recall tasks, metric direction, chance baselines where
   applicable, context length, prompt templates, model revision hashes and tokenizer revision. The
   SKU-A long-context task is run at 128K through its recall path; a successful short-window run does
   not substitute for it.

**Closure criterion:** the planted-control demonstration, plus a σ estimate per metric, both logged,
**before** the harness is used to judge any conversion.

---

## 9. How to work: stages, gates and stopping points

Follow this ordering. It is designed so the **cheapest questions that could kill the project are asked
first** — this project's standing discipline, and the reason it has killed bad ideas cheaply.

| stage | what | cost | gate |
|---|---|---|---|
| **0** | **Read the ground.** This document, `SCALEUP_ARCHITECTURE.md`, `PHASE64_BUDGET.md`, `PHASE64_DECISIONS.md`, `engine.c`, the export scripts. Then the prior-art survey (§11). | desk | a written statement of what you believe and what you doubt, **before** you measure anything |
| **−1** | **PTQ ternary kill gate — before engine work.** On one reproducible 1–3B donor, run the best calibration-only rotation/ternarization method selected from the stage-0 survey. No SSM conversion, no engine export, no new runtime work. Compare perplexity/log-likelihood with the original under a pre-registered threshold and a planted corruption control. | days; CPU-first; no GPU session without owner approval | If the pre-registered PTQ fidelity gate fails, stop the ternary-primary route and ask the Owner whether a named 4-bit/mixed-precision engine path is worth pursuing. This is evidence against the tested, bounded PTQ path — not a claim that all possible PTQ is impossible. |
| **1** | **The arithmetic pass.** Redo §5 properly for a shortlist of real donors: footprint inventory, streaming budget, KV budget, head cost, per-SKU eligibility. All desk, all free. | desk | a ranked shortlist with the arithmetic shown; **any donor whose arithmetic fails is eliminated here, before any compute** |
| **2a** | **The killer-question screen (§4).** In a reference apparatus only, make the cheapest possible per-layer measurement of attention→SSM without healing, and test whether it is necessary at all (hypothesis (iii)). It cannot make a general-retention claim. | small | pre-registered local-fidelity threshold and a written route decision |
| **3** | **Engine generalization (§8.J).** Runtime dimensions, format v2, refusal path. | engineering | bit-exact parity, both-directions guard exercise |
| **4a** | **The eval harness (§8.N).** Including the planted control and the σ estimates. | engineering | harness detects a deliberately broken model |
| **4b** | **SKU-A recall fusion (§8.I).** Fuse the existing recall tier into the engine behind an explicit interface, with recall-off behaviour preserving baseline logits and a 128K end-to-end plumbing/overhead exercise. This is an engineering readiness gate, not proof of converted-model quality. | engineering | recall-disabled parity; both-direction interface guards; measured 128K memory and overhead inventory; no silent fallback to an 8K–32K-only result |
| **5** | **One donor, end to end.** The single most promising candidate: convert, export, run on the engine, measure the sealed retention, 10 tok/s, RAM, and SKU-context gates. **One variable at a time.** | the real cost | full property gate set, pre-registered |
| **6** | **Generalize or fail honestly.** Second donor to test whether the recipe transfers, or a documented negative result with the mechanism named. | — | §10 |

**Stopping points where you must stop and ask a human:**
- before **anything** that consumes a GPU session or money;
- before proposing to amend a sealed constraint (S1–S4);
- when a measurement contradicts a claim in this document — **that is a finding, and it goes back to the
  Architect immediately, not into a footnote**;
- when you find yourself about to loosen a gate you pre-registered. **That is the moment the project's
  central law exists for.**

**Scope rule.** At stage 5, choose one primary conversion route and state the axes deliberately held
constant. Do not combine attention→SSM, MoE-ification, activation sparsification, tokenizer changes,
head approximation and a new recall fusion in a single claimed result. A route that misses a gate may
open one named next variable; it does not authorize changing the entire stack at once.

---

## 10. What failure looks like, and why it is acceptable

**The honest negative results available here, in descending order of value:**

1. **"Transformation-only conversion loses X% of capability, the loss is concentrated in [mechanism],
   and here is the measurement that localizes it."** This is a genuinely valuable result. It tells
   everyone what the healing budget actually buys and why.
2. **"Donor class C is convertible; donor class D is not, and the discriminating property is P."** An
   eligibility predicate with measured teeth is a real contribution.
3. **"The SSM backbone is not load-bearing for CPU speed; the ternary-sparse-cache-resident properties
   are."** This would restructure the project's thesis — and it is *cheap to test* (§4.iii).
4. **"It works."** The best outcome and the least likely on the first attempt.

**What is NOT acceptable:**
- a result whose gate was adjusted after the number appeared;
- a speedup measured in a microbenchmark and reported as a system speedup (**measured failure mode in
  this project — Phase 61**);
- a quality number quoted without its scale, seed count and protocol;
- a working system whose correctness was never verified end-to-end against a reference;
- **a plausible artefact** — the tool that returned zeros, the checkpoint that loaded silently with the
  wrong architecture, the guard that never fired. This is the project's characteristic failure and it has
  happened repeatedly. Assume it is happening to you right now and go check.

---

## 11. Prior art to survey — leads, not facts

**Verify every one of these before relying on it.** They are listed because they are the neighbourhoods
where the answers probably live, not because their claims are established here. The project's law: a
mechanism claim must be derived or measured, never asserted because it sounds right.

- **Low-bit CPU kernels:** T-MAC (LUT GEMV via `pshufb`/`tbl`), bitnet.cpp, DeepGEMM. This project's
  measured 4.2-5.0× on Zen 2 sits inside their published range; the novelty claimed here is the old-gen
  no-VNNI silicon plus end-to-end SSM integration, not the kernel idea.
- **Ternary / low-bit training and PTQ:** BitNet b1.58, Spectra/TriLM (the small-scale gap), and the
  post-training ternary literature reporting large degradation at 7B-70B. **The QAT-vs-PTQ distinction
  is the crux of §8.C** — do not blur it.
- **Rotation and invariance-based quantization:** the "computationally invariant" family — Hadamard and
  learned-orthogonal transforms applied to make weights and activations quantizable by removing outlier
  structure, and orthogonal-transform-then-slice pruning. **This is the most promising mathematical
  neighbourhood for S1** and should be surveyed first and in depth.
- **Calibration-based PTQ:** error-compensating sequential quantization, salience-weighted scaling,
  activation-to-weight scale migration. Layer-local, cheap, and squarely inside S1's grey zone.
- **Activation sparsity:** ReLU-substitution and dReLU-conversion work, and the "ReLU strikes back"
  observation about intrinsic sparsity. **Check the training cost of every published recipe** — that is
  the number that decides whether it is available to you.
- **MoE-ification:** splitting dense FFNs into experts by co-activation clustering; expert
  segmentation/granularity work; upcycling. **The clustering half is training-free; check what the
  router costs.**
- **Attention→SSM conversion:** the Mamba-in-the-Llama and MOHAWK families — QKV-to-SSM initialization
  plus distillation. **Read these for the initialization mathematics and note the healing budget as the
  thing you are trying to avoid.**
- **Hybrid architectures:** models that already interleave attention with SSM/linear-attention layers,
  which may be donors that arrive half-converted (§8.A.4).
- **Long context:** KV-cache quantization and eviction, retrieval-augmented attention — the
  neighbourhood of §8.I.3.

---

## 12. Where things live in this repo

| what | where |
|---|---|
| the engine | `benchmarks/phase60/engine.c` (dimensions at `:51-75`, loader at `:194-240`, layer-type map at `:202`) |
| weight export | `benchmarks/phase60/e4_export.py` (MoE, `E4M1`), `e1_export.py` (dense, `E1M1`) |
| build and gates | `Makefile` (`make engines`, `make selftest`, `make gates`) |
| the blueprint | `docs/SCALEUP_ARCHITECTURE.md` — read §1, §2, §3 in full |
| the measured budget model | `docs/PHASE64_BUDGET.md` §1, §1b, §2 — the rate curves live here |
| sealed design decisions | `docs/PHASE64_DECISIONS.md` (D1-D9) |
| sizing arithmetic | `docs/SIZING.md` — including its own honest list of unknowns |
| eval contract | `docs/CANONICAL_EVAL.md` |
| reproduction | `docs/REPRODUCE.md` |
| recall tier | `benchmarks/phase56/` |
| the narrative history | `HANDOFF.md`, `docs/silicon_book/` (Appendix C is the reusable findings register) |
| knowledge graph | `graphify-out/` — use `graphify query "<question>"` before grepping |

---

## 13. First response expected from you

Before any code and before any measurement, produce:

1. **What you understood the problem to be**, in your own words, including which of §4's three branches
   you think is most likely and why.
2. **What you doubt** in this document — including any number you think is wrong or any inference you
   think does not follow. This document was written by the Architect from the repo's measurements; it is
   not above correction, and **finding an error in it is a contribution, not an impertinence.**
3. **Your proposed pre-registration for stage −1** — the named 1–3B donor and exact revision, PTQ
   method selected from the survey, calibration corpus/hash and budget, perplexity/log-likelihood
   fidelity threshold, control corruption, and the explicit decision requested if it fails. This must
   be written before the PTQ result exists.
4. **The §9 stage-1 arithmetic**, at least in draft: a first donor shortlist with the *complete* RAM
   inventory, implementation-aware streaming budget, KV, head, prefill/TTFT risk and per-SKU context
   contract. Label every physical lower bound as such. This is free, it is desk work, and it will
   probably eliminate more candidates than any experiment.
5. **Your proposed pre-registration for stage 2a** — the local killer question, its fidelity criterion,
   calibration and local-solve budget, planted control, and cost, written before the measurement exists.
6. **A decision matrix** showing the three §4 routes, the next cheapest discriminating measurement for
   each, and what result would lead to GO, NO-GO or an amendment request. State which conversion axes
   the first end-to-end donor trial will hold constant.

Do not start implementing. Start by being right about what the problem is.
