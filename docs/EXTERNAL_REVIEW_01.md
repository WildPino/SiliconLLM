# External Review #1 — Scale-up risk audit + our responses

**Date:** 2026-06-30
**Subject:** external technical review of the blueprint `docs/SCALEUP_ARCHITECTURE.md` (C engine, CPU-native SLM for Zen 2).
**Status:** review received; responses with measured data. Re-opened at probe-4 (MoE) on 2026-07-02 — see Addendum. Still to be re-opened at the execution-model stage (MTP).
**Provenance:** AI-assisted adversarial review, solicited by the author — a red-team exercise, not an independent third-party audit.

---

## Reviewer's verdict (summary)

The design was judged among the most elegant/ambitious seen for a small language model on edge/CPU: it combines recent state-of-the-art (Mamba/SSM, BitNet b1.58, T-MAC, granular MoE, speculative/MTP) into a single architecture co-designed for the physical limits of the hardware. The vulnerability — the reviewer argues — **is not in the software but in the emergent behavior of the networks**: very small networks hate extreme quantization and struggle to make reliable multi-token predictions. Three potential engineering traps in scale-up.

**Our note:** the reviewer's point #2 is not an outside critique — it is our own **ρ-law**, re-derived from scratch. It is the spine of the project. The method we use (empirical probes with pre-registered gates, not paper-design) exists *precisely* because we do not trust that the papers compose: we measure it on real Zen 2 / real training.

## Scorecard

| # | Reviewer's concern | Our state | Key datum |
|---|---|---|---|
| 1 | Ternary quality collapse at micro-scale (BitNet matches FP16 only ~3B) | **MEASURED** | +0.028 BPB at ~8.3M (worst-case of scale), MLP-only |
| 2 | MoE → random DRAM latency (pointer-chasing, ~70-90 ns) | **This is the ρ-law; defended in 3 layers; measured today (58.A)** | churn confirmed (persistence=null) BUT in-place predictability 88% |
| 3 | Insufficient MTP acceptance rate at small scale | **Pre-registered bound, deferred** | honest a≈2-4, content-dependent; acceptance not yet measured |

Two of three already on the ground with numbers; one bounded and queued. None is a blind spot.

---

## The reviewer's closing question — have you already run the diagnostic BitLinear158 training at 5M/10M on TinyStories vs FP32?

**Yes — that is exactly probe-1.** ~8.3M Arch-A, ternary MLP trained from scratch (QAT) vs fp32, matched recipe, on TinyStories:

> **BPB 0.8382 (ternary) vs 0.8104 (fp32) = +0.0278** (+3.4% relative).

Not an explosion, not aphasia. And it is an **upper bound**: the ternary curve was still descending at 10785 steps (0.86→0.834 in the last third) while the 0.8104 fp32 is the more-converged reference → part of the gap is under-training, not quantization. The empirical premise ("at micro-scale the BPB explodes") is falsified on our data, on the reviewer's own test bench.

---

## Point 1 — Micro-scale ternary collapse

**Concern:** ternary models need over-parameterization; BitNet b1.58 matches FP16 only around ~3B params. At 5M/100M the resolution is too low to absorb quantization noise → risk of BPB exploding → "syntactic aphasia".

**Response — right direction, wrong magnitude.** The reviewer is right that we are below the 3B crossover (ternary IS slightly worse); wrong on the magnitude (+0.028, not catastrophic). Three reasons:

1. **We ternarize ONLY the MLP, not the whole model.** SSM/attention/embedding/head/norms stay fp32. BitNet's 3B crossover is for *full* ternarization (every linear). We do mixed precision: ternary only the byte-sink (MLP, ~63% of weights). The resolution of the recurrent backbone — which carries the sequential/long-range structure — is preserved at fp32. A structurally different experiment from "full BitNet at 5M".
2. **The gap SHRINKS with scale, it does not explode.** The reviewer's fear ("small networks hate quantization") points in the opposite direction to BitNet's scaling: the ternary gap *decreases* as you scale up. We are at the **pessimistic** end of the curve (micro = worst-case) and even there it is +0.028. At 50M/500M it should shrink further.
3. **We do not claim "lossless".** The BitNet-lossless claim is at the 2B/4T frontier. Ours: small-cost = +0.028 BPB for 4-5× on the matmul + ~10× fewer bytes/weight. A trade made knowingly.

**Honest concession:** this is TinyStories (simple prose, narrow distribution). Probe-1 de-risks the mechanism and the micro-scale cost, NOT the full-distribution / scale-up cost. That remains an open unknown. But the specific claim (explosion at micro-scale) is refuted.

---

## Point 2 — MoE and random DRAM latency

**Concern:** MoE solves bandwidth but exposes you to *random latency*. The router calls expert 3, then 42, then 12 → continual cache misses → pointer-chasing in DRAM (~70-90 ns per random access) → the CPU stalls waiting for the expert → cancels the advantage of the 16 MB slice.

**Response — this is our ρ-law. Three-layer defense, plus a fresh concession from today.**

**Layer 1 (probe-3): the hot pool is sized to live in L3, not to be streamed from DRAM per token.** The entire keystone (`active_slice ≤16MB = L3-per-CCX`) exists to avoid *exactly* this scenario. If the hot experts live in L3, the lookup is an L3 access (~10-15 ns), not random DRAM (~70-90 ns). The reviewer assumes the experts are in DRAM; the design keeps them in L3.

**Layer 2 (block-decode / R-F): converts "per-token random gather" into "per-block sequential bulk".** The key architectural point. The chassis processes N positions together: you compute the router for all N, take the **union** of experts touched, load each expert in the union **once**, and apply it to the subset of positions that route to it. The router's randomness is paid **in COMPUTE** (cheap, in-cache once loaded), **NOT in memory latency**. Memory sees a sequential bulk load of the hot pool per block, prefetchable, bandwidth-bound. The 70-90 ns hit the unpredictable single-element gather, not "load these N known contiguous expert blocks".

**Layer 3 (predictor / Phase 58, in flight): forecast the union from cheap state → load only what's needed, hide the latency by issuing early.** This is *precisely* why the predictor is sequenced **before** the MoE: it turns unpredictable routing into predicted bulk loads. Point #2, in our architecture, is the reason Phase 58 exists.

**Fresh concession (58.A run on 2026-06-30, on `sp_drelu.pt`):** we measured the temporal locality the concern depends on. A mixed, honest verdict:
- **Persistence = NULL** (`P(active_t | active_{t-1}) ≈ base-rate`, temporal independence). → **the churn the reviewer fears is CONFIRMED**: the active set reshuffles every token, no free token-to-token reuse, no "overlap bonus" that shrinks the union.
- **BUT in-place predictability is HIGH: 83–89% at block-32, 88–96% at block-8** (predictor-free, on a frozen + under-trained checkpoint). → the active set is *unpredictable in time* but *predictable from the current state* = the raw material for Layer 3.

The defense **does not rest on persistence** (we falsified it) — it rests on in-place predictability (high) + per-block batching. **Still open:** whether the union / hot pool actually fits in 16 MB at real scale (hundreds of experts) = probe-4; whether the predictor holds at expert granularity = what 58.B is validating. Ultimate resolution pending, but the signal (88% in-place) is encouraging and the mechanism (block-batch → sequential) is structural.

---

## Point 3 — MTP acceptance rate at small scale

**Concern:** speculative/MTP only wins if the head is accurate; small networks have intrinsically higher perplexity → low acceptance → wasted cycles on discarded tokens → *slower* than plain autoregressive. Will a 6.7M/50M model hold a sufficient acceptance rate?

**Response — a pre-registered risk, not a surprise.**

1. **The number is already bounded in the design.** R-F/R-H fix the honest acceptance at **a≈2-4** (geometric decay), NOT N; the few-step carve at **c≈3-5**, limited by conditional independence, not tuning. We never promised N×. The claimed speedup is already the conservative one, not the ceiling.
2. **Content-dependence is the lever, and it points to the product.** R-H: acceptance/carve is content-dependent — much higher on structured Cat-A content (code, logs, structured output) than on free prose. The concern is calibrated to **prose** (TinyStories, high perplexity) = our worst-case, not the target. Capstone: the engine is fastest where the product points (agentic = high-`c` Cat-A).
3. **On an SSM the rejection cost is lower than on a transformer-AR.** R-H reversal: on an SSM the carve dissolves the state-rollback crux (you recompute the block → nothing to roll back); verification is O(N) **linear** (parallel scan, no KV, no tree). Cycles on rejected tokens cost less: one scan pass, not a quadratic tree with KV.
4. **MTP is not the only batching lever.** The chassis also gets parallelism from R-G (a-priori plan → skeleton-of-thought parallel fill) and from multi-stream (server side). The design does not live-or-die on single-user acceptance.

**Honest concession:** the acceptance rate is not **measured** at any scale. The execution-model/MTP is an **open** piece (queued after predictor + MoE). We have a conservative pre-registered bound + a content-dependence argument + a structural-SSM argument. The number itself is future work. We don't oversell it.

---

## Summary

The reviewer's meta-diagnosis is correct: the vulnerability is in the emergent behavior of the networks, not in the software. That is exactly why our method is probes-with-pre-registered-gates, not paper-design. Of the three traps: two are already on the ground with numbers (ternary +0.028 measured; MoE latency defended in 3 layers with 58.A in support), one is bounded-and-queued (MTP acceptance, pre-registered bound a≈2-4). To be re-opened with fresh data at probe-4 (does the hot pool fit in 16 MB at scale?) and at the execution-model stage (real acceptance).

---

## Addendum — probe-4 (MoE) results, 2026-07-02

Point #2 re-opened as promised, now with trained arms and direct locality measurements (matched total params ~22.5M, 4k steps, router fp32 + Switch load-balance aux; A dense-1024 BPB 0.8799 | B dense-4096 0.8674 | **C MoE E32×h128 top-8 = 0.8589** | D MoE E8×h512 top-2 = 0.8637).

**Concession, measured: there is NO hot pool.** The expert working-set over a block of N positions sits at the i.i.d. baseline (C: 84-89% of all experts touched at N=8 vs 90% for random routing; saturation ~100% by N=16-32), and expert persistence equals the base-rate k/E ≈ 25% (D slightly below = mildly anti-persistent). The temporal-independence finding of 58.A replicates at expert granularity — twice measured, at two granularities: routing does not concentrate. The reviewer's instinct on churn was right. (Caveat cutting both ways: TinyStories is homogeneous, so domain-conditional concentration cannot show up here; we do not count on it appearing at scale.)

**But the quantified consequence is a bounded refinement, not a collapse:**
- The expert tier is re-classified from "L3-warm pool" to **DRAM-streamed, granularity-bounded**. Only that tier loses the ~3× residency multiplier; backbone/router/head stay L3-resident.
- Bandwidth: a granular ternary expert (h=128) is tens of KB; top-k × layers = a few MB per token → at the measured 28 GB/s DRAM floor this supports thousands of tok/s on that tier — not the bottleneck.
- Latency (the original concern): each expert load is a tens-of-KB **contiguous** chunk, loaded in bulk per block — prefetcher-friendly sequential streaming, not 64B-grain pointer-chasing. The ρ-penalty measured in probe-3 applies to fine-grain random gather, not to this pattern.
- Both pre-registered gates passed: router health (0 dead experts, max/mean ≤1.47×) and quality (BPB(C) 0.8589 vs gate ≤0.8899 — C in fact beat the dense-4096 arm at matched total params; granular beat coarse, consistent with the fine-grained prior from Phase 58).

**Final answer to point #2:** hot-pool = no (measured, conceded); latency-trap = no (by construction: L3-resident backbone + bulk contiguous expert streaming); bandwidth = costed and cheap under ternary + granularity. The keystone budget becomes explicitly **two-pool**: resident (≤16 MB, reused, ~90-116 GB/s) + streamed (experts, ~28 GB/s, costed in bytes/token).

Point #3 (MTP acceptance) remains open, as stated.
