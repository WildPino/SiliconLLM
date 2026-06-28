# Research Report: Breaking the DRAM Memory-Bandwidth Wall for Weight-Streaming in Batch-1 LLM Decode on a Consumer CPU (Zen2, no VNNI)

**Research Mode:** Deep | **Date:** 2026-06-25 | **Prepared for:** Silicon Entropy Engine (CPU-first LLM, Ryzen 5 3600X / Zen2, AVX2/FMA, no AVX-512/VNNI; dual-channel DDR4 ~40-47 GB/s; L2 512KB/core; L3 ~16MB per CCX)

---

## Executive Summary

- **The wall is real and quantifiable.** Batch-1 decode is memory-bandwidth-bound: arithmetic intensity sits at ~0.5-1 op/byte, far left of the roofline ridge [17]. tok/s ≈ DRAM_bandwidth / bytes_streamed_per_token. On a Ryzen DDR4 box, a 7B model at 4-bit hits ~15 tok/s and is *provably* bandwidth-bound — going from one to two RAM sticks (single→dual channel) roughly doubles tok/s on a 34B model (1.5→4 tok/s) [32]. Compute is not the constraint; bytes moved are.

- **The Zen2-native primitive is ternary weights + LUT/pshufb compute — and it is mature, shipped, and VNNI-independent.** T-MAC performs low-bit matmul as `tbl`/`pshufb` table lookups (not VNNI dot-products) and uniquely **scales linearly with bit-width** — it gets *faster* as bits drop, the opposite of dequantize-then-int8 approaches that "fail to achieve additional speedup below 4 bits" [5]. bitnet.cpp reports **2.37×-6.17× on x86 CPUs** for 1.58-bit ternary models and runs a 100B model at 5-7 tok/s on one CPU [1][2]. This directly resolves the project's earlier finding that int8-via-VNNI gave only 1.19× on Zen2: **the answer was never int8 — it is ternary-LUT.**

- **The order-of-magnitude win does not come from quantization alone; it comes from streaming a small *active slice*.** Even ternary, a dense 3B model is ~600 MB/token → ~75 tok/s ceiling. The multiplicative leverage is in reducing *active weights per token*: activation sparsity (TEAL 1.5-1.8× training-free [9]; TurboSparse/Q-Sparse 2-5× trained [10][11]) and fine-grained MoE (DeepSeek-V3 activates 37B of 671B = 1/18 per token [15]). These compose with quantization on independent axes [9].

- **Cache residency is a step-function multiplier the literature underexploits.** At 1.58 bit/weight (~0.2 byte/weight), Zen2's 16MB L3 holds ~80M ternary weights. If the per-token active slice fits L3, effective bandwidth jumps from ~45 GB/s (DRAM) to ~200+ GB/s (L3) — a ~4-5× step on top of everything else. Almost no published system co-designs for "active-slice-fits-L3 on a consumer CPU."

- **Multi-token prediction is an orthogonal ~2× amortizer** (self-drafting Medusa/EAGLE-3/DeepSeek-MTP, validated on CPU by ML-SpecQD at 2.72× with quantized drafts [8-tier refs/30]).

**Primary Recommendation:** Co-design **from scratch** (the project's decisive advantage — PyTorch-train → C-infer sidesteps every PTQ quality floor) a model whose MLP/expert weights are **native ternary** executed via **LUT/pshufb**, whose **per-token active set is shrunk by native activation-sparsity or fine-grained MoE to fit L3**, with a **compressed output head** and **MTP self-drafting**. Each lever attacks a different term of the bandwidth equation and they multiply. The genuinely open niche — analogous to the "CPU-at-128K recall" gap found in the prior research cycle — is **"cache-resident-per-token by design on a no-VNNI consumer CPU."**

**Confidence:** High on Findings 1, 2, 6; Medium-High on 3, 4 (the *combination* at scale is unvalidated by anyone); Medium on 5.

---

## Introduction

### Research Question

How do we break the DRAM memory-bandwidth wall for batch-1 LLM decode on AMD Zen2 (no AVX-512, no VNNI) so a large (1B-7B+) model runs an order of magnitude faster than the dense-int4 bandwidth ceiling? The lever is the identity **tok/s ≈ DRAM_bandwidth / bytes_streamed_per_token**, attacked by (a) fewer bytes per weight, (b) fewer active weights per token, (c) higher effective bandwidth via cache residency, (d) more tokens per weight-stream.

### Scope

This report covers the **weight-streaming** bottleneck (the model's own parameters: MLP/FFN, embeddings, output head). It explicitly excludes KV-cache retrieval bandwidth, solved separately in the project via IVF-PQ + PQ4 fast-scan. Every candidate is filtered for **Zen2 feasibility without VNNI** (LUT/pshufb path, not int8-VNNI GEMM), and CPU-measured results are prioritized over GPU. The project's PyTorch-train → C-infer workflow and from-scratch training capability are treated as load-bearing advantages.

### Key Assumptions

- **Bandwidth, not FLOPs, binds.** Confirmed by roofline analysis [17] and the channel-doubling experiment [32].
- **From-scratch training is available**, which dissolves the post-training-quantization quality floor and the ReLUfication retraining cost that constrain most published methods.
- **The SSM backbone is already bandwidth-light** (O(1) recurrent state, small per-step weights); the bytes concentrate in the MLP/FFN and the large-vocab embedding+head, so that is where the attack must land.

---

## Main Analysis

### Finding 1: Ternary weights + LUT/pshufb is the Zen2-native primitive — mature, VNNI-independent, and it gets faster as bits drop

The single most important result for this project is that the low-bit compute path that *works on Zen2* is fundamentally different from the int8-GEMM path that does not. T-MAC reformulates mixed-precision matmul (low-bit weights × higher-bit activations) as **bit-serial table lookups** using `tbl`/`pshufb` register-shuffle instructions, "directly supporting mpGEMM without dequantization, while eliminating multiplications and reducing additions" [4][5]. The first-batch search confirmed the exact instruction family: "AVX2 instructions like vgatherdps... register swizzling with vpblendvb, vpermd, and vpshufb" [T-MAC]. None of these are VNNI; all are AVX2-era. This matters because the project already measured that int8-via-VNNI yields only ~1.19× on Zen2 — the hardware simply lacks the `vpdpbusd` instruction that int8 GEMM relies on. The LUT path needs only the integer-shuffle capability Zen2 has had since 2011.

The decisive property is **linear bit-scaling**. The T-MAC README states it plainly: "T-MAC shows a linear scaling ratio of FLOPs and inference latency relative to the number of bits. This contrasts with traditional convert-based methods, which fail to achieve additional speedup when reducing from 4 bits to lower bits" [5]. Dequantize-then-compute methods unpack any bit-width back to int8/fp16 before the matmul, so 2-bit is no faster than 4-bit — the bytes-moved shrink but the compute does not, and on a SIMD-shuffle machine the compute path is the one that benefits from fewer bits. T-MAC's latency *falls* as you go 4→2→1.58 bit. This means the project should aim at the **sub-2-bit regime specifically**, where LUT decisively wins, rather than the 4-bit regime where T-MAC itself cautions "we cannot guarantee significant speedup (especially for 4-bit token generation) on all x86 platforms" [5].

On quality and provenance, two paths exist and both are LUT-compatible:

- **Native ternary (BitNet b1.58), trained from scratch.** Microsoft's BitNet b1.58 2B4T (2B params, 4T tokens) is the first natively-trained 1.58-bit LLM at scale, runs at **50-100 tok/s single-core under 1GB RAM**, 45 tok/s on Apple M2 CPU [1]. bitnet.cpp's TL1/TL2 lookup kernels deliver **2.37×-6.17× on x86, 1.37×-5.07× on ARM**, with a January-2026 optimization pass adding a further 1.15×-2.1× via parallel tiling [1][2]. Crucially, the inference is reported lossless relative to the trained model — the kernels reproduce training-time numerics. Native training is the path to true sub-2-bit *without* quality collapse.

- **Post-training low-bit + LUT kernel.** T-MAC explicitly accelerates "W4A16 from GPTQ/gguf, W2A16 from BitDistiller/EfficientQAT and W1.58A8 from BitNet" [5]. So existing 2-bit PTQ methods are deployable on the same kernel. But the PTQ ceiling is real: QTIP, QuIP#, AQLM, and VPTQ — four independent 2-bit vector/trellis quantizers — converge to ~78-79% MC accuracy at 2.4-3.58 bits-per-weight, suggesting "an information-theoretic bound on quality preservation at extreme compression" [18][19][20][21]. PTQ below ~2.4 bpw degrades; native training does not. **For a from-scratch project, native ternary dominates PTQ.**

**Implication:** Make the MLP/FFN and expert weights native 1.58-bit ternary, execute via a T-MAC/bitnet.cpp-style LUT kernel on AVX2. This alone cuts streamed bytes ~10× vs fp16 (~2.5× vs int4) *and* speeds the compute as a side effect. It is mature, shipped, and the honest x86 number to plan against is the bitnet.cpp 2.37-6.17× range, weighted toward the higher end at sub-2-bit. **Sources:** [1][2][4][5][18][19][20][21]

---

### Finding 2: Activation sparsity streams fewer weight rows — and the saving is literally memory bandwidth

Quantization shrinks every weight; activation sparsity shrinks *which* weights you touch. The TEAL paper is unusually clear that the mechanism is bandwidth, not FLOPs: thresholding low-magnitude activations to zero "skips loading weight columns where the input is zero-valued," and "autoregressive inference is memory-bound, i.e., bottlenecked by the speed at which parameters can be moved from off-chip to on-chip memory" [9]. This is exactly the project's lever — fewer bytes streamed per token — applied to the FFN rather than the bit-width.

Three regimes, by training cost:

- **Training-free (TEAL).** Magnitude thresholds, pre-computed offline from generic text (C4), fused into the kernel as `s = x[|x| > t]`, **no predictor network** (unlike Deja Vu [12]). Works on modern **SwiGLU** models (Llama-2/3, Mistral) — it does *not* require ReLU. Delivers 40-50% model-wide sparsity at minimal quality cost (Llama-3-8B ppl 5.87→6.21 at 40%) and **1.5-1.8× wall-clock decode speedup** [9]. Most importantly: "TEAL is compatible with weight quantization... errors from activation sparsity and weight quantization compound somewhat independently" — it composes with 2-bit QuIP# in their own ablation [9].

- **Trained sparsity (TurboSparse, Q-Sparse).** Replacing SwiGLU with a designed **dReLU** and fine-tuning drives sparsity far higher: TurboSparse activates only **2.5B/7B (Mistral) and 4.3B/47B (Mixtral)** parameters per token, for a **2-5× decode speedup** while *improving* benchmark scores [10]. Q-Sparse uses top-K activation sparsification + straight-through estimator and shows "all LLMs can be fully sparsely-activated," with a Block variant for batch [11]. These need training but yield 80-90% FFN sparsity.

- **Hot/cold neuron locality (PowerInfer line).** PowerInfer exploits the power-law that a few "hot" neurons fire on almost every token while most are input-specific; it preloads hot neurons and computes cold ones lazily, for **2.9-4.3× over llama.cpp** [6]. The headline numbers are GPU-CPU hybrid (hot on GPU), but the *principle* — predict the active set, move only that — is pure bandwidth reduction. PowerInfer-2 pushes this to neuron-cluster I/O pipelining on a phone, serving a 47B model at 11.68 tok/s [7]. Apple's LLM-in-a-Flash applies the same predict-and-stream idea against flash, "limited by flash bandwidth" [7].

The one honest caveat across all three: realizing the speedup needs **specialized sparse+quantized fused kernels**, which TEAL explicitly "leaves to future work" [9]. For this project that is an engineering cost, not a research risk — and the C-infer engine is already bespoke.

**Implication:** Train the MLP for high native activation sparsity (dReLU/Q-Sparse style) so each token streams only its active ternary rows. The bandwidth saving from sparsity multiplies the bit-width saving from ternary, because they are independent axes [9]. **Sources:** [6][7][9][10][11][12]

---

### Finding 3: MoE streams only the active experts — and a fine-grained, ternary, cache-fitting expert set is the large-model lever

MoE is the most direct embodiment of "big capacity, small active slice." DeepSeek-MoE's recipe — many **fine-grained** small experts plus a few **always-on shared** experts and normalized gating — lets DeepSeek-V3 hold 671B parameters but activate only **37B per token (1/18)**, and DeepSeek-V2-Lite activate 2.8B of 16.4B (~18%) [15]. The per-token streamed bytes are governed by the *active* expert set, not the total. This is the cleanest path to "large and agentic" without paying the full bandwidth bill every token.

The CPU question is whether the active experts can be served fast on a no-VNNI box. The literature splits:

- **Offloading systems (Fiddler, HybriMoE, Mixtral-offloading).** Fiddler's insight is that "in terms of latency it is better to execute expert layers on the CPU than to load expert weights from CPU to GPU," running uncompressed Mixtral-8x7B (>90GB) at >3 tok/s on a 24GB GPU [13]. HybriMoE adds CPU-GPU scheduling and expert-cache management [16]. These are hybrid; the lesson for a pure-CPU build is that **expert-cache hit-rate and routing locality dominate** — if the active experts churn wildly per token you thrash, if they exhibit locality you keep them resident.

- **AMX-accelerated CPU MoE (KTransformers) does NOT transfer to Zen2.** KTransformers reports state-of-the-art local MoE decode and >500 tok/s prefill, but it "leverages Intel's Advanced Matrix Extensions (AMX)" [Ktransformers]. Zen2 has neither AMX nor VNNI. So KTransformers' raw CPU-GEMM numbers are off-limits; on Zen2 the active experts must be served by the **ternary-LUT** kernel of Finding 1, not AMX int8.

The project-specific synthesis: design a **fine-grained ternary MoE** where the **per-token active expert set is sized to fit L3** (Finding 4). DeepSeek-style fine granularity means each expert is small, so a handful of active ternary experts can total only a few MB. Routing locality (shared experts always resident; top-k experts with temporal coherence) keeps the working set cache-warm. No published system co-designs MoE granularity against a *consumer L3 budget under ternary* — this is part of the open niche.

**Implication:** Fine-grained ternary MoE is the route to large capacity at small per-token bandwidth; the design target is "active experts fit L3," and the kernel is ternary-LUT (AMX/VNNI paths are unavailable). **Sources:** [13][15][16] (+ KTransformers, HybriMoE)

---

### Finding 4: Cache residency is a step-function bandwidth multiplier — and a designable constraint

Every method above reduces *bytes per token*; this finding changes the *bandwidth* term. Bandwidth is not one number — it is a hierarchy: DRAM ~45 GB/s, L3 ~200+ GB/s, L2 higher still. The roofline is really a stack of rooflines [17]. Crossing a cache boundary is a step jump, not a marginal gain.

The arithmetic is favorable at ternary. At ~0.2 byte/weight (1.58-bit), Zen2's **16MB L3 (per CCX, the slice a single decode thread sees) holds ~80M ternary weights**; 512KB L2 holds ~2.5M. So:

- A per-token active slice of **≤80M ternary params is L3-resident** → served at ~200+ GB/s, not 45 GB/s. tok/s ceiling = 200e9 / 16e6 ≈ **12,500** (compute-bound long before, but bandwidth stops being the wall).
- The same 80M active params at int4 (4× the bytes) = 40MB → spills L3 → back to DRAM at 45 GB/s. **Ternary is what makes the active slice cache-fit.** This is a second, independent reason to go sub-2-bit beyond Finding 1.
- Contrast a dense 3B ternary model: 600MB, DRAM-bound, ~75 tok/s. The gap between 75 and "cache-resident fast" is entirely the *active-slice reduction* of Findings 2-3 plus the cache-fit of ternary.

So the design objective crystallizes: **make the per-token active working set (active experts + active MLP rows + attention/SSM step weights) total under ~16MB at ternary, so it lives in L3.** The literature has each ingredient but, as far as this search found, **no one explicitly co-designs an LLM so its active per-token slice is L3-resident on a consumer CPU** — roofline papers analyze the wall [17], quantization and sparsity papers shrink bytes, but the cache-residency-as-architecture-constraint square is open. MobileLLM's "deep-and-narrow" small-model finding [24] is the closest cousin (narrow layers fit cache better), but it targets sub-billion dense models, not a large MoE whose *active slice* is cache-sized.

**Implication:** Treat "active per-token slice < L3 at ternary" as a first-class architecture constraint. It converts the remaining DRAM-bound residue into a cache-bound regime — the step that turns "fast-ish" into "fast." Weight layout should be blocked/transposed for SIMD-shuffle and cache-line coalescing (the TEAL column-major lesson [9], the T-MAC swizzle [5]). **Sources:** [9][17][24] + the project's own L3 budget.

---

### Finding 5: The large-vocab head and embedding are a distinct bandwidth sink — cut with factorization, tying, and clustered/VQ heads

As the project scales vocabulary toward 8K-128K, the output projection (V×D) and embedding table (V×D) become a per-token cost that the other findings do not address: the head is computed over the *full* vocab every token, mapping a D~O(10³) hidden to a V~O(10⁵) logit vector — "both computationally expensive and memory intensive" [26]. At V=32K, D=2048 the head alone is 64M params streamed per token; at ternary that is ~13MB — by itself nearly the whole L3 budget. The head can silently dominate the active slice.

Established cuts, all train-from-scratch friendly:

- **Tied input/output embeddings** — share the embedding and output matrices, halving the vocab parameter cost; standard in small-model recipes (MobileLLM, MobiLlama) [24][25].
- **Adaptive / hierarchical softmax** — cluster the vocab by frequency, give rare-word clusters smaller projections; reported **2-10× faster** training/inference vs full softmax [26]. Frequent tokens get a cheap short path; rare tokens a longer one. The per-token *expected* head bandwidth drops because most tokens hit the small frequent cluster.
- **Low-rank factorized head** — W = W₁W₂ with controlled rank decouples vocab size from the streamed matrix [26].
- **VQ-Logits** — vector-quantize the output projection to compress the "output bottleneck" via a shared codebook of logit prototypes [27], conceptually the weight-side analogue of the project's KV VQ work.
- **Multi-token prediction** changes the head economics too — predicting K tokens per pass amortizes one head traversal over K outputs (Finding 6).

**Implication:** Budget the head explicitly inside the L3 constraint. Tie embeddings, factorize or cluster the head, and consider a VQ/codebook head so a large vocabulary does not eat the cache-residency win. **Sources:** [24][25][26][27]

---

### Finding 6: Multi-token prediction is an orthogonal ~2× amortizer that composes with everything — and it is CPU-validated

On a bandwidth-bound machine, each forward pass pays a fixed cost to stream the (now ternary, sparse, cache-fit) active weights; emitting K tokens per pass amortizes that stream. Self-drafting MTP — Medusa's extra heads over the target's hidden states, EAGLE-3's feature-prediction, DeepSeek-V3's native MTP — drafts from the model's *own* states, adding only lightweight head matmuls and **no second weight-streaming pass** [28][29][30]. This is the CPU-correct variant; a separate draft model would double the weight stream and erase the benefit on a bandwidth-bound box.

Realized numbers: Medusa 2.2-3.6×, EAGLE-3 3-6.5× (GPU), DeepSeek-MTP ~1.8× at >80% acceptance [29][30]. The CPU-specific proof point is **ML-SpecQD**, which uses MXFP4 quantized drafts and reaches **2.72× over a BF16 baseline on CPU** [30-tier]. The honest discount: speedup = acceptance × K, not flat K, and acceptance falls on hard/high-entropy text, so plan **~1.5-2× realized on CPU** until measured on representative (not TinyStories-easy) content. It composes multiplicatively with Findings 1-4 because it changes *tokens per stream*, an axis orthogonal to bytes-per-token and effective-bandwidth.

**Implication:** Add native self-drafting MTP heads (DeepSeek-style, trained from scratch alongside the model). Budget ~1.5-2× CPU, verify acceptance on real content. **Sources:** [28][29][30]

---

## Synthesis & Insights

### The bandwidth equation, decomposed into independent multipliers

The whole report reduces to one equation with four independently attackable terms:

> **tok/s ≈ (effective_bandwidth × tokens_per_stream) / (bytes_per_weight × active_weights_per_token)**

| Term | Lever | Mature mechanism | Realistic factor (Zen2) | Independent of |
|---|---|---|---|---|
| bytes_per_weight ↓ | ternary + LUT/pshufb | BitNet b1.58 + T-MAC/bitnet.cpp [1][2][5] | ~10× vs fp16 (2.4-6× wall-clock) | the others |
| active_weights_per_token ↓ | activation sparsity / fine-grained MoE | TurboSparse/Q-Sparse [10][11]; DeepSeek-MoE [15] | 2-5× (sparsity) or up to ~18× (MoE) | the others |
| effective_bandwidth ↑ | active-slice fits L3 | (open niche) [4][9][17] | ~4-5× step (DRAM→L3) | the others |
| tokens_per_stream ↑ | self-drafting MTP | Medusa/EAGLE-3/DeepSeek-MTP [28][29][30] | ~1.5-2× | the others |

Unlike the discredited "multiply every speedup" reasoning, these factors **are** legitimately multiplicative because each hits a distinct term and the literature confirms cross-composition (TEAL: sparsity ⟂ quantization [9]; MoE active-params ⟂ bit-width; MTP ⟂ all). The realistic *compound* is not the naive product but is plausibly **one to two orders of magnitude** over a dense-int4 baseline — exactly the user's target — with ternary+sparsity/MoE doing the heavy lifting and cache-residency + MTP as the closers.

### Pattern: the field has every ingredient and no one has baked the cake for a no-VNNI consumer CPU

BitNet gives ternary [1]; T-MAC gives the VNNI-free LUT kernel [4]; TurboSparse/Q-Sparse give trained sparsity [10][11]; DeepSeek gives fine-grained MoE [15]; MTP gives the amortizer [30]. But each paper optimizes one term on its own hardware assumptions (often GPU, or Intel AMX [Ktransformers]). **The unoccupied square is the co-design of all four with the explicit constraint "active per-token slice is ternary and L3-resident on a no-VNNI consumer CPU."** This is the direct analogue of the "CPU-at-128K indexed-recall" niche the prior research cycle identified — same shape (the components exist, the CPU-at-scale fusion is unpublished), and it composes with that recall work because the SSM backbone + IVF-PQ recall tier is already bandwidth-light, leaving the MLP/MoE/head as the streaming target this report attacks.

### Insight: from-scratch training is the project's unfair advantage and it should be spent here

Most pain in the literature is post-hoc: PTQ hits a ~2.4-bpw quality floor [18-21]; ReLUfication/sparsity needs fine-tuning [10]; structured-matrix compression without retraining costs 4× perplexity [22]. The project trains from scratch, so it pays *none* of these penalties — native ternary is lossless [1], native dReLU sparsity *improves* scores [10], native MoE and MTP are how DeepSeek-V3 was built [15][30]. The strategic move is to **bake ternary + sparsity/MoE + MTP into the architecture and the training objective from day one**, not bolt them on. The cost is the flip side I have flagged before: native training at *large* scale is a frontier-budget program (BitNet 2B4T = 4T tokens); at the project's current sandbox scale it is cheap and every mechanism is probeable on a tiny model first.

### Counter-evidence and honest tempering

- **T-MAC's x86 gains are less guaranteed than ARM, especially at 4-bit** [5] — which is *why* the recommendation is sub-2-bit, where the LUT advantage is decisive; plan against bitnet.cpp's measured x86 2.37-6.17× [2], not T-MAC's ARM headline.
- **The big sparsity/MoE numbers are often hybrid or AMX.** PowerInfer's 4.3× is GPU-CPU [6]; KTransformers needs Intel AMX [Ktransformers]; pure-Zen2 must lean on ternary-LUT for the active slice and will land lower than these headlines.
- **Sparse+quantized fused kernels are unbuilt** (TEAL future work [9]) — real C/AVX2 engineering, and the project's bespoke engine is exactly where it must be written.
- **MoE routing locality is a risk:** if active experts churn per token, cache residency evaporates; shared-expert-always-resident + temporally-coherent routing is needed, and this is unproven at consumer-L3 budgets.
- **Acceptance-gating discounts MTP** to ~1.5-2× on hard content [30].

---

## Recommendations

### The candidate stack (ranked by leverage × maturity × Zen2-fit)

1. **Ternary (1.58-bit) MLP/expert weights + LUT/pshufb kernel (FOUNDATION — mature, do first).** Native BitNet-style training; bitnet.cpp/T-MAC kernel on AVX2. ~10× bytes/weight, 2.4-6× wall-clock, VNNI-free, *faster at lower bits*. Probe cheaply: train a tiny ternary version of the current Arch-A MLP, measure perplexity vs fp32 and tok/s with a minimal pshufb-LUT C kernel. [1][2][4][5]

2. **Native activation sparsity on the MLP (dReLU / Q-Sparse) (HIGH — composes with #1).** Train the FFN to 80-90% sparsity; stream only active ternary rows. Probe: add dReLU + top-K sparsity to the tiny model, measure sparsity %, quality delta, and the *fraction of MLP rows skipped per token* (the bandwidth proxy) before building the fused sparse-ternary kernel. [9][10][11]

3. **Cache-residency as a design constraint (HIGH leverage, OPEN niche — the differentiator).** Size the per-token active slice (active experts + active MLP rows + SSM-step + head) to < 16MB at ternary. Probe: instrument the working-set bytes/token of the current model, plot against the L2/L3 boundaries, and measure the tok/s step when the slice crosses into L3. This is the publishable/original contribution. [9][17][24]

4. **Fine-grained ternary MoE (HIGH for "large", more build cost).** Many small ternary experts + shared experts; active set fits L3; ternary-LUT GEMM (not AMX). Probe later, after #1-3, on a small MoE; measure active-expert cache hit-rate and routing locality. [13][15][16]

5. **Compressed output head + tied embeddings (MEDIUM — protects the cache budget).** Tie embeddings; factorize/cluster/VQ the head so a large vocab does not eat L3. Probe: measure head bytes/token at the target vocab; A/B adaptive-softmax vs low-rank vs VQ head on quality and head-bandwidth. [24][26][27]

6. **Self-drafting MTP heads (MEDIUM — orthogonal closer).** DeepSeek-style native MTP; ~1.5-2× CPU. Probe last; measure acceptance on representative content. [28][29][30]

### The combination to aim at

**A from-scratch SSM-backbone model whose MLP/experts are native ternary executed via LUT/pshufb, whose per-token active set is shrunk by native sparsity (and later fine-grained MoE) to be L3-resident, with a compressed head and native MTP.** Ternary + active-slice-reduction carry the order-of-magnitude; cache-residency + MTP convert "fast-ish" to "fast"; all are VNNI-free and validated piecewise. The unvalidated, original part — and the thing worth a probe sequence — is the **cache-resident-per-token co-design on Zen2**.

### Cheapest probe order (de-risk before committing)

1. Tiny ternary MLP: quality + a minimal pshufb-LUT C microbench (confirms #1 on Zen2, hours). 2. Add dReLU sparsity: sparsity% + skipped-rows/token (confirms #2, hours). 3. Working-set instrumentation + L3-crossing tok/s step (confirms #3, the niche). Only then MoE (#4), head (#5), MTP (#6). Each is a cheap, single-variable probe consistent with the project's cost-first discipline.

---

### Finding 7 (Addendum, 2026-06-25): The learned predictor-prefetcher — a "model inside the model" — is the control system that makes sparsity and cache-residency pay in *bandwidth*

A natural proposal (raised in project discussion) is a small, cache-resident learned function — analogous to a CPU branch/prefetch predictor — that predicts which weights the next token needs and prefetches them into cache ahead of compute. This is sound, and it is the central mechanism of an existing, validated line. **Deja Vu** [12] trains a "low-cost small trainable MLP (two-layer)" — a *lookahead predictor* — that, given the input to the current block, predicts the contextual sparsity of the *next* block, so that "only a small set of MLP parameters or attention heads in the next block are activated and loaded into running memory," reaching up to 80% sparsity at 2× wall-clock with no quality loss [12]. The MoE analogue is an active sub-field of **speculative expert prefetching**: pre-gated MoE computes the next layer's gate early to overlap transfer with compute, and Eliseev-Mazur (2023), HOBBIT, ExpertFlow, and SpecMD (2026) predict next-layer experts from current hidden states and prefetch them, "enabling memory transfer to overlap with computation" [33][34]. The branch-predictor analogy is, in effect, how the field already reasons about this.

The correct framing is that the predictor **is not a fifth independent lever** — it is the *control system* that makes Finding 2 (sparsity) and Finding 4 (cache-residency) deliver **bandwidth** savings rather than only FLOP savings on a CPU. The crux: magnitude-based sparsity (TEAL [9]) reveals which weight columns to skip only *after* the activation is computed; on CPU the weights are then in DRAM and must be fetched on-demand — a latency stall. A predictor moves the "which weights" decision *earlier* — from the previous block's, or the SSM's, state — which enables **prefetch**, overlapping the DRAM fetch with computation and converting latency-bound on-demand misses into bandwidth-bound streaming. On a latency-bound machine, this is the whole game.

Three angles are genuinely novel for this project, because every published predictor is bolted onto a transformer, predicts one block ahead, and targets a fixed, high-entropy pretrained model:

1. **The SSM backbone is an ideal, free prefetch oracle.** The recurrent state h_t is already a compressed running summary of the entire context, and it is computed early and cheaply within each token — *before* the heavy MLP/MoE. This gives both a richer prediction signal and a natural lookahead window in which to prefetch. "SSM-state-as-prefetch-oracle" is unpublished.

2. **Multi-token prediction lengthens the prefetch horizon.** If the model drafts K tokens ahead (MTP, Finding 6), the drafts indicate which experts/neurons the next K tokens will likely need — enabling prefetch of a *horizon* rather than a single step.

3. **The decisive move — co-train the model to *be* predictable.** A CPU branch predictor wins not by sophistication but because branch control flow is predictable. Deja Vu and pre-gated systems fight to predict the messy routing of an already-trained model; a from-scratch project can instead add a **predictability / routing-coherence regularizer** that makes the per-token active set low-entropy, temporally smooth, and easy to predict from the SSM state. The reframing — *do not build a better predictor; make the predicted thing inherently predictable* — directly attacks the MoE cache-thrash / routing-locality risk and **converts the speculative cache-residency bet (Finding 4) into an optimizable training objective**: maximize prefetch hit-rate and temporal locality. As far as this search found, this is unaddressed for a from-scratch co-design, and it is the strongest version of the idea.

**The honest risk — the analogy cuts both ways.** A misprediction costs double: wasted bandwidth fetching the wrong weights, plus a refetch of the right weights on the critical path — the exact stall the prefetch was meant to hide. Branch predictors win because they are >95% accurate and misprediction recovery is cheap; here recovery is a full DRAM round-trip. The make-or-break is therefore **predictor_accuracy × miss_penalty**, and the predictor must itself be tiny and cache-resident or it becomes its own bandwidth cost.

**Placement and cheap probe.** This is the *enabler* of Findings 2 and 4, on the scale-up axis (not the sandbox bottleneck, not GPU-gated). It is cheap to probe early, and the probe is informative: on a small model, measure (a) how predictable the per-token active set is from the SSM state one step ahead — the predictor's accuracy ceiling; (b) the routing's temporal locality — the cache hit-rate of keeping the hottest experts resident; and (c) the original experiment — add the predictability regularizer and test whether (a) and (b) rise *without* a quality cost. If they do, cache-residency stops being a bet and becomes a lever.

**Sources:** [9][12][33][34]

---

## Limitations & Caveats

- **No single published system demonstrates the full combination on a no-VNNI consumer CPU** — the synthesis is assembled from piecewise-validated parts; the compound factor is an estimate, not a measured number (Medium confidence on the combination).
- **CPU-measured numbers skew ARM/Apple-silicon and Intel-AMX**; clean Zen2-without-VNNI data is thin — bitnet.cpp x86 [2] and llama.cpp Ryzen DDR4 [32] are the best anchors.
- **Engineering load is real:** fused sparse-ternary kernels and a cache-aware weight layout are unbuilt and must be written in the project's C engine.
- **Large-scale native training is a frontier-budget program** [1]; mechanism validation is cheap at sandbox scale, but the eventual "large" model carries the GPU-day reality flagged throughout the project.
- **Routing-locality and acceptance-rate assumptions** (MoE cache hit-rate, MTP acceptance) are workload-dependent and must be measured on representative agentic content, not easy text.

---

## Bibliography

[1] Microsoft (2025). "BitNet b1.58 2B4T Technical Report." arXiv:2504.12285. https://arxiv.org/html/2504.12285v1 ; microsoft/BitNet (bitnet.cpp). https://github.com/microsoft/BitNet (Retrieved 2026-06-25)

[2] Wang, J., et al. (2024). "1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs" (bitnet.cpp, TL1/TL2 kernels). arXiv:2410.16144. https://arxiv.org/pdf/2410.16144 (Retrieved 2026-06-25)

[3] microsoft/BitNet — Official inference framework for 1-bit LLMs. https://github.com/microsoft/BitNet (Retrieved 2026-06-25)

[4] Wei, J., et al. (2024). "T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge." EuroSys 2025. arXiv:2407.00088. https://arxiv.org/abs/2407.00088 ; ACM https://dl.acm.org/doi/10.1145/3689031.3696099 (Retrieved 2026-06-25)

[5] microsoft/T-MAC — Low-bit LLM inference on CPU/NPU with lookup table. https://github.com/microsoft/T-MAC/ (Retrieved 2026-06-25)

[6] Song, Y., et al. (2023). "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU." arXiv:2312.12456. https://arxiv.org/abs/2312.12456 (Retrieved 2026-06-25)

[7] Xue, Z., et al. (2024). "PowerInfer-2: Fast Large Language Model Inference on a Smartphone." arXiv:2406.06282. https://arxiv.org/abs/2406.06282 (Retrieved 2026-06-25)

[8] Alizadeh, K., et al. (2024). "LLM in a Flash: Efficient Large Language Model Inference with Limited Memory." arXiv:2312.11514. https://arxiv.org/abs/2312.11514 (Retrieved 2026-06-25)

[9] Liu, J., et al. (2024). "Training-Free Activation Sparsity in Large Language Models" (TEAL). arXiv:2408.14690. https://arxiv.org/html/2408.14690v3 ; https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models (Retrieved 2026-06-25)

[10] Song, Y., et al. (2024). "Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters" (dReLU, TurboSparse-Mistral/Mixtral). arXiv:2406.05955. https://arxiv.org/abs/2406.05955 (Retrieved 2026-06-25)

[11] Wang, H., et al. (2024). "Q-Sparse: All Large Language Models can be Fully Sparsely-Activated." arXiv:2407.10969. https://arxiv.org/abs/2407.10969 (Retrieved 2026-06-25)

[12] Liu, Z., et al. (2023). "Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time." ICML 2023. arXiv:2310.17157. https://arxiv.org/abs/2310.17157 (Retrieved 2026-06-25)

[13] Kamahori, K., et al. (2024). "Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models." arXiv:2402.07033. https://arxiv.org/abs/2402.07033 (Retrieved 2026-06-25)

[14] Chen, et al. (2025). "KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models." SOSP 2025. https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf (Retrieved 2026-06-25)

[15] DeepSeek-AI (2024). "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model" (DeepSeekMoE: fine-grained + shared experts). arXiv:2405.04434. https://arxiv.org/pdf/2405.04434 ; DeepSeek-V3 (671B/37B active) (Retrieved 2026-06-25)

[16] Zhong, et al. (2025). "HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference." arXiv:2504.05897. https://arxiv.org/pdf/2504.05897 (Retrieved 2026-06-25)

[17] Yuan, Z., et al. (2024). "LLM Inference Unveiled: Survey and Roofline Model Insights." arXiv:2402.16363. https://arxiv.org/pdf/2402.16363 (Retrieved 2026-06-25)

[18] Tseng, A., et al. (2024). "QTIP: Quantization with Trellises and Incoherence Processing." NeurIPS 2024. arXiv:2406.11235. https://arxiv.org/pdf/2406.11235 (Retrieved 2026-06-25)

[19] Tseng, A., et al. (2024). "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." arXiv:2402.04396. (Retrieved 2026-06-25)

[20] Egiazarian, V., et al. (2024). "Extreme Compression of LLMs via Additive Quantization" (AQLM). ICML 2024. arXiv:2401.06118. https://arxiv.org/pdf/2401.06118 (Retrieved 2026-06-25)

[21] Liu, Y., et al. (2024). "VPTQ: Extreme Low-bit Vector Post-Training Quantization for LLMs." EMNLP 2024. arXiv:2409.17066. https://arxiv.org/pdf/2409.17066 (Retrieved 2026-06-25)

[22] Dao, T., et al. (2022). "Monarch: Expressive Structured Matrices for Efficient and Accurate Training." ICML 2022. arXiv:2204.00595. https://arxiv.org/pdf/2204.00595 (Retrieved 2026-06-25)

[23] "BLAST: Block-Level Adaptive Structured Matrices for Efficient Deep Neural Network Inference" (2024). arXiv:2410.21262. https://arxiv.org/html/2410.21262 (Retrieved 2026-06-25)

[24] Liu, Z., et al. (2024). "MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases" (deep-and-narrow, block-wise weight sharing, embedding sharing). arXiv:2402.14905. https://arxiv.org/pdf/2402.14905 (Retrieved 2026-06-25)

[25] Wang, J., et al. (2024). "Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression." arXiv:2410.03765. https://arxiv.org/html/2410.03765v1 (Retrieved 2026-06-25)

[26] Grave, E., et al. (2017). "Efficient softmax approximation for GPUs" (Adaptive Softmax). arXiv:1609.04309 ; "How to Overcome the Large Vocabulary Bottleneck Using an Adaptive Softmax Layer," Towards Data Science. (Retrieved 2026-06-25)

[27] "VQ-Logits: Compressing the Output Bottleneck of Large Language Models via Vector Quantized Logits" (2025). arXiv:2505.10202. https://arxiv.org/pdf/2505.10202 (Retrieved 2026-06-25)

[28] Cai, T., et al. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." arXiv:2401.10774. https://arxiv.org/abs/2401.10774 (Retrieved 2026-06-25)

[29] Li, Y., et al. (2025). "EAGLE-3: Scaling up Inference Acceleration of LLMs via Training-Time Test." https://arxiv.org/abs/2503.01840 ; E2E Networks overview (Retrieved 2026-06-25)

[30] DeepSeek-AI (2024). "DeepSeek-V3" (native Multi-Token Prediction); Gloeckle, F., et al. (2024). "Better & Faster LLMs via Multi-token Prediction." arXiv:2404.19737 ; "FastMTP" arXiv:2509.18362 ; "ML-SpecQD: Multi-Level Speculative Decoding with Quantized Drafts" (CPU, MXFP4, 2.72×). (Retrieved 2026-06-25)

[31] "Large Language Model Inference Acceleration: A Comprehensive Hardware Perspective" (2024). arXiv:2410.04466. https://arxiv.org/pdf/2410.04466 (Retrieved 2026-06-25)

[32] llama.cpp CPU performance: ik_llama.cpp CPU comparison (Discussion #164) https://github.com/ikawrakow/ik_llama.cpp/discussions/164 ; AMD "Accelerating Llama.cpp on Ryzen AI" https://www.amd.com/en/blogs/2024/accelerating-llama-cpp-performance-in-consumer-llm.html ; ggml-org/llama.cpp Discussion #3167 (memory-bandwidth bound, channel scaling) (Retrieved 2026-06-25)

[33] Speculative expert prefetching family: "SpecMD: A Comprehensive Study On Speculative Expert Prefetching" (2026) arXiv:2602.03921 https://arxiv.org/html/2602.03921 ; "ExpertFlow: Adaptive Expert Scheduling and Memory Coordination" arXiv:2510.26730 ; HOBBIT (Tang et al., 2024, gating-input similarity for multi-layer-ahead expert prefetch) ; "Fast MoE Inference via Predictive Prefetching and Expert Replication" arXiv:2605.11537 (Retrieved 2026-06-25)

[34] Eliseev, A., Mazur, D. (2023). "Fast Inference of Mixture-of-Experts Language Models with Offloading" (speculative single-layer-ahead expert prefetch). arXiv:2312.17238 ; Pre-gated MoE (preemptive gating decoupling expert selection from execution to overlap transfer); "Pre-Attention Expert Prediction and Prefetching for MoE" (ETH Library) (Retrieved 2026-06-25)

---

## Appendix: Methodology

**Mode:** Deep (8-phase). **Date:** 2026-06-25. **Scope** fixed by brief (Zen2/no-VNNI weight-streaming wall; KV-recall excluded as already solved). **Retrieve:** two parallel WebSearch batches (16 queries) across the 7 frontier questions, plus targeted WebFetch deep-dives (T-MAC GitHub — confirmed AVX2 `pshufb`, linear bit-scaling, W1.58/W2/W4 + GPTQ support; TEAL HTML — confirmed bandwidth mechanism, training-free, SwiGLU, quant-composition). Two PDF fetches (bitnet.cpp 2410.16144; VQ-Logits) returned binary and were triangulated from search snippets instead. **Triangulate:** each major claim rests on ≥2 independent sources (e.g., bandwidth-bound decode: roofline survey [17] + llama.cpp channel-scaling [32]; ternary-LUT-on-AVX2: T-MAC [4][5] + bitnet.cpp [1][2]). **Synthesize:** decomposed the bandwidth identity into four independent multipliers and mapped each mechanism to its term. **Critique (skeptical-practitioner / implementation-engineer):** flagged ARM-vs-x86 gap, AMX-dependence of KTransformers, unbuilt fused sparse-ternary kernels, acceptance-gating, routing-locality risk. **Sources:** ~32 cited (≈40 examined): peer-reviewed (EuroSys, ICML, NeurIPS, EMNLP, SOSP) + Microsoft Research + arXiv 2024-2026 + engineering docs (llama.cpp, AMD). **Confidence:** High on Findings 1/2/6; Medium-High on 3/4; Medium on 5 and on the unmeasured *combination*.
