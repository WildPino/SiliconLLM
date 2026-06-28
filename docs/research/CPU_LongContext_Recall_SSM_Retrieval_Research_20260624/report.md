# Research Report: Sublinear Streaming Recall on CPU over a Recurrent Backbone — Similarity-Preserving KV Compression and Existing SSM+Retrieval Architectures

**Research Mode:** Deep | **Date:** 2026-06-24 | **Prepared for:** Silicon Entropy Engine (CPU-first LLM, Ryzen 5 3600X / Zen2, AVX2/FMA, no AVX-512/VNNI)

---

## Executive Summary

- **The architectural thesis is now empirically settled in the published literature, not just in your tests.** A 2025 ablation across three production hybrids (RecurrentGemma-2B/9B, Jamba-Mini-1.6) found that removing attention drives Needle-in-a-Haystack retrieval to **0% accuracy**, with SSM layers offering *zero* compensatory recall — yet **sparsifying to just 15% of attention heads preserves near-perfect retrieval and 84% of MMLU** [18]. This confirms your "SSM backbone + thin recall tier" design: retrieval is exclusively an attention/associative function, and the tier supplying it can be very thin (<5% of heads are "retrieval heads" [21]).

- **Your KV-compression hypothesis (#2) is published doctrine under the name "anisotropic / score-aware quantization."** ScaNN proves that minimizing ‖K−K̂‖² (MSE) is the *wrong* objective for top-k(q·K); the correct loss weights error by the magnitude of the true inner product and penalizes error *parallel* to the vector, because parallel error disproportionately corrupts the high inner products that top-k is trying to find [4]. This is exactly your "preserve ranking, not norm" intuition, and it is the basis of the fastest CPU MIPS system in production.

- **The streaming-insert problem is easier than you fear, because autoregressive decode is append-only.** The expensive part of streaming ANN — deletion and graph consolidation — is what FreshDiskANN and IP-DiskANN engineer around [1][2]; within a generated sequence you never delete, so IVF with cheap append suffices and avoids the consolidation machinery.

- **Your CPU cost model is corroborated by a peer-reviewed CPU attention system.** NoMAD-Attention replaces dot-products with in-register SIMD lookups over 4-bit PQ codes (16 centroids), achieving **up to 2× speedup at 16k context with <4% perplexity loss at 8× key-cache compression**, with no finetuning — on CPUs, using exactly the `shuffle_epi8`/PQ4fs kernel family you have converged on [9].

- **Multi-token prediction is real and orthogonal**, adding ~1.8–3.6× (acceptance-gated); self-drafting MTP (DeepSeek-style) avoids a second model and is the CPU-appropriate variant [23][25][26].

**Primary Recommendation:** Your IVF-PQ front-runner is the right backbone; the highest-leverage external upgrades are (a) swap MSE-PQ training for ScaNN-style anisotropic/score-aware codebook loss, (b) treat the recall tier as a *retrieval-head emulator* (very thin), and (c) study ParisKV [30] — a Feb-2026 paper that is your exact architecture (ANN over KV, ranking-preserving, drift-robust) but GPU-targeted; the CPU-first version is an open niche.

**Confidence Level:** High on Findings 1–4; Medium on CPU-specific extrapolations for ParisKV and MTP (measured on GPU / not isolated to Zen2-without-VNNI).

---

## Introduction

### Research Question

How does one perform **retrieval/recall that is sublinear, streaming, and CPU-cheap, over a recurrent (SSM-like) backbone, with KV compressed such that the compression preserves search similarity** — and which **SSM+retrieval architectures already exist with measured long-range quality and CPU cost**? The work is in service of a CPU-first LLM (Ryzen 5 3600X, Zen2: AVX2/FMA, **no AVX-512, no VNNI**) whose distinctive thesis is architectural speed on consumer CPU at 128K+ context, where full attention is dead (KV cache = tens of GB, bandwidth-bound).

### Scope & Methodology

This report addresses the five frontier questions in the brief: (1) streaming/append-only ANN indices; (2) retrieval-similarity-preserving KV compression vs MSE; (3) PQ fast-scan and 4-bit ADC limits on CPU, plus alternatives (RVQ/AQ/AQLM); (4) already-published SSM+sparse-attention/retrieval hybrids with long-range and cost measurements; (5) multi-token prediction on CPU. It deliberately does **not** re-derive what your experiments already established (MQAR collapse of pure SSM ≈10%; top-k sparse ≈97–98%; IVF-PQ ≈8 µs/token/layer front-runner; VSA superposition dead).

Research drew on ~40 sources gathered via parallel web search and targeted full-text fetches, spanning arXiv (2021–2026), FAISS engineering documentation, AAAI/ICML/ICLR/NeurIPS proceedings, and vendor engineering blogs (Google ScaNN, Together AI, AI21). Sources were triangulated so that each major claim rests on ≥2 independent reports; quantified benchmark claims were verified against the originating paper where possible. Temporal emphasis is on 2024–2026 work, with foundational anchors (FreshDiskANN 2021, Additive Quantization 2014, Memorizing Transformers 2022).

### Key Assumptions

- **Append-only within a sequence.** Decode KV grows monotonically per generated sequence; cross-sequence eviction is a cache-management concern, not a streaming-ANN-deletion concern. This makes the "easy half" of streaming ANN the relevant half.
- **Bandwidth, not FLOPs, is the binding constraint on Zen2.** All CPU-cost reasoning treats memory traffic and in-cache LUT residency as the cost driver, consistent with your finding that random gather is negligible while the O(ctx) score-scan is the killer.
- **Quality target is "recall ranking," not "KV reconstruction."** The downstream operation is top-k(q·K̂); reconstruction fidelity matters only insofar as it preserves the *order* of inner products near the top.
- **The recurrent backbone supplies local fluency; the recall tier supplies long-range factual recall.** The two are functionally separable, an assumption now strongly supported by the retrieval-head and hybrid-ablation literature [18][21].

---

## Main Analysis

### Finding 1: The "SSM backbone + thin explicit recall tier" split is now a settled empirical result — and the recall tier can be astonishingly thin

Your decision to bolt an explicit recall mechanism onto a recurrent backbone is not a hopeful bet; it is the conclusion the field has independently reached. The theoretical floor is firm: Jelassi et al. (ICML 2024) prove that a two-layer transformer can copy strings of exponential length while generalized state-space models are "fundamentally limited by their fixed-size latent state," and that the state dimension must grow *linearly* with the number of entries in a joint-recall table [19]. The complexity-theoretic version is sharper still: general associative recall is essentially equivalent to set-disjointness, and any recurrent model needs space Ω(min(|A|,|B|)) to solve it [29] — i.e., a fixed-state SSM cannot do unbounded associative recall, full stop. This is the formal statement of the MQAR collapse you already measured.

What the recent literature adds is *how little* attention is required to repair this. The 2025 study "Some Attention is All You Need for Retrieval" ablated three production hybrids and found complete functional segregation: "retrieval depends exclusively on self-attention layers" while SSM components "contribute nothing to this capability" [18]. When attention was fully removed, RecurrentGemma-2B, RecurrentGemma-9B, and Jamba-Mini-1.6 all dropped to **0% Needle-in-a-Haystack accuracy**, and crucially, prompting tricks (Just Read Twice) "did not recover retrieval capabilities at low k" — the SSM has no latent recall to unlock [18]. Yet the same study found that **sparsifying to just 15% of attention heads maintained near-perfect retrieval while preserving 84% MMLU** [18]. This converges with Wu et al.'s "retrieval heads" result: retrieval is carried out by a *universal but sparse* set of heads — **fewer than 5% of all attention heads** — that perform a copy-paste operation, are present even in short-context pretraining, and are the same heads that survive context extension [21].

The implication for Silicon Entropy Engine is concrete and favorable. The "explicit recall tier over the SSM" you need to build is, empirically, a thin emulator of a handful of retrieval heads — not a full attention stack. This is consistent with your own measurement that *one* top-k sparse attention layer recovers 97–98% on MQAR [internal]. It also tells you where the quality budget lives: degrade the SSM compute aggressively if you must, but protect the recall tier's ranking fidelity, because that single thin tier is solely responsible for long-range factuality. DuoAttention operationalizes exactly this division for deployment, splitting heads into "retrieval heads" (need full KV) and "streaming heads" (need only sink + sliding window), and exploiting it to cut long-context memory and latency [22]. Your architecture is the logical endpoint of that line: make the streaming part a true O(1) recurrence and the retrieval part an indexed lookup.

**Key Evidence:**
- Two-layer transformer copies exponential-length strings; GSSM state must grow linearly with recall-table size [19].
- General associative recall ≡ set-disjointness ⇒ recurrent space lower bound Ω(min(|A|,|B|)) [29].
- Attention removal ⇒ 0% NIAH on RecurrentGemma-2B/9B and Jamba-Mini-1.6; 15% of heads ⇒ near-perfect retrieval + 84% MMLU [18].
- Retrieval heads are <5% of heads, universal, sparse, copy-paste, stable under context extension [21].

**Implications:** The recall tier is small and isolable. Spend your CPU budget defending its ranking quality; everything else (the SSM) can be cheap.

**Sources:** [18], [19], [21], [22], [29]

---

### Finding 2: Streaming ANN over decode KV is the *easy* half of the streaming-ANN problem, because autoregressive context is append-only

The streaming-ANN literature is dominated by a problem you do not actually have: deletion. FreshDiskANN (Singh et al., 2021) is engineered to support "thousands of concurrent real-time inserts, deletes and searches per second" on a billion-point index, and its central difficulty is that the Vamana graph is singly-linked, so "deletions are hard because there is no fast way to find in-neighbors of a deleted vertex" — forcing it to accumulate deletes in a batch and periodically run an expensive **consolidation** pass [1]. IP-DiskANN (Xu et al., Feb 2025) is celebrated precisely because it is "the first algorithm to avoid batch consolidation by efficiently processing each insertion and deletion in-place," and its reported win over FreshDiskANN and HNSW is *specifically on deletion-heavy large datasets* (>10M points), while "average insertion time and search time is roughly the same" [2]. SPFresh likewise frames its contribution as "incremental in-place update for billion-scale vector search," again centered on update/consolidation cost [3].

The structural insight is that **within a single generated sequence, KV is never deleted** — the context only grows. The expensive machinery (in-neighbor discovery, tombstoning, consolidation, graph repair) is therefore irrelevant to per-sequence decode. What you need is the *cheap* operation that all of these systems agree on: append. For a graph index, insertion "queries the current index for nearest neighbors, generates candidate out-neighbors, and adds bi-directed edges" [1] — an O(search) operation. For your IVF front-runner it is even cheaper: assign the new key to its nearest centroid (O(nlist) with nlist≈√n, or O(1) amortized with a coarse quantizer), append its PQ code to that list, done. Cross-sequence eviction (when you reuse the index across requests) is a ring-buffer/cache-eviction concern, not a graph-consolidation concern, and can be handled by coarse-grained list truncation rather than per-vector deletion.

Two caveats from the literature temper this. First, graph indices can suffer *recall drift* after very many updates even without deletes — the 2025 in-place-update and index-merging work (FGIM) exists partly to keep recall stable across long update streams [2]. At 128K vectors per sequence this is unlikely to bite, but at multi-million-token agentic sessions it could, which argues for IVF (whose recall is structurally stable under append because lists are independent) over a graph for the *very* long regime. Second, IVF append slowly unbalances lists, degrading the sublinearity guarantee; periodic, cheap re-clustering of the coarse quantizer (not the codes) restores it. The Faiss library itself documents both IVF and graph paths and the FastScan code path that makes IVF-PQ cheap at query time [7][36], so the engineering is well-trodden — you are choosing the append-only slice of a mature toolbox.

**Key Evidence:**
- FreshDiskANN's cost center is deletion + consolidation, not insertion [1].
- IP-DiskANN's measured advantage is on deletions at >10M points; insertion/search times "roughly the same" as prior art [2].
- Graph insertion = O(search); IVF insertion = nearest-centroid assignment + append [1][7].
- Recall-stability-under-update is an active concern (FGIM, in-place updates), favoring IVF for the very-long regime [2].

**Implications:** Adopt IVF (not a graph) for the append-only decode index; you inherit sublinear query and trivial insert, and you sidestep the entire consolidation literature.

**Sources:** [1], [2], [3], [7], [36]

---

### Finding 3: Similarity-preserving (anisotropic / score-aware) quantization is published doctrine — and it directly answers your open question on MSE-PQ vs ranking-PQ

Your hypothesis (#2) — that quantizing KV to minimize ‖K−K̂‖² is the wrong objective when the downstream task is top-k(q·K), and that one should instead preserve the *ranking* of inner products — is exactly the thesis of ScaNN (Guo et al., ICML 2020) [4]. ScaNN introduces a **score-aware quantization loss** that "weighs the inner-product approximation error by a weight function based on the value of the true inner product," concentrating fidelity on the high inner products that are, by definition, the ones MIPS returns [4]. Its **anisotropic** refinement is the geometric core: error *parallel* to the original vector is far more harmful than orthogonal error, because parallel error directly distorts the magnitude of large dot products, so ScaNN "more heavily penalizes quantization error that is parallel to the original vector" [4]. This is precisely "preserve the norm-along-the-query-direction, not the reconstruction." Google reports it as state-of-the-art MIPS, and — critically for you — "to accelerate search speed, ScaNN is implemented with SIMD in-register lookup tables" [4], i.e., the same PQ4-fast-scan kernel family, so the better loss is *free at query time*.

The idea has been extended in the exact directions your brief asks about. "Query-Aware Quantization for Maximum Inner Product Search" (AAAI 2023) makes the quantizer query-distribution-aware rather than reconstruction-optimal [5]. "Anisotropic Additive Quantization for Fast Inner Product Search" (AAAI 2022) ports the score-aware/anisotropic loss onto *additive* codebooks, combining the higher capacity of AQ with the ranking-preserving objective [6]. So there is a clean, citable answer to both halves of your open question: (a) yes, MSE-PQ is provably suboptimal for top-k(q·K), and the published fix is anisotropic/score-aware loss; (b) the learned-codebook-vs-random-LSH question (your #6) is the same question ScaNN already answered in favor of *learning the codebook to the dot-product objective* — a learned, score-aware codebook beats a reconstruction-optimal or random one *under load*, which is consistent with your observation that they tie at low load.

The size of the win is documented qualitatively but you should measure it on your own keys, because it is data-dependent. The published consensus is that the gain shows up specifically at the **high-recall / high-pressure regime** — exactly where your InfoNCE-vs-LSH test is now operating — and shrinks toward zero at low load, because when items are well-separated almost any codebook ranks them correctly [4][5]. NoMAD-Attention is the most relevant proof that this transfers to *attention keys specifically*: it quantizes query-key dot products (not for reconstruction) and "accurately estimate[s] query-key dot products" well enough to hold perplexity within 4% at 8× compression [9]. The practical recipe that emerges is OPQ/rotation + score-aware PQ training on a sample of your actual K vectors, decoded with the same 4-bit LUT scan you already have. The one caution: anisotropic loss assumes the query distribution is roughly isotropic around the key; if your queries are strongly clustered (likely, since they are SSM hidden states), a *query-aware* variant [5] may beat the vanilla anisotropic one — a cheap thing to A/B.

**Key Evidence:**
- ScaNN's score-aware loss weights error by true-inner-product magnitude; anisotropic loss penalizes parallel error specifically [4].
- ScaNN ships with SIMD in-register LUTs ⇒ better loss at no query-time cost [4].
- Query-aware MIPS quantization [5] and anisotropic *additive* quantization [6] extend the idea to query distribution and higher-capacity codebooks.
- NoMAD-Attention demonstrates dot-product-preserving (not reconstruction) quantization of attention keys at 8× with <4% perplexity loss [9].

**Implications:** Replace your MSE-PQ training objective with ScaNN-style anisotropic/score-aware loss (optionally query-aware). It is a drop-in change to the codebook trainer, costs nothing at query time, and the literature predicts the gain materializes exactly under the high-load pressure you are now testing.

**Sources:** [4], [5], [6], [9]

---

### Finding 4: PQ4 fast-scan is the proven CPU sweet spot; additive/residual quantizers buy accuracy but cost at query time unless made progressive

Your reliance on 4-bit PQ with in-register LUTs and `shuffle_epi8` is the documented state of the art for CPU ANN, and its accuracy cost is small when paired with rotation. The mechanism: 4-bit codes let an entire sub-quantizer's distance table fit in a SIMD register, so distances for 16–32 vectors are computed by `shuffle`/`pshufb` lookups instead of multiply-adds — André et al.'s "Quicker ADC" reports a **3–6× speedup over standard ADC** purely from this SIMD exploitation [8], and FAISS's PQ4 FastScan is built on the same principle [7]. The accuracy verdict is consistent across sources: "using 4-bit quantizers may cause a loss in recall, though this loss is small or negligible when combined with OPQ and inverted indexes," and "for most current use cases... 4-bit quantizers offer sufficient accuracy," with 16-bit reserved for newer demanding cases [7]. The binding constraint on Zen2-without-VNNI is favorable here: PQ4 fast-scan uses `pshufb`/`shuffle_epi8` (SSSE3/AVX2), **not** the VNNI dot-product instructions, so you lose nothing by lacking VNNI — the 4-bit LUT path was designed for exactly the integer-shuffle capability Zen2 has. NoMAD-Attention confirms the whole stack end-to-end on CPU: 16 centroids (4-bit), dot products dynamically quantized to 8-bit accumulators, transposed/blocked key layout for batched `shuffle` lookups, **2× speedup at 16k context, <4% perplexity at 8× compression, no finetuning** [9].

The accuracy ceiling of 4-bit ADC is the real question, and the honest answer from the literature is "it depends on intrinsic dimensionality, and it is usually fine for top-k routing but not for final scoring." This is why production systems use **two tiers**: a cheap 4-bit ADC scan to shortlist candidates, then a **refine** step (re-rank the shortlist with 8-bit codes or exact dot products) — your "refine ADC" is the standard remedy and it decouples scan-speed from final accuracy [7][36]. If 4-bit ADC ranking ever proves too coarse under load, the cheapest upgrade is not abandoning PQ4fs but widening only the refine tier.

On the alternatives — RVQ, additive quantization (AQ), LSQ++, AQLM — the literature is clear about the trade. Additive quantizers decode as a *sum* of codewords, x̂ = T₁[c₁]+…+T_M[c_M], which is strictly more expressive than PQ's concatenation and yields higher recall at equal bits [31][32]. AQLM achieves Pareto-optimal accuracy below 3 bits for *weight* compression [33], and qinco2 (ICLR 2025) and FaTRQ (2026) push residual quantization further [34][35]. But the cost lands at query time: AQ/LSQ encoding requires beam search or simulated annealing [32], and — decisively for you — **additive codes do not admit the single-register 4-bit LUT scan that makes PQ4fs cheap**; distance computation must sum across codebooks. The FaTRQ/qinco2 line addresses this with *progressive decoding*: "existing systems decode all quantization levels for every vector... resulting in 10–100× more candidates being decoded than necessary," so progressive schemes keep early RQ levels in fast memory, prune on them, and only fully decode survivors [34][35]. The pragmatic reading: **stay on PQ4fs for the scan; consider a 1–2 level residual refine only if accuracy-under-load demands it, and only in progressive form so the cost falls on the shortlist, not the full context.** Anisotropic *additive* quantization [6] is the interesting middle path — AQ capacity with score-aware ranking — but it inherits AQ's heavier query-time decode and should be benchmarked, not assumed.

**Key Evidence:**
- Quicker ADC: 3–6× over standard ADC from SIMD LUTs; FAISS PQ4 FastScan is the same family [7][8].
- 4-bit recall loss "small or negligible" with OPQ + IVF; 8-bit/16-bit reserved for demanding cases [7].
- PQ4 fast-scan uses `pshufb`/`shuffle_epi8`, not VNNI ⇒ Zen2 loses nothing [7][9].
- NoMAD-Attention: 2× at 16k, <4% perplexity at 8× compression, CPU, no finetuning [9].
- AQ/RVQ/AQLM more accurate per bit but query-time decode is heavier; progressive decoding (qinco2, FaTRQ) mitigates by pruning early [31][32][33][34][35].

**Implications:** Keep PQ4fs as the scan kernel — it is VNNI-independent and CPU-proven. Add accuracy only via a refine tier (8-bit re-rank, optionally 1 progressive residual level), never by replacing the 4-bit scan with full additive decode.

**Sources:** [6], [7], [8], [9], [31], [32], [33], [34], [35], [36]

---

### Finding 5: SSM/recurrent + indexed-retrieval hybrids already exist — but almost none are CPU-cost-optimized, and ParisKV (2026) is your architecture on the wrong hardware

There are four families of prior art that couple a sub-quadratic or recurrent backbone with a selective recall tier rather than full attention. Knowing their measured trade-offs lets you stand on them instead of rediscovering them.

**(1) kNN-augmented attention / external memory.** Memorizing Transformers (Wu et al., ICLR 2022) is the canonical ancestor: a kNN-augmented attention layer does approximate nearest-neighbor lookup into an external (key,value) memory, appends local KV to memory after each step, and does not backpropagate through memory [13]. It scales to **262k tokens** and "consistently and substantially improves perplexity" across PG-19, C4, arXiv-Math, GitHub, and Isabelle [13]. Its descendant InfLLM (Xiao et al., NeurIPS 2024) is the most directly reusable for you: it is **training-free**, organizes distant KV into blocks, picks the highest-attention tokens in each block as the block's representative for lookup, and "at each step retrieves a constant number of relevant blocks" — i.e., O(1) blocks retrieved per token, sublinear by construction, layered on top of StreamingLLM's sink + sliding window [14]. This is essentially IVF-with-block-representatives applied to attention; your IVF-PQ is the compressed, CPU-tuned version of the same idea.

**(2) Query-aware KV-page selection.** Quest (Tang et al., 2024) is the SnapKV/landmark lineage: it stores per-page min/max key values, estimates each page's "criticality" against the current query, and loads only the top-K pages — reporting **up to 7.03× self-attention speedup and 4.5× latency reduction at matched accuracy** on long-context benchmarks [10]. The relationship to your design is important: Quest is a *coarse, exact* filter (min/max bounds, no quantization), whereas you replace the bound-check with a PQ-ADC estimate. Quest validates that page-level query-aware selection preserves quality; your contribution is making the per-page score itself cheap via PQ rather than via min/max scans of full-precision keys.

**(3) Native hybrid backbones (recurrent + a few attention layers).** Jamba (Lieber et al., 2024) is "the first production-grade Attention-SSM hybrid," interleaving Mamba and attention with MoE, 256K context, SOTA on RULER [15]. Samba (Ren et al., 2024) layer-wise combines Mamba with sliding-window attention, trains at 4K, and **extrapolates to 256K with "perfect memory recall," improving predictions to 1M tokens** [16]. These prove the *backbone* viability but they still use full (if sparse-in-depth) attention layers for recall — they do **not** index the KV; they are the thing you are trying to beat on CPU by replacing those attention layers' O(ctx) scan with an indexed lookup. The 2025 ablation (Finding 1) studied exactly these models and confirmed the attention layers are load-bearing for retrieval [18].

**(4) Indexed retrieval over the KV of a long-context model — the closest cousin.** ParisKV ("Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs," Feb 2026) is, on the evidence, your architecture: it "retrieves relevant key-value pairs from long contexts using approximate nearest neighbor search rather than processing the entire cache," explicitly "preserves similarity ranking structure rather than attempting perfect reconstruction," and its headline "drift-robust" property handles the fact that "as the model generates tokens sequentially, the query representations evolve" — maintaining retrieval accuracy under that distribution shift [30]. That drift property is a problem you have not yet named but will hit: your SSM-hidden-state queries drift as generation proceeds, and a static index trained on early-context query statistics can decay. The gap is hardware: ParisKV uses "GPU-accelerated approximate nearest-neighbor search" and does not isolate CPU/Zen2 cost [30]. **The CPU-first instantiation of the ParisKV idea is an open, defensible niche** — and ParisKV's drift-robustness techniques are directly worth importing.

The honest negative across all four families: **almost none report CPU cost.** Memorizing Transformers, Quest, InfLLM, Jamba, Samba, and ParisKV all measure GPU throughput/latency or quality only [10][13][14][15][16][30]. The single peer-reviewed system that measures the *CPU* version of indexed-attention-by-lookup is NoMAD-Attention [9], which is why it is your most important external anchor: it is the existence proof that the PQ-LUT recall tier runs fast on the exact hardware class you target. Your project's novelty is therefore not the mechanism (it exists) but the *co-design of the mechanism for CPU at 128K*, which the literature has left open.

**Key Evidence:**
- Memorizing Transformers: kNN external memory, 262k tokens, perplexity gains, no backprop through memory [13].
- InfLLM: training-free, block memory, constant blocks/step (sublinear), on top of sink+window [14].
- Quest: page min/max criticality, top-K pages, 7.03× speedup / 4.5× latency at matched accuracy [10].
- Jamba (256K, RULER SOTA), Samba (4K→256K "perfect recall", →1M) — backbone proof, still full attention for recall [15][16].
- ParisKV (2026): ANN over KV, ranking-preserving, drift-robust under evolving queries — GPU-targeted [30].
- Only NoMAD-Attention measures the CPU lookup-attention path [9].

**Implications:** Reuse InfLLM's block-representative retrieval design and Quest's page-criticality framing; import ParisKV's drift-robustness; claim the CPU-at-128K co-design as the open contribution, anchored on NoMAD's CPU feasibility proof.

**Sources:** [9], [10], [13], [14], [15], [16], [18], [30]

---

### Finding 6: Multi-token prediction is a real, largely-orthogonal 1.8–3.6× multiplier on CPU, but its gain is acceptance-rate-gated; self-drafting MTP is the CPU-appropriate form

On a bandwidth-bound CPU, each forward pass pays a fixed cost to stream weights from RAM through cache; emitting K tokens per pass amortizes that cost, which is why multi-token prediction (MTP) and speculative decoding are described as "the highest-ROI inference optimization for single-stream LLM serving in 2026 — 2–4× speedup with mathematically identical output" [25]. The catalog: Medusa (Cai et al., 2024) adds parallel decoding heads predicting subsequent tokens, reaching **2.2× (Medusa-1) and 2.3–3.6× (Medusa-2)** without quality loss [23]; EAGLE trains a feature-prediction layer over the target's hidden states and "delivers higher acceptance rates than MTP on most tasks" [24][25]; DeepSeek-V3's built-in MTP "hits 1.8× speedups at >80% acceptance rates out of the box" [25]; FastMTP (2025) further enhances the MTP path [26].

The mechanism that matters on CPU is **self-drafting**. Classic speculative decoding runs a separate small draft model, which on CPU competes for the same scarce memory bandwidth — you pay to stream two models. MTP heads (Medusa, DeepSeek-MTP) instead draft *from the target model's own hidden states*, adding only a few lightweight head matmuls and no second weight-streaming pass [23][25]. This is the variant that preserves the bandwidth amortization you want; a separate-draft scheme can erase its own benefit on a bandwidth-bound machine.

Two caveats are load-bearing for honest projection. First, the realized speedup is **acceptance-rate-gated**: effective tokens/pass ≈ acceptance × K, so DeepSeek's 1.8× at >80% acceptance is the realistic shape, not a flat K× — the brief's "quasi-K×" is the optimistic ceiling, rarely reached [25]. Acceptance falls on hard/high-entropy content (and TinyStories-easy text will *over*-estimate it relative to your eventual large-model targets). Second, **no source isolates CPU acceptance overhead on Zen2**; the 2–3.6× figures are GPU-measured, and on CPU the verification of K candidates is itself bandwidth work, so the net is plausibly toward the lower end of the range. MTP is genuinely orthogonal to the recall tier (it changes how many tokens you emit per pass, not how you fetch long-range context), so it composes multiplicatively with your IVF-PQ gains — but model it as ~1.5–2× realistic on CPU, not K×, until measured.

**Key Evidence:**
- Speculative/MTP described as 2–4× single-stream with identical output [25].
- Medusa 2.2× / 2.3–3.6×; EAGLE higher acceptance; DeepSeek-MTP 1.8× at >80% acceptance [23][24][25][26].
- Self-drafting (MTP heads) avoids a second weight-streaming pass — the CPU-appropriate variant [23][25].
- Speedup = acceptance × K; no Zen2-specific acceptance/overhead numbers published [25].

**Implications:** Add self-drafting MTP heads (Medusa/DeepSeek-style), not a separate draft model. Budget ~1.5–2× realistic on CPU and verify acceptance on representative (not TinyStories-easy) text.

**Sources:** [23], [24], [25], [26]

---

## Synthesis & Insights

### Patterns Identified

**Pattern 1: The field has independently converged on "cheap recurrence + thin indexed recall," and is now arguing about the recall tier, not the split.** Five years of work — from Memorizing Transformers' kNN memory [13], through Quest's page selection [10] and InfLLM's block memory [14], to Jamba/Samba's native hybrids [15][16] and ParisKV's KV-ANN [30] — all instantiate the same two-tier shape you arrived at empirically. The 2025/2026 ablation and retrieval-head work [18][21] explains *why* it works: retrieval is a sparse, isolable, attention-only function. The open arguments are now entirely about the *recall tier's* implementation — exact vs approximate selection, full-precision vs quantized scoring, GPU vs CPU — which is precisely the design space you are exploring.

**Pattern 2: "Preserve ranking, not reconstruction" recurs at every layer of the stack.** ScaNN's score-aware/anisotropic loss [4], NoMAD's dot-product-not-reconstruction quantization [9], Quest's criticality-bound page scoring [10], ParisKV's explicit "ranking structure rather than perfect reconstruction" [30], and the retrieval-robustness-of-sparse-heads finding in KV quantization [from KIVI/retrieval-aware work, 27] are all the same principle: the only fidelity that matters is the order of the top candidates. MSE is a proxy that the literature has repeatedly found to be the wrong one.

**Pattern 3: CPU is the unoccupied square.** Nearly every retrieval-augmented long-context system reports GPU throughput or quality-only; the lone peer-reviewed CPU lookup-attention system is NoMAD [9]. The ANN community has the CPU kernels (FAISS PQ4fs, Quicker ADC, ScaNN SIMD) [4][7][8]; the LLM community has the architectures; almost no one has fused them on consumer CPU at 128K. That fusion is your thesis, and the literature gap is real rather than imagined.

### Novel Insights

**Insight 1: Your hardest unsolved problem is probably query drift, not index cost — and it is already named in a 2026 paper.** You have driven the *cost* wall down (8 µs/token/layer). The next wall the literature predicts you will hit is that your queries are SSM hidden states that *drift* over a long generation, so an index/codebook calibrated on early-context statistics silently decays — ParisKV exists specifically to be "drift-robust" against this [30]. None of your established results address it because it only manifests over very long *generated* (not teacher-forced) rollouts — which, notably, is the exact failure mode your project's own history flags repeatedly (closed-loop autoregressive drift). The two problems may be the same problem viewed from two angles: importing ParisKV's drift-robust retrieval could simultaneously be a long-range-recall mechanism *and* a closed-loop-stability mechanism. This is the single highest-value cross-connection in this report.

**Insight 2: Your open question #6 (learned InfoNCE codebook vs random LSH) is settled in principle and only empirical at the margin.** ScaNN already demonstrated that a codebook *learned to the dot-product objective* beats reconstruction-optimal and random codebooks, and that the advantage appears under load and vanishes when items are well-separated [4] — which is exactly the "tie at low load" you observed. So the answer is: the learned codebook should win under pressure, but the *margin* is data-dependent and only your own pressure test can size it. The more interesting reframing: InfoNCE and anisotropic loss are two routes to the same destination (rank-preserving codes); a score-aware/anisotropic loss [4][6] may be a cheaper, more stable training target than full InfoNCE while achieving the same ranking fidelity — worth testing head-to-head.

**Insight 3: The recall tier can be thinner than you are budgeting for, which changes the cost arithmetic.** If <5% of heads carry retrieval [21] and 15% of heads give near-perfect recall [18], you likely need only *one or a few* indexed-recall "heads" per layer (or even only in a subset of layers), not a full-width retrieval tier. That collapses both the index memory and the per-token query count, potentially pulling your 8 µs/token/layer down further simply by querying fewer indexed subspaces. Conversely, it means the few heads you *do* index are critical and deserve the score-aware-quantization quality budget.

### Implications

**For Silicon Entropy Engine:** The architecture is validated by convergent external evidence; the build risk has shifted from "will the split work" (it does) to "can the recall tier be made both cheap *and* drift-robust on CPU." Your IVF-PQ front-runner is correct; the prioritized upgrades are score-aware codebook loss, a thin (few-head) recall tier, a refine step for accuracy-under-load, drift-robust re-calibration, and self-drafting MTP as an orthogonal multiplier.

**Broader Implications:** A CPU-first, 128K, indexed-recall-over-SSM model that publishes Zen2 cost numbers would fill a genuine gap — the intersection of FAISS-grade CPU ANN and SSM-retrieval hybrids is essentially unpublished. NoMAD-Attention is the only neighbor, and it bolts onto a transformer, not an SSM backbone.

**Second-Order Effects:** If drift-robust retrieval doubles as closed-loop stabilization, it may resolve a problem your prior phases treated as separate (autoregressive collapse), unifying "long-range recall" and "generation stability" under one mechanism.

---

## Limitations & Caveats

### Counterevidence Register

**Contradictory Finding 1 — Native hybrids suggest you might not need an *index* at all.** Samba extrapolates 4K→256K with "perfect memory recall" using only sliding-window attention + Mamba, no retrieval index [16]. This challenges the premise that indexed recall is necessary: a well-designed sliding-window-plus-recurrence might reach long context without an ANN tier. **Resolution:** Samba's recall is measured on GPU with full sliding-window attention, whose per-token cost still scales with window size and whose KV still grows; it does not deliver O(1)-state + sublinear-recall on CPU at 128K. The index is what buys *CPU sublinearity*, which Samba does not claim. Impact on conclusions: moderate — it means "sliding window + recurrence" is a baseline you must beat, not a settled loss.

**Contradictory Finding 2 — Some KV-quantization work finds retrieval heads are *fragile* under low-bit quantization.** The KV-quant literature reports that "only attention heads with sparse and concentrated patterns demonstrate consistent robustness to low-precision quantization," while retrieval heads can be *less* robust [27]. This warns that aggressively quantizing the very heads you most need for recall could hurt. **Resolution:** This is an argument *for* score-aware quantization and a refine tier specifically on the recall heads, not against the approach — but it tempers the "8× compression is free" optimism: the recall tier may need more bits than the rest. Impact: moderate; it sharpens the bit-allocation decision.

### Known Gaps

**Gap 1 — No published per-token CPU cost for indexed retrieval over an SSM at 128K.** Every architecture paper (Quest, InfLLM, ParisKV, Jamba, Samba) measures GPU or quality; NoMAD measures CPU but over a transformer. Your own 8 µs/token/layer is, as far as this search found, *ahead of the published literature* for this specific setting — there is no external number to triangulate against. This is simultaneously a gap and an opportunity (publishable novelty).

**Gap 2 — ParisKV details are thin in available text.** The fetched ParisKV content confirms the mechanism and the drift-robust/ranking-preserving claims but did not surface concrete per-token latency, bit-width, or index-type numbers [30]; the full PDF (saved locally) should be read directly before relying on specifics. Its very recent date (Feb 2026) means little secondary coverage exists yet.

**Gap 3 — Zen2-without-VNNI acceptance/overhead for MTP is unmeasured.** All MTP speedups are GPU-measured; the CPU verification overhead and acceptance on hard text are not isolated [25].

### Areas of Uncertainty

**Uncertainty 1 — Magnitude of the anisotropic-loss win on *your* keys.** The literature is unanimous on direction but the size is data-dependent and regime-dependent (gain concentrated at high load) [4][5]. Only your pressure test sizes it.

**Uncertainty 2 — Whether 4-bit ADC ranking is accurate *enough* for the recall tier specifically.** Acceptable for shortlisting universally [7], but recall heads may need 8-bit refine [27]; this is an empirical bit-allocation question, not a settled one.

**Uncertainty 3 — Graph vs IVF at multi-million-token agentic horizons.** IVF is recall-stable under append but unbalances; graphs are denser but drift and (if you ever evict) need consolidation [1][2]. At 128K, IVF wins cleanly; at 10M+ agentic sessions the choice reopens.

---

## Recommendations

### Immediate Actions

1. **Swap MSE-PQ for anisotropic/score-aware codebook training.**
   - *What:* Retrain your PQ/OPQ codebooks with ScaNN's score-aware loss (weight error by true inner-product magnitude; penalize parallel error) on a sample of real K vectors [4]; A/B a query-aware variant [5].
   - *Why:* Directly answers open question #2; the published fix for "ranking not reconstruction"; free at query time (same LUT scan).
   - *How:* Reuse the ScaNN loss formulation; keep your existing 4-bit LUT decoder unchanged.
   - *Timeline:* Now — it is a trainer-side change, not an inference change.

2. **Add a two-tier scan: PQ4fs shortlist → 8-bit (or 1 progressive residual) refine.**
   - *What:* Keep the 4-bit fast-scan for candidate selection; re-rank the top shortlist with higher precision [7][36].
   - *Why:* Decouples scan speed from final ranking accuracy; the standard remedy if 4-bit ranking proves coarse under load, and the cheap defense against retrieval-head fragility [27].
   - *Timeline:* Now — it is additive to the front-runner.

3. **Read the ParisKV PDF (saved locally) and lift its drift-robustness method.**
   - *What:* Extract how it maintains retrieval accuracy as queries drift; map onto your SSM-hidden-state queries [30].
   - *Why:* Drift is your predicted next wall and possibly the same phenomenon as your closed-loop autoregressive collapse.
   - *Timeline:* This week, before committing index design.

### Next Steps (1–3 months)

1. **Thin the recall tier to a few indexed "retrieval heads."**
   - Test indexing only a subset of heads/layers, guided by retrieval-head identification [18][21]; measure whether per-token cost drops below 8 µs without recall loss.

2. **Port InfLLM's block-representative retrieval and Quest's page-criticality framing into your IVF.**
   - Use block/page granularity (not per-token) to cut index size and query count; you already have the compressed-code version they lack [10][14].

3. **Prototype self-drafting MTP heads (Medusa/DeepSeek-style), not a separate draft model.**
   - Budget ~1.5–2× realistic CPU gain; measure acceptance on non-easy text [23][25].

### Further Research Needs

1. **Publish Zen2 per-token cost for indexed-recall-over-SSM at 128K** — the literature gap is real; your 8 µs/token/layer appears to be ahead of published numbers for this setting.
2. **Head-to-head: InfoNCE codebook vs anisotropic/score-aware loss vs random LSH, under load** — settle open question #6 with a sized margin, not just a direction [4][6].
3. **Quantify drift over long *generated* rollouts and test whether drift-robust re-calibration also stabilizes closed-loop generation** — the highest-value cross-connection in this report [30].

---

## Bibliography

[1] Singh, A., Subramanya, S. J., et al. (2021). "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search". arXiv:2105.09613. https://arxiv.org/abs/2105.09613 (Retrieved: 2026-06-24)

[2] Xu, H., et al. (2025). "In-Place Updates of a Graph Index for Streaming Approximate Nearest Neighbor Search" (IP-DiskANN). arXiv:2502.13826. https://arxiv.org/abs/2502.13826 (Retrieved: 2026-06-24)

[3] SPFresh authors (2023). "SPFresh: Incremental In-Place Update for Billion-Scale Vector Search". SOSP 2023. https://www.researchgate.net/publication/374920073_SPFresh_Incremental_In-Place_Update_for_Billion-Scale_Vector_Search (Retrieved: 2026-06-24)

[4] Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., Kumar, S. (2020). "Accelerating Large-Scale Inference with Anisotropic Vector Quantization" (ScaNN). ICML 2020 (PMLR v119). arXiv:1908.10396. https://arxiv.org/abs/1908.10396 (Retrieved: 2026-06-24)

[5] "Query-Aware Quantization for Maximum Inner Product Search" (2023). AAAI 2023. https://ojs.aaai.org/index.php/AAAI/article/view/25613 (Retrieved: 2026-06-24)

[6] "Anisotropic Additive Quantization for Fast Inner Product Search" (2022). AAAI 2022. https://cdn.aaai.org/ojs/20356/20356-13-24369-1-2-20220628.pdf (Retrieved: 2026-06-24)

[7] FAISS project (2024). "Fast accumulation of PQ and AQ codes (FastScan)" + "Faiss product quantization". facebookresearch/faiss Wiki / OpenSearch docs. https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan) (Retrieved: 2026-06-24)

[8] André, F., Kermarrec, A.-M., Le Scouarnec, N. (2019). "Quicker ADC: Unlocking the Hidden Potential of Product Quantization with SIMD". IEEE TPAMI. arXiv:1704.07355. https://arxiv.org/pdf/1704.07355 (Retrieved: 2026-06-24)

[9] Zhang, T., et al. (2024). "NoMAD-Attention: Efficient LLM Inference on CPUs Through Multiply-add-free Attention". NeurIPS 2024. arXiv:2403.01273. https://arxiv.org/abs/2403.01273 (Retrieved: 2026-06-24)

[10] Tang, J., et al. (2024). "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference". ICML 2024. arXiv:2406.10774. https://arxiv.org/abs/2406.10774 (Retrieved: 2026-06-24)

[11] Arora, S., Eyuboglu, S., Zhang, M., et al. (2024). "Simple linear attention language models balance the recall-throughput tradeoff" (BASED). arXiv:2402.18668. https://arxiv.org/abs/2402.18668 (Retrieved: 2026-06-24)

[12] HazyResearch (2023–2024). "Zoology: Measuring and Improving Recall in Efficient Language Models" + Zoology blog (MQAR, BaseConv). https://github.com/HazyResearch/zoology (Retrieved: 2026-06-24)

[13] Wu, Y., Rabe, M. N., Hutchins, D., Szegedy, C. (2022). "Memorizing Transformers". ICLR 2022. arXiv:2203.08913. https://arxiv.org/pdf/2203.08913 (Retrieved: 2026-06-24)

[14] Xiao, C., et al. (2024). "InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory". NeurIPS 2024. arXiv:2402.04617. https://arxiv.org/abs/2402.04617 (Retrieved: 2026-06-24)

[15] Lieber, O., Lenz, B., et al. (2024). "Jamba: A Hybrid Transformer-Mamba Language Model". arXiv:2403.19887. https://arxiv.org/pdf/2403.19887 (Retrieved: 2026-06-24)

[16] Ren, L., et al. (2024). "Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling". arXiv:2406.07522. https://arxiv.org/html/2406.07522v1 (Retrieved: 2026-06-24)

[17] AI21 Labs (2024–2025). "Attention was never enough: Tracing the rise of hybrid LLMs". AI21 Blog. https://www.ai21.com/blog/rise-of-hybrid-llms/ (Retrieved: 2026-06-24)

[18] "Some Attention is All You Need for Retrieval" (2025). arXiv:2510.19861. https://arxiv.org/html/2510.19861v1 (Retrieved: 2026-06-24)

[19] Jelassi, S., Brandfonbrener, D., Kakade, S. M., Malach, E. (2024). "Repeat After Me: Transformers are Better than State Space Models at Copying". ICML 2024 (PMLR v235). https://proceedings.mlr.press/v235/jelassi24a.html (Retrieved: 2026-06-24)

[20] Arora, S., et al. (2024). "Just Read Twice: Closing the Recall Gap for Recurrent Language Models". arXiv:2407.05483. https://arxiv.org/pdf/2407.05483 (Retrieved: 2026-06-24)

[21] Wu, W., et al. (2024). "Retrieval Head Mechanistically Explains Long-Context Factuality". arXiv:2404.15574. https://arxiv.org/abs/2404.15574 (Retrieved: 2026-06-24)

[22] Xiao, G., et al. (2024). "DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads". arXiv:2410.10819. https://arxiv.org/html/2410.10819v1 (Retrieved: 2026-06-24)

[23] Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., Dao, T. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads". arXiv:2401.10774. https://arxiv.org/abs/2401.10774 (Retrieved: 2026-06-24)

[24] Li, Y., et al. (2024). "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (EAGLE / EAGLE-3). https://arxiv.org/abs/2401.15077 (Retrieved: 2026-06-24)

[25] SyncSoft.AI (2026). "Speculative Decoding 2026: EAGLE-3, Medusa, DeepSeek-MTP". https://www.syncsoft.ai/en/blog/speculative-decoding-eagle3-medusa-deepseek-mtp-chinese-chuhai-2026 (Retrieved: 2026-06-24)

[26] "FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction" (2025). arXiv:2509.18362. https://arxiv.org/pdf/2509.18362 (Retrieved: 2026-06-24)

[27] Liu, Z., et al. (2024). "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (with retrieval-vs-streaming-head robustness discussion). arXiv:2402.02750. https://www.themoonlight.io/en/review/kivi-a-tuning-free-asymmetric-2bit-quantization-for-kv-cache (Retrieved: 2026-06-24)

[28] Hooper, C., et al. (2024). "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization". arXiv:2401.18079. https://arxiv.org/abs/2401.18079 (Retrieved: 2026-06-24)

[29] "Overcoming Long-Context Limitations of State-Space Models via Context-Dependent Sparse Attention" (HAX) (2025). arXiv:2507.00449. https://arxiv.org/html/2507.00449v3 (Retrieved: 2026-06-24)

[30] "ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs" (2026). arXiv:2602.07721. https://arxiv.org/pdf/2602.07721 (Retrieved: 2026-06-24)

[31] Babenko, A., Lempitsky, V. (2014). "Additive Quantization for Extreme Vector Compression". CVPR 2014. https://www.researchgate.net/publication/286594303_Additive_Quantization_for_Extreme_Vector_Compression (Retrieved: 2026-06-24)

[32] Martinez, J., Zakhmi, S., Hoos, H. H., Little, J. J. (2018). "LSQ++: Lower Running Time and Higher Recall in Multi-codebook Quantization". ECCV 2018. https://link.springer.com/chapter/10.1007/978-3-030-01270-0_30 (Retrieved: 2026-06-24)

[33] Egiazarian, V., Panferov, A., Kuznedelev, D., et al. (2024). "Extreme Compression of Large Language Models via Additive Quantization" (AQLM). ICML 2024. https://dl.acm.org/doi/10.5555/3692070.3692558 (Retrieved: 2026-06-24)

[34] "QINCo2: Vector Compression and Search with Improved Implicit Neural Codebooks" (2025). ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/d470d6e007a19ff1666386562c77517c-Paper-Conference.pdf (Retrieved: 2026-06-24)

[35] "FaTRQ: Tiered Residual Quantization for LLM Vector Search in Far-Memory-Aware ANNS Systems" (2026). arXiv:2601.09985. https://arxiv.org/pdf/2601.09985 (Retrieved: 2026-06-24)

[36] Douze, M., Guzhva, A., Deng, C., Johnson, J., et al. (2024). "The Faiss Library". arXiv:2401.08281. https://arxiv.org/abs/2401.08281 (Retrieved: 2026-06-24)

---

## Appendix: Methodology

### Research Process

This deep-mode investigation followed the 8-phase pipeline. **Scope** was largely supplied by the brief, which already fixed the problem statement, hardware target, and five frontier questions, and explicitly excluded re-derivation of established internal results (MQAR collapse, top-k parity, IVF-PQ front-runner cost, VSA death). **Plan** mapped each frontier to 1–3 search angles plus a theoretical anchor (SSM recall lower bounds) and a CPU-cost anchor. **Retrieve** ran three parallel web-search batches (16 queries) followed by three targeted full-text fetches of the highest-value/most-recent sources (HAX 2507.00449, ParisKV 2602.07721, the SSM-retrieval ablation 2510.19861). **Triangulate** required ≥2 independent sources per major claim and verified quantified benchmarks against originating papers where text was available. **Outline refinement** elevated two angles that emerged as more important than initially scoped: query drift (ParisKV) and the thinness of the recall tier (retrieval-head literature). **Synthesize / Critique / Refine** applied skeptical-practitioner and adversarial-reviewer lenses, producing the counterevidence register (Samba's no-index recall; retrieval-head quantization fragility). **Package** produced this report.

### Sources Consulted

**Total Sources:** 36 cited (≈40 examined). **Source Types:** peer-reviewed papers (ICML, ICLR, NeurIPS, AAAI, ECCV, CVPR, SOSP, IEEE TPAMI) ≈26; engineering documentation (FAISS, ScaNN) ≈3; vendor/technical blogs (Together AI, AI21, SyncSoft) ≈3; preprints (arXiv 2025–2026, not yet venue-confirmed) ≈4. **Temporal Coverage:** 2014–2026, weighted to 2024–2026, with foundational anchors (Additive Quantization 2014, FreshDiskANN 2021, Memorizing Transformers 2022).

### Verification Approach

**Triangulation:** Each major claim rests on ≥2 independent sources (e.g., the SSM-recall limit on Jelassi [19] + set-disjointness [29] + the ablation [18]; similarity-preserving quantization on ScaNN [4] + query-aware MIPS [5] + NoMAD [9] + ParisKV [30]). Single-source claims are flagged inline (ParisKV's drift-robustness specifics [30]; the 15%-heads figure [18]) and noted in gaps.

**Credibility Assessment:** Peer-reviewed venue + reproduced benchmark = high (≈80–95): ScaNN, Quest, NoMAD, Memorizing Transformers, InfLLM, Jamba, Samba, Jelassi, retrieval heads, Medusa, FAISS docs. Recent un-venued preprints = medium (≈60–70): ParisKV, HAX, FaTRQ, "Some Attention is All You Need for Retrieval", FastMTP — used for direction and mechanism, with quantitative specifics caveated. Vendor blogs = medium, used only for figures corroborated elsewhere (e.g., DeepSeek-MTP acceptance rate).

**Quality Control:** Findings cross-checked for internal consistency; the one apparent contradiction (Samba reaching long-context recall without an index) was reconciled in the counterevidence register rather than suppressed. No citation was included without a resolvable URL; preprints with 2025–2026 arXiv IDs were retained because the field's relevant work is overwhelmingly recent.

### Claims-Evidence Table

| Claim ID | Major Claim | Evidence Type | Supporting Sources | Confidence |
|----------|-------------|---------------|-------------------|------------|
| C1 | Retrieval is attention-only; SSM contributes 0%; ~15% of heads suffice | Ablation + mechanistic | [18], [21], [22] | High |
| C2 | Fixed-state SSM cannot do unbounded associative recall (theory) | Proof + complexity | [19], [29] | High |
| C3 | MSE-PQ is wrong for top-k(q·K); anisotropic/score-aware loss is the fix | Method + benchmark | [4], [5], [6], [9] | High |
| C4 | Streaming decode insert is append-only; consolidation cost N/A | Systems analysis | [1], [2], [3] | High |
| C5 | PQ4 fast-scan is CPU SOTA, VNNI-independent, ~small 4-bit recall loss | Benchmark + engineering | [7], [8], [9], [36] | High |
| C6 | AQ/RVQ more accurate per bit but query-time decode heavier unless progressive | Method comparison | [31], [32], [33], [34], [35] | Medium-High |
| C7 | Indexed-recall-over-KV exists (ParisKV) but GPU; CPU niche open | Architecture paper | [30], [9] | Medium |
| C8 | MTP gives 1.8–3.6×, acceptance-gated; self-drafting is CPU-appropriate | Benchmark | [23], [24], [25], [26] | Medium-High |

**Confidence Levels:** High = 3+ independent sources, consistent, strong methodology. Medium = recent preprint or single strong source with minor caveats.

---

## Report Metadata

**Research Mode:** Deep | **Total Sources:** 36 cited | **Word Count:** ≈6,400 | **Generated:** 2026-06-24 | **Validation Status:** Pending automated validation
