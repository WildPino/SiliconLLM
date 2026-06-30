# Scale-Up Architecture — CPU-First Large Agentic LLM (Zen2, no-VNNI)

**Status: design draft, 2026-06-30.** Consolidates the validated foundation into one buildable architecture, sizes it against the measured Zen2 budget, and marks every open unknown with the probe that resolves it. This is the blueprint the C inference engine will implement. Locked decisions are measured on the real hardware (Ryzen 5 3600X / Zen2, AVX2/FMA, no AVX-512/VNNI, DDR4 ~28-45 GB/s, L2 512KB/core, L3 16MB/CCX). Narrative history: `HANDOFF.md`. Research detail: `docs/research/`.

---

## 0. Thesis

A large, agentic-capable LLM made fast on a consumer CPU by **architectural co-design**, not a fast runtime over a stock model. One principle:

> **Split the *thinking* from the *knowing*.** Keep the compute core small, ternary, and cache-resident (reused for depth); push capacity into a large, sparse, indexed memory that lives in DRAM but is barely touched per token. Refactor so the heavy part is a **lookup**, not a **multiply** — and stream everything at ternary, through cache, sequentially.

Target: large agentic model, 128K+ context, batch-1 decode, on no-VNNI consumer silicon. "Choices today must not preclude tomorrow."

---

## 1. The bandwidth equation, with factors now MEASURED on Zen2

Batch-1 decode is memory-bandwidth-bound: `tok/s ≈ (eff_bw × tokens_per_stream) / (bytes_per_weight × active_weights_per_token)`. Four independent multipliers; each de-risked on real hardware:

| Term | Lever | Status | Measured on 3600X |
|---|---|---|---|
| bytes_per_weight ↓ | ternary 1.58-bit + pshufb-LUT | **LOCKED** (Probe-1) | kernel **4.2-5.0×** vs fp32, faster-as-bits-drop; +0.028 BPB quality; ~0.5 B/weight LUT packing |
| active_weights ↓ | gated-dReLU activation sparsity | **LOCKED** (Probe-2) | **2.12×** predictor-free shrink, ~0 quality cost (79% gate sparsity) |
| active_weights ↓ | fine-grained ternary MoE | OPEN → probe-4 | — (sized by the L3 budget below) |
| eff_bw ↑ | active-slice fits L3 | **LOCKED** (Probe-3) | **~3×** step (88→28 GB/s) at the **16MB L3 cliff**; compute-bound while resident |
| tokens_per_stream ↑ | block-decode `a` / MTP | OPEN → exec-model + probe-6 | `a`≈2-4× single-user (lit.) |

Ternary × sparsity already compose to **~21× fewer MLP bytes/token vs fp32-dense** at ~+0.03 BPB. The honest compound across all terms is **one-to-two orders of magnitude over dense-int4** — *not* a naive product; do not anchor on a single number.

**The ρ-law (measured, foundational):** random-DRAM access vs sequential-cache-resident = **~14× worst-case** on Zen2 (~2× in-cache, ~4.3× in DRAM). **Rule: every byte path is sequential and cache-resident; no unnecessary random gather.** This single law killed SIMVQ, vector-codebook weight-quant, ANN graphs, and IMI — and it is why the recall tier uses compact in-cache codes, never raw-vector gather.

---

## 2. The keystone design constraint (the number that sizes everything)

**Active per-token weight slice ≤ 16 MB (L3), comfortably ≤ 8-12 MB (high plateau).** At the real LUT packing (~0.5 B/weight), that is:

> **~24-32M active ternary params per token** (the compute-bound ceiling).

Everything large-model-specific is sized by this: the MoE granularity, the sparsity targets, the head budget, and the reuse depth must all keep the per-token *active* set under ~24-32M ternary params so the model stays in the L3 compute-bound regime (where the Probe-1 4-5× kernel speed is realized). Spill L3 → DRAM-bound → the kernel speed stops mattering. **This is the architecture's hard wall, measured on the target silicon.**

---

## 3. The architecture (component map)

```
   ┌─────────────────── THE THINKING (small, ternary, cache-resident) ──────────────────┐
   │  SSM backbone (selective/diagonal, HiPPO) — O(1) state, bandwidth-light             │
   │  + gated-ternary MLP (dReLU-sparse) — the byte-sink, attacked                       │
   │  + depth via REUSE of one cache-resident block (adaptive halting) — depth, no bytes │
   └───────────────┬──────────────────────────────────────────────────┬─────────────────┘
       SSM state (compressed running summary; computed early & cheap)  │ predicts active set
                   │                                                   ▼
   ┌───────────────▼─────────────── THE KNOWING (large, sparse, in DRAM, barely touched) ┐
   │  Recall tier: SSM-state query → IVF-PQ two-stage index (InfoNCE rep + Hadamard part) │
   │  Capacity: product-key / fine-grained ternary MoE — large N, active slice ≤ budget   │
   │  All access SEQUENTIAL + cache-resident codes (ρ-law); compressed head protects L3   │
   └──────────────────────────────────────────────────────────────────────────────────────┘
   Execution: block-decode layer-major, double-buffered, per-position state-checkpoint.
   Decode hygiene LOCKED: greedy + rep-penalty 1.2/win128, no top-p (loop-safe; = linear-verify asset).
```

### 3.1 Backbone — SSM (LOCKED)
Selective state-space (diagonal recurrence + HiPPO init), O(1) recurrent state, no growing KV. Bandwidth-light by construction; the bytes concentrate in the MLP/head, which is where the attack lands. The SSM state is also the *prefetch/predict oracle* (§3.6) — a compressed running summary computed early and cheaply each token.

### 3.2 MLP — gated, ternary, sparse (LOCKED mechanism)
Gated (SwiGLU-style) MLP — **not** the current plain `up→SiLU→down`. The gate-first structure lets gate-sparsity skip up+down predictor-free (Probe-2: 2.12×). Weights native 1.58-bit ternary via pshufb-LUT (Probe-1). dReLU activation for high native sparsity. *Note:* gated adds the gate matrix (more params) but they are ternary and the active set shrinks — net per-token active bytes win. Plain-vs-gated final call pending the predictor (§3.6).

### 3.3 Depth via reuse (K2, OPEN → probe)
One cache-resident ternary block reused N times for depth, with adaptive halting, instead of N distinct streamed layers. "Buy depth without paying bandwidth." Probe: weight-tied-reuse vs untied param-matched (BPB + the depth/quality curve), then halting. Open.

### 3.4 Recall tier (LOCKED — Phase 56 closed)
SSM-state queries → **IVF-PQ two-stage** recall slot: 4-bit ADC shortlist + exact rerank of top-16. ~18 µs/tok/layer @128K (dim=128), under the 30 µs gate. **Originality = the InfoNCE *representation*** (load-bearing: reaches the attention ceiling, frozen-LSH plateaus at 84%). **Partition = data-independent Hadamard** (≥ learned, drift-proof by construction → no streaming rebalancer, no drift-recalibration). Drift is a non-problem on Mamba-1 (bounded state-norm; recall flat-vs-distance to 21× context). The dense-O(T²) training harness can't reach 128K, but inference uses the sparse index; a sparse-slot rewrite is the engineering prereq to *train* the recall tier at long context (§5).

### 3.5 Capacity — sparse memory / fine-grained ternary MoE (K1, OPEN → probe-4)
Grow total params N while keeping `active × N` small: product-key memory and/or fine-grained ternary MoE (small experts + always-resident shared experts), with the active expert set sized to ≤ the §2 budget. The published cousins (DeepSeek-MoE, UltraMem) exist; the open piece is co-designing granularity against the consumer-L3 budget under ternary, with the experts served by the pshufb-LUT kernel (AMX/VNNI paths unavailable). Probe-4: mini-MoE, measure active-expert cache hit-rate + routing locality vs the budget.

### 3.6 Predictability — the control system (K3 / Finding-7, OPEN — highest-originality)
A tiny cache-resident predictor maps the SSM state → the next active set (which experts/rows), enabling **SKIP** (don't stream the inactive — pure bandwidth saving, silicon-independent, the durable mode) and secondarily **prefetch** (Zen2-only, fragile). The decisive, unpublished move: **co-train the model to *be* predictable** (a predictability/routing-coherence regularizer) — convert the speculative cache-residency bet into an optimizable training objective. Attaches to the *irregular* tier (K1 gather / MoE routing), not the regular SSM scan (already HW-prefetched). Make-or-break first measured cheaply: how predictable is the active set from the SSM state, and does the regularizer raise it without quality cost. Open.

### 3.7 Head — compressed (OPEN → probe-5)
At target vocab (32K-128K) the output head can eat the entire L3 budget. Tie input/output embeddings; factorize / cluster (adaptive-softmax) / VQ the head so a large vocab does not break cache-residency. Probe when vocab scales.

### 3.8 Execution model — block-decode chassis (R-F/G/H, OPEN main-loop)
Layer-major block-decode is the roofline keystone: process N positions per weight-load → arithmetic intensity ∝N → bandwidth-bound becomes compute-bound (where multi-core and §2 cache-residency finally pay). Honest single-user ceiling = acceptance length **`a`≈2-4×** (speculative), more on structured/low-entropy **Cat-A** content (code/log/JSON) where conditional-independence allows wider parallel commits — *the engine is fastest exactly where the agentic product operates*. SSM-native: the parallel scan **is** the block-verifier (O(N), no KV). **Crux = per-position state-checkpoint for partial-accept rollback** (the #1 C-kernel risk). **Main-loop OPEN:** speculative-AR (keeps the locked decode-hygiene via linear verify; needs state-checkpoint) **vs** carve/SBD (re-scan, *dissolves* the rollback crux, but the confidence-ordered unmasking re-opens the loop-hygiene question). Likely content-adaptive. The a-priori "plan" (R-G) is **throughput infrastructure** (parallel width for the single-user), not a quality lever (AR plans internally already) — discrete cheap skeleton only, separable content only.

### 3.9 I/O — entropy-patched (K4, optional, SEE DNA)
Enter/exit at compression-unit boundaries (Byte-Latent-Transformer-style patching), the project's historical compression soul. Medium lever, late integration.

---

## 4. Open unknowns → probe map

| Unknown | Probe | Cost | Why it matters |
|---|---|---|---|
| MoE cache-hit / routing locality | probe-4 (mini-MoE) | mid (small train) | sizes capacity vs the §2 budget |
| Predictability of active set + regularizer | Finding-7 probe | cheap-ish | turns cache-residency from bet to lever; highest originality |
| Reuse depth vs quality (K2) | depth probe | cheap (sandbox) | depth-without-bandwidth |
| Head at vocab 32-128K | probe-5 | cheap | protects L3 budget |
| MTP acceptance on real content | probe-6 | cheap | `tokens_per_stream` closer |
| Main-loop: spec-AR vs carve + `c` | exec-model probes (C) | cheap | the C-engine core loop |
| Train recall tier at long ctx | sparse-slot rewrite | engineering | prereq for the large model |

---

## 5. Training-cost lever: distillation, not from-scratch-blind

From-scratch is the unfair advantage (dissolves the PTQ ~2.4-bpw floor: native ternary near-lossless, native sparsity improves scores). But from-scratch at *large* scale is a frontier-budget program. The cost lever: **distill an open-source teacher into this ternary-SSM-recall architecture** (transformer→SSM distillation is proven cheap — Mamba-in-the-Llama, ~tens of B tokens vs trillions). This is exactly the **QAD** path already identified (QAT-ternary + logit-distillation): the teacher feeds the from-scratch ternary student, composing — it doesn't abandon the unfair advantage, it sources its knowledge cheaply. A stock model **cannot** run on this engine (transformer ≠ SSM operators; and it lacks the ternary/sparse/cache properties); but it can *teach* it.

---

## 6. Originality (honest)

Not the pieces — every brick is published (BitNet ternary, T-MAC LUT, TurboSparse dReLU, DeepSeek-MoE, product-key memory, IVF-PQ, block-decode, distillation). The originality is the **fusion** and the **target**: three systematically-empty niches, complementary —
1. **"active-per-token slice ternary and L3-resident on a no-VNNI consumer CPU"** (the cache-residency co-design, Probes 1-3),
2. **indexed sublinear recall at 128K on CPU** with a representation/data-independent-partition split (Phase 56),
3. **SSM × few-step set-block decode** (the execution model),
plus the unpublished **predictability-as-trained-objective** (§3.6). Honest: a recomposition of real parts at a point nobody has occupied, not new physics.

---

## 7. Roadmap

Foundation validated (recall tier + 3 bandwidth multipliers, measured) → **this consolidation** → remaining probes in context (predictor / MoE / depth / head / MTP / exec-model main-loop) → freeze architecture → build the C engine on the block-decode chassis → scale-up training (distilled). Nothing here is committed code; it is the design the engine implements.
