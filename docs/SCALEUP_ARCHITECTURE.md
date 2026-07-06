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
| active_weights ↓ | fine-grained ternary MoE | **VALIDATED** (probe-4) | quality ≥ dense at matched total params; experts = DRAM-streamed pool, no residency bonus (§3.5) |
| eff_bw ↑ | active-slice fits L3 | **LOCKED** (Probe-3) | **~3×** step (88→28 GB/s) at the **16MB L3 cliff**; compute-bound while resident |
| tokens_per_stream ↑ | block-verify chassis (n-gram draft) | **MEASURED, CONDITIONAL** (Phase 63) | mechanism proven: streamed bytes/token = AR ÷ tpp exactly (tpp 1.75 code / 2.6-2.8 TS); pays only where *shared*-streamed traffic dominates compute (**C/T ≲ 0.25-0.30**; at 8.3M C/T=1.6 → scoped out); routed/expert pools do **not** amortize (i.i.d. unions, §3.8) |

Ternary × sparsity already compose to **~21× fewer MLP bytes/token vs fp32-dense** at ~+0.03 BPB. The honest compound across all terms is **one-to-two orders of magnitude over dense-int4** — *not* a naive product; do not anchor on a single number.

**The ρ-law (measured, foundational):** random-DRAM access vs sequential-cache-resident = **~14× worst-case** on Zen2 (~2× in-cache, ~4.3× in DRAM). **Rule: every byte path is sequential and cache-resident; no unnecessary random gather.** This single law killed SIMVQ, vector-codebook weight-quant, ANN graphs, and IMI — and it is why the recall tier uses compact in-cache codes, never raw-vector gather.

---

## 2. The keystone design constraint (the number that sizes everything)

**Active per-token weight slice ≤ 16 MB (L3), comfortably ≤ 8-12 MB (high plateau).** At the real LUT packing (~0.5 B/weight), that is:

> **~24-32M active ternary params per token** (the compute-bound ceiling).

Everything large-model-specific is sized by this: the MoE granularity, the sparsity targets, the head budget, and the reuse depth must all keep the per-token *active* set under ~24-32M ternary params so the model stays in the L3 compute-bound regime (where the Probe-1 4-5× kernel speed is realized). Spill L3 → DRAM-bound → the kernel speed stops mattering. **This is the architecture's hard wall, measured on the target silicon.**

**Second wall (measured, Phase 63): the per-position compute floor.** The keystone bounds *bytes*; it does not bound the elementwise per-position compute (scan recurrence, activations) — at the 8.3M config this floor is ~1.1-1.3 ms/token and *is* the current tok/s; ternary does not shrink it, and it grows with D×N×L. The Phase 64 design gate must budget both walls together (it is this floor that scoped out the block-verify speedup at 8.3M, §3.8).

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
   Execution: AR-lossless main-loop + threads (2.4× measured); block-verify chassis built &
   lossless (Activation Replay commit) — engaged only in shared-streamed-dominated configs (§3.8).
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

### 3.5 Capacity — sparse memory / fine-grained ternary MoE (K1, VALIDATED — probe-4, 2026-07-02)
Grow total params N while keeping `active × N` small: fine-grained ternary MoE (small experts + always-resident shared parts), served by the pshufb-LUT kernel. **Probe-4 verdict (matched arms, 4k steps, equal total params ~22.5M):** quality gate smashed — MoE-gran (E32×h128, top-8, active 1024) BPB **0.8589** vs dense-1024 0.8799 (gate ≤0.8899) *and* vs dense-4096 0.8674: the granular MoE beat the dense upper-bound arm at matched total params; granular > coarse (E8×h512 top-2 = 0.8637). Router healthy under Switch aux (0 dead, max/mean ≤1.47×). **Routing locality FALSIFIED (the pre-registered red flag fired):** expert working-set over a block ≈ i.i.d. baseline (84-89% of E at N=8 vs 90% random; ~100% by N=16-32) and expert persistence = base-rate k/E — the temporal-independence finding of Phase 58 replicates at expert granularity. **Consequence: the expert tier is re-classified from "L3-warm pool" to "DRAM-streamed, granularity-bounded".** It loses the ~3× residency multiplier (backbone/router/head keep it); bandwidth stays cheap (a ternary h=128 expert is tens of KB; top-k × layers = a few MB/token at the 28 GB/s floor — not the bottleneck) and ρ-safe (tens-of-KB *contiguous* bulk loads per block, not 64B-grain gather). **The §2 budget becomes explicitly two-pool: resident (≤16MB, reused, ~90-116 GB/s) + streamed (experts, ~28 GB/s, costed in bytes/token).**

### 3.6 Predictability — the control system (K3 / Finding-7, OPEN — highest-originality)
A tiny cache-resident predictor maps the SSM state → the next active set (which experts/rows), enabling **SKIP** (don't stream the inactive — pure bandwidth saving, silicon-independent, the durable mode) and secondarily **prefetch** (Zen2-only, fragile). The decisive, unpublished move: **co-train the model to *be* predictable** (a predictability/routing-coherence regularizer) — convert the speculative cache-residency bet into an optimizable training objective. Attaches to the *irregular* tier (K1 gather / MoE routing), not the regular SSM scan (already HW-prefetched). Make-or-break first measured cheaply: how predictable is the active set from the SSM state, and does the regularizer raise it without quality cost. Open.

### 3.7 Head — compressed (OPEN → probe-5)
At target vocab (32K-128K) the output head can eat the entire L3 budget. Tie input/output embeddings; factorize / cluster (adaptive-softmax) / VQ the head so a large vocab does not break cache-residency. Probe when vocab scales.

### 3.8 Execution model — MEASURED (E5 design gate + Phase 63 engine, 2026-07-05): AR-lossless + threads; block-verify built, CONDITIONAL
The 2026-06 design imagined two lanes; both were measured. What survived:
- **Multi-token generation from state (carve/set-block AND spec-AR with learned heads) is CLOSED at this scale (model-side E5 gate, pre-registered rules held):** the hidden state carries ~1 reliable lookahead step *even when co-trained to carry more* (E5.1 frozen: t+2 top-1 16%, 1.96 tokens/pass < the ≥2.0 rule; E5.2 co-trained carve: t+2 = 2.376 vs t+1 = 1.245 bits/byte with the t+1 anchor intact — predicting t+2 must marginalize over the uncommitted t+1; that gap is true conditional entropy). **The model stays AR-lossless.**
- **Block-verify chassis (zero-cost n-gram draft + linear verify): BUILT and correctness-proven in the C engine (Phase 63).** Token-identical to AR everywhere tested (lossless by construction, locked hygiene preserved). The state-checkpoint crux flagged here as the #1 C-kernel risk is **retired by Activation Replay**: stash per-position post-GEMV recurrence inputs (~28 KB/position) during the speculative pass; commit = elementwise recurrence replay, zero GEMV, no weight re-touched. Weights-once proven by counting: streamed bytes/token = AR ÷ tpp exactly (K ∈ {2,4,8}). Free drafter figures: 1.75 tokens/pass on code (order-5, saturates), 2.6-2.8 on TS prose.
- **The speedup is CONDITIONAL and currently scoped out (pre-registered outcome):** all-weights-cold emulation at 8.3M gives 0.68× best — the per-position compute floor (1.14 ms, fp32 scan) exceeds the weight traffic (0.71 ms). **gain(K) = (T+C)/(T/tpp + C·K/tpp); the lever pays iff C/T ≲ 0.25-0.30.** A scale-up design that wants it must buy ~10× more *shared* streamed bytes/token AND a several-times-cheaper resident position cost (ternary resident + scan optimization). **Roofline caveat (audit 2026-07-05): the formula is optimistic — measured gains ran 0.61× (K2) to 0.74× (K8) of it (AR hides traffic under compute: measured AR-cold 1.305 ms vs T+C = 1.85 ms; block pays stash/replay/rotation overheads growing with K) → C/T ≲ 0.25-0.30 is *necessary, not sufficient*; working threshold ≈ 0.15-0.20, to be re-measured at any candidate design point.**
- **Routed pools do NOT amortize (new law, Phase 63):** block amortization of a routed class = topk·tpp/union, and measured unions match pure i.i.d. E·(1−(1−topk/E)^K) (routing≈i.i.d., third independent confirmation) → break-even at best (0.96×/0.86× at K4/K8), *worse* with granularity. **Block-verify helps only SHARED streamed weights; the expert pool's levers remain threads and §3.6 predictability.**
- **Threads (adopted): 2.45× dense / 2.39× MoE at 6 cores**, bit-identical by construction (output-independent loops only, reductions serial); scaling = bandwidth aggregation (spread ≥ close on Zen2 2-CCX — locality lost to aggregate bandwidth); DRAM-streamed ceiling ~1.4-1.5×; SMT regresses.
Blueprint consequence: the §3.5 two-pool as drawn (expert-dominated streamed class) does **not** benefit from block-verify; the scale-up design gate must either provision a shared streamed class that clears the C/T win-condition or leave the lever parked (the chassis stays in the engine behind `--block K`, default off, zero cost on the AR path).

### 3.9 I/O — entropy-patched (K4, optional, SEE DNA)
Enter/exit at compression-unit boundaries (Byte-Latent-Transformer-style patching), the project's historical compression soul. Medium lever, late integration.

---

## 4. Open unknowns → probe map

| Unknown | Probe | Cost | Why it matters |
|---|---|---|---|
| MoE cache-hit / routing locality | probe-4 — **DONE 2026-07-02** | — | locality ≈ i.i.d. (no hot pool) → expert tier re-classified DRAM-streamed; quality + router gates passed (§3.5) |
| Predictability of active set + regularizer | Finding-7 probe | cheap-ish | turns cache-residency from bet to lever; highest originality |
| Reuse depth vs quality (K2) | depth probe | cheap (sandbox) | depth-without-bandwidth |
| Head at vocab 32-128K | probe-5 | cheap | protects L3 budget |
| MTP acceptance on real content | probe-6 — **DONE 2026-07-04/05 (E5.0/E5.1/E5.2)** | — | n-gram floor 1.75 code / >2-3 prose; learned MTP-frozen 1.96 < 2.0 rule; co-trained carve failed its gate → multi-token-from-state CLOSED at this scale, model stays AR-lossless (§3.8) |
| Main-loop: spec-AR vs carve + `c` | exec-model — **DONE 2026-07-05 (Phase 63)** | — | block-verify chassis built + lossless (Activation Replay); speed conditional on C/T ≲ 0.25-0.30, routed pools don't amortize; adopted main-loop at this scale = AR + threads (2.4×) |
| Train recall tier at long ctx | sparse-slot rewrite | engineering | prereq for the large model |

---

## 5. Training-cost lever: distillation, not from-scratch-blind

From-scratch is the unfair advantage (dissolves the PTQ ~2.4-bpw floor: native ternary near-lossless, native sparsity improves scores). But from-scratch at *large* scale is a frontier-budget program. The cost lever: **distill an open-source teacher into this ternary-SSM-recall architecture** (transformer→SSM distillation is proven cheap — Mamba-in-the-Llama, ~tens of B tokens vs trillions). This is exactly the **QAD** path already identified (QAT-ternary + logit-distillation): the teacher feeds the from-scratch ternary student, composing — it doesn't abandon the unfair advantage, it sources its knowledge cheaply. A stock model **cannot** run on this engine (transformer ≠ SSM operators; and it lacks the ternary/sparse/cache properties); but it can *teach* it.

---

## 6. Originality (honest)

Not the pieces — every brick is published (BitNet ternary, T-MAC LUT, TurboSparse dReLU, DeepSeek-MoE, product-key memory, IVF-PQ, block-decode, distillation). The originality is the **fusion** and the **target**: three systematically-empty niches, complementary —
1. **"active-per-token slice ternary and L3-resident on a no-VNNI consumer CPU"** (the cache-residency co-design, Probes 1-3),
2. **indexed sublinear recall at 128K on CPU** with a representation/data-independent-partition split (Phase 56),
3. **SSM × few-step set-block decode — now occupied by MEASUREMENT, answer negative at this scale** (E5.2 + Phase 63): the state carries ~1 lookahead step even co-trained; the chassis is built, lossless and weights-once-proven, but pays only where shared streamed traffic dominates compute (§3.8). First measured result on pure-SSM set-block either way — the niche is *mapped*, no longer just empty,
plus the unpublished **predictability-as-trained-objective** (§3.6) — after Phase 63 this is the highest-value open originality claim. Honest: a recomposition of real parts at a point nobody has occupied, not new physics.

---

## 7. Roadmap

Foundation validated (recall tier + 3 bandwidth multipliers, measured) → **this consolidation** → **C engine BUILT and gate-sealed through Phase 63** (E1-E4 + threads adopted ~2.3-2.45×; AR-lossless main-loop; block-verify chassis optional, conditional) → remaining probes (predictability/Finding-7, depth-reuse K2, head probe-5, recall long-ctx) feed the **Phase 64 scale-up design gate** (§8) → scale-up training (distilled). The engine is committed code; the rest is the design it implements.

---

## 8. Phase 64 — the scale-up design gate (OPENED 2026-07-05)

**The question this gate answers:** what sized, trainable, portable design does the product thesis (CPU-native *agentic code* model) commit to — and is its predicted performance worth the training spend? **Output:** this document upgraded from blueprint to *frozen spec*, plus a costed training plan. The go/no-go on the training spend is the owner's.

**Method (the E5-gate discipline, mirrored):** desk-first; measurements only where a decision is blocked; every decision criterion pre-registered *before* its numbers; one variable per stage; **push-before-run in its falsifiable form** (protocol + apparatus pushed before any measurement they govern — restated after the Phase 63 slip); all hardware numbers parametric (Portability law: LLC, bandwidths, topology are runtime/config inputs; the 3600X is the reference instance, not the target).

**Stage 64.0 — ground-truth refresh (engine exists, no GPU, no gates — instrumentation only):**
- **(a) Streamed thread ceiling** (the audit's derived-not-measured input): AR-cold emulation (`--g3c` machinery) at threads {1, 2, 3, 6} — effective cold-stream GB/s per thread count. Closes the "~1.4-1.5×" assumption with a measurement; also re-prices E4's @28 GB/s figures (rotating-stream measured 22 GB/s single-thread). **Binding protocol constraint (found in the apparatus smoke, sealed before the registered run): the emulation buffer must be ≫ total L3 — `--emu-mb 128` on the reference (a 32 MB buffer returned 64 GB/s at 3 threads = an L3-resident artifact, not a DRAM ceiling: the ρ-law cache cliff applied to the measuring instrument itself). The pinning configuration used (none/close/spread) is reported alongside the numbers.**
- **(b) Compute-floor decomposition:** per-component time breakdown of the current engine (scan recurrence / projection GEMVs / LUT-MLP / head / norms-glue), dense + MoE, threads {1, 6}, end-to-end protocol — the empirical basis for extrapolating C(model dims) in 64.1. Current anchor: C ≈ 1.14 ms/token at Dn·N·L = 512·96·6.

**Stage 64.1 — the budget document (desk, Architect):** the two §2 walls as *functions*: W1 bytes (active slice vs LLC); W2 compute — **per-component scaling from the 64.0b decomposition, not a single Dn·N·L factor** (the apparatus smoke already corrects the earlier shorthand: C is *not* scan-dominated — proj-GEMV ≈53% [fp32, ∝ Dn·D·L, memory-bound, thread-scaled 2.45×], LUT-MLP ≈24% [∝ active ternary bytes], scan recurrence ≈13% [∝ Dn·N·L], SWA/head/norms the rest — each term extrapolated by its own dims and its own thread factor); traffic T (streamed bytes/token ÷ measured cold GB/s at the operating thread count). Produces the **sizing-vs-tok/s curve** on the reference class for candidate designs (two-pool splits, MoE configs, vocab sizes), every assumption listed. **Owner-fixed parameters (asked and sealed 2026-07-05):** (P1) product speed — **hard floor 10 tok/s** on the reference class, target band 20-50, exact point picked from the curve (floor pre-registered as the anti-goalpost guardrail); (P2) **total footprint ≤ 16 GB** (target class = 32 GB-RAM x86-64-v3 machines; the ≤8 GB variant is carried as a row on the curve for a possible smaller SKU); (P3) training resource — **2× T4 at ~90 h/week** (fp16 — Turing, no bf16), high patience, cash top-up possible: practical student ceiling ~100-500M total params over weeks-to-months. The 64.4 cost table prices token budgets against this quota, and must price QAD's teacher-inference cost explicitly (precomputed-logits vs online-teacher) plus a 2-GPU data-parallel validation before any long run (never used in this project so far).

**Stage 64.2 — design decisions, each against a criterion pre-registered before its numbers:** two-pool split & MoE config (i.i.d. union law and granularity bounds priced in); vocab & head (per-domain BPE at code scale; probe-5 only if vocab must grow); block-verify in/out (**C/T ≤ 0.15-0.20 operational threshold on the candidate's *shared* streamed class** — already sealed, §3.8); depth-reuse (K2) in/out; recall slot in/out of the *first* scale-up model; predictability/SKIP (Finding-7) load-bearing or deferred.

**Stage 64.3 — probes only where a 64.2 decision is blocked** (sandbox-scale GPU, owner launches after MM push, per-probe pre-registered gates). Candidates, engaged only on demand: Finding-7 predictability-regularizer probe (highest-originality open claim), K2 depth-reuse, head probe-5, recall-slot integration smoke.

**Stage 64.4 — frozen spec + costed training plan:** QAD teacher shortlist (Researcher brief: license-clean code teachers, distillation terms, tokenizer implications), data plan per the Phase 62 rules (pinned licensed code corpora; logs eval-only), token budget, GPU cost table → **STOP: the owner decides the spend.**

Carry-overs folded in: code-checkpoint engine export + parity golden (engine-side demo, P64 prep); n-gram singleton pruning (asset hygiene); record-only popularity-skew measurement attached to whatever scale-up training runs.
