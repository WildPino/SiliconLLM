# The Inventor — 00: The Big Picture (2026-07-17)

**Role:** inventive analysis of the whole project — looking for where the architecture can be improved by substituting, fusing, splitting, inverting. **Discipline:** every idea here is *desk-level, not measured*, and NOTHING touches the frozen v1 spec or the MVE run in progress. This is the inventory of light-bulbs; selection and priorities come later, with the owner.

---

## 1. The project in one sentence

An LLM co-designed for the consumer CPU (x86-64-v3, no VNNI): **the thinking** (ternary SSM, cache-resident, reused) split from **the knowing** (MoE + indexed recall, streamed from DRAM, barely touched) — with a bit-exact C engine as the product, and an experimental method (pre-registration, gates never loosened, end-to-end parity) as the epistemic asset.

## 2. Macro-areas (the map)

| # | Area | Where it lives | Status |
|---|---|---|---|
| **A** | **C engine** — ternary pshufb-LUT kernel, fast-exp, exact skip, MoE two-pool, threads, block-verify, 7/7 bit-identical oracle chain | `benchmarks/phase60/engine.c`, `phase63/`, `bin/` | consolidated, 848/701 tok/s @8.3M; engine-v2 queue (expert dispatch ~8.4µs, SIMD ADC, denser trit-pack) |
| **B** | **Measured physics** — the laws: ρ-law ~14×, L3 cliff at 16MB (now: aggregate-LLC slope), routing≈i.i.d. (3 confirmations), block-verify shared-only, per-position compute floor, fp32 organs precision-hungry | `SCALEUP_ARCHITECTURE.md`, `PHASE64_BUDGET.md` | FROZEN SPEC v1; **constraint inversion**: speed slack 12-130×, the real walls are training/data/quality |
| **C** | **Training & ladder** — MVE 22M → S0 30M → S1 105M → S2 206M; curriculum A→F (span chain-rule KD → QAT → MoE upcycle + recall → reverse-KL); Kaggle ×4, 3060, 5090 window | `PHASE64_TRAINING_PLAN.md`, `benchmarks/phase64/mve/` | **RUN IN PROGRESS** — do not touch |
| **D** | **Recall tier ("the knowing", indexed)** — InfoNCE + data-independent Hadamard partition, IVF + 4-bit ADC, 29µs @128K | `phase64/recall_probe.c`, `mve_model.py:RecallSlot`, P55/56 lineage | IN v1, clause-2 armed at the MVE |
| **E** | **Data & eval** — per-domain BPB canon, pinned CPython code-val, mandatory P62 decontamination, Stack-v2 via SWH-S3 | `docs/CANONICAL_EVAL.md`, data brief | HF-token blocker for rung-1 |
| **F** | **Compressor DNA (dormant)** — SEE, range coder, symbolic-expert MoE, LZ, regime router: the entire Phase 24-40 era | `archive/`, historical `src/` | archived at the Phase-40 pivot — **but it is working machinery, not just history** |
| **G** | **Method** — prereg push-before-run, one-variable-per-stage, end-to-end parity (E4 law), MM/commit discipline | spread across the plans | the real reproducible unfair advantage |

## 3. What I see looking at EVERYTHING at once (cross-cutting observations)

**O1 — The project has two eras and treats them as separate; they are not.** The compressor era (F) built exactly the tools the LLM era now wants: online per-byte entropy estimation, deterministic range coding, a regime router, ~zero-cost symbolic experts. The LLM era lives in a *bandwidth-bound* regime where trading compute for bytes is the right trade. Sparks S2, S3, S7 below are all born from this fusion.

**O2 — The constraint inversion (64.1) changes the very definition of "improvement".** With speed slack at 12-130×, an idea that *costs* tok/s but *buys* quality-per-param or footprint bytes is now a good idea — the exact opposite of the hierarchy six months ago. Half of the project's mental queue ("make it faster") is pricing in the wrong era; the right queue is "more quality/capacity inside the same footprint, and cheaper bytes on the road to 10B" (where bandwidth bites again).

**O3 — The residual compute wall and the byte wall share one name: the fp32 projections.** P61 measured that ternarizing them costs +0.018-0.022 BPB → they stay fp32. But P61 tested *one* alternative only (less precision). The orthogonal axis — **less structure** at full precision — has never been measured. See S1: the spark I rate biggest.

**O4 — "Routing ≈ i.i.d." was filed as a negative; mathematically it is also a license.** If no temporal locality exists, then the *layout* of the experts is a completely free degree of freedom: the pool can be re-encoded, delta-compressed, re-ordered without losing anything (there was nothing to lose). Every measured negative is one fewer constraint on the design. See S3.

**O5 — The curriculum has already discovered (the hard way) a general principle it never stated:** the magnitude-matched upcycle law (+0.49 → +0.0056 BPB) and the RecallSlot's zero-init gate are the same theorem: **every curriculum switch must be an ε-identity of the function at switch time**. But stage D (QAT hot-swap) *violates* this principle: ternarizing in one shot is NOT an identity. See S5 — the most immediately actionable idea on the ladder (after the MVE, never during).

**O6 — The recall slot and the MoE router are the same mathematical operator written twice.** Both: query from the state → top-k over a key set → weighted sum of values. One reads the context's past, the other selects static weights. At E256 (S2) the router is one step from *needing* the same index infrastructure as the recall tier (Hadamard-IVF). Unifying them is not aesthetics: it is one C code-path, one scaling law, and it opens the door to "fast weights" (retrieved values = weight deltas). See S6.

## 4. The sparks (inventory, in order of fervor)

### S1 — Projections: attack the STRUCTURE, not the precision (the wrong wall was tested)
The SSM projections (in/x/dt/out) are 52.7% of engine time and a large slice of the resident bytes. P61 tried ternary → quality fail. But there is a family of **fp32-exact, near-zero-byte matrices**: structured transforms of the form `W ≈ D₂·H·D₁·H·D₀` (diagonal × Hadamard sandwich, Fastfood/ACDC/butterfly lineage). The Hadamard transform **has no weights** (it is a ± recursion, O(D log D), branchless, AVX2-perfect — and the project ALREADY uses it in the recall tier as the data-independent partition); what remains is O(D) fp32 diagonal parameters. Effect: projection bytes go from O(D²) to O(D), compute from O(D²) to O(D log D), full precision (the P61 verdict "precision-hungry organs" is respected — the *shape* changes, not the bits). Attacks BOTH §2 walls in one move. Cheap test: sandbox-scale A/B, pre-registered BPB gate, same P61 discipline. Honest risk: the lost expressivity might cost as much BPB as ternary did — but it is an axis *never measured*, and the butterfly literature suggests mixing layers tolerate structure far better than they tolerate quantization.

### S2 — Entropy-coded weights: the compressor compresses the model (fusing the two eras)
Ternary weights have entropy ≤ 1.58 bits, and at the measured sparsity rates (79-92% zeros on *activations*, plus the natural weight sparsity post-QAT) the real entropy of the expert pool is plausibly ~1 bit/weight or less. Today's pack is 4 bits/weight (2.4× off the ideal — the 5-trits-per-byte pack is already queued). The extra inventive move: **stream-decode with a fixed-table decoder** (static rANS or fixed-length block codes, deterministic, SIMD-able) as the bytes arrive from DRAM. In the streamed regime at 22-36 GB/s the decode is free if it costs less than the byte-time saved — and the constraint inversion says compute is slack. Toward 10B the streamed bytes ARE the product; this is a ~2-4× multiplier on pool bytes, composable with everything else. The range coder and the determinism discipline already live in-house (era F).

### S3 — Delta-coded expert pool (the i.i.d. license)
Corollary of O4+S2: since routing is i.i.d., ANY representation of the pool can be chosen. Experts stored as **ternary residuals against a shared centroid expert** (or per-cluster centroids): the centroid is resident (a few KB, reused by everyone = it amortizes as a *shared* weight — which is the only class block-verify and the cache reward!), the residuals are sparser → lower entropy → S2 bites harder. Note the perverse synergy: the shared part is exactly the weight class that ALL the project's measured laws favor (block-verify amortizes shared only; residency pays only for what is reused). This *moves mass from the disfavored regime to the favored regime by mathematical construction*.

### S4 — The silicon-native tokenizer: constant-entropy patches (K4, home-grown)
K4 (entropy-patched I/O, BLT-style) sits in the blueprint as "medium lever, late integration". Observation: the project already owns an online, deterministic, ~zero-cost per-byte entropy meter — the SEE. A "tokenizer" whose units are **constant-entropy patches decided by the SEE** (bit budget per patch, not characters per token) would give: more informative units for BPB (the project's canonical metric, not by accident), adaptive fertility on code (whitespace/boilerplate → long patches for free), and a clean originality claim ("the compressor is the tokenizer"). It conflicts with the sealed D3/vocab for v1 → it is v2 research, but the prototype is cheap: re-segment the code-val with the SEE and measure the current model's BPB over the new units.

### S5 — Curriculum: every switch is an ε-identity → gradual QAT (α-scheduling)
State the law from O5 and apply it to the one switch that currently violates it: stage D. Instead of the hard ternary hot-swap, `W_eff(α) = (1-α)·W + α·ternary(W)` with α: 0→1 over a few thousand steps (STE unchanged; α is a scheduled scalar, zero new parameters). Falsifiable prediction: the transition-BPB at the C→D switch becomes continuous just as the E1 switch did after magnitude-matching, and the "divergence at the QAT switch" risk (a risk line in the plan's §8!) dissolves by construction. Implementation cost: ~10 lines. To be proposed as a *post-MVE* amendment (MVE gate iv measures exactly the KD→QAT stability: if it passes clean, S5 stays in the pocket; if it shows a shock, S5 is the fix already written).

### S6 — One "knowing" operator: unify recall and router (and fast weights)
Formalize the recall slot and the MoE router as instances of the same operator `lookup(q; K, V, k)` over a Hadamard-partitioned key space. Practical consequences: (a) a single C implementation (the recall tier's 4-bit ADC also serves routing once E grows past full-scan); (b) a single scaling law to measure; (c) the door to **fast weights**: values retrieved from the context can be *low-rank deltas on the MLP weights* instead of an additive residual on the state — "the context temporarily writes into the knowing". That is the dream-big version; the immediately measurable version is just (a)+(b).

### S7 — Draft model from era F: the n-gram drafter is a SEE expert in disguise
Block-verify is parked by the C/T law, fine. But if it ever re-enters (at 10B the shared bytes dominate again — and S3 *deliberately grows* the shared class), the order-5 drafter (15.8MB, tpp 1.75 on code) is exactly an LZ/TOKPFX expert from the compressor era. The symbolic MoE of that era (LZ+TOKPFX+BI+UNI with online fixed-share) is a *regime-adaptive* drafter already written, deterministic, RAM-latency footprint. Synergy recorded, no action now.

### S8 — (Micro, engine) Fuse the 4 projection GEMVs — **VERIFIED NEGATIVE (2026-07-17, $0)**
Checked against `e4_export.py`/`engine.c`: the premise was wrong. `in_proj` is already merged (2·Dn×D single tensor), SWA `qkv` is already merged (3D×D), and the remaining projections do NOT share an input — `x_proj` reads the post-conv Dn stream, `dt_proj` reads the rank-16 dt slice, `out_proj` reads the scan output: a sequential dependency chain, not parallel same-input GEMVs. Nothing left to fuse. Closed as a clean verified-negative; kept here as the record.

## 5. Reading grid (which wall each spark hits)

| Spark | Wall attacked | Era | Cost of testing | When |
|---|---|---|---|---|
| S1 structured proj | resident bytes + compute floor | — | sandbox A/B, prereg gate | post-MVE, strong candidate |
| S2 entropy-weights | streamed bytes (the 10B story) | fusion A+F | decode microbench + entropy count on existing ckpts ($0, CPU) | now, offline |
| S3 delta-experts | streamed bytes → shared | fusion | analysis on existing `moe_gran.pt` ($0) | now, offline |
| S4 SEE tokenizer | quality/BPB, originality | fusion | code-val re-segmentation ($0, CPU) | v2 research |
| S5 α-QAT | curriculum stability | C | ~10 lines + short run | post-MVE amendment |
| S6 unified lookup | code/scaling/capacity | D | design doc, then C | v2 design |
| S7 SEE drafter | (parked with block-verify) | F | zero | recorded |
| S8 fused GEMV | compute floor | A | export layout | engine-v2 |

**Method note:** S2 and S3 can be *quantified today at zero cost, no GPU*: measuring the empirical entropy of the ternary weights of `moe_gran.pt` and the sparsity of the from-centroid residuals is a CPU script over checkpoints already in-house. That is the natural first step if the owner wants a light-bulb turned into a number.

---

*End of the big picture. Stopping here as agreed: awaiting direction (deepen one spark, quantify S2/S3 at zero cost, or something else).*
