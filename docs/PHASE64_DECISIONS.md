# Phase 64.2 — Design Decisions (pre-registered criteria → verdicts)

**Status: DONE 2026-07-12 (Architect). Inputs: `PHASE64_BUDGET.md` (64.0/64.1/64.1b measured), the banked phase verdicts (probe-1..4, P55/56, P61, P62, P63), the adjudicated training dossiers (KD-cross-tokenizer constraint), and the owner's sealed goal (the ladder toward the 10B claim). Output: the decision set that 64.4 freezes into the spec + the single 64.3 probe it engages. Canonical gate plan: `SCALEUP_ARCHITECTURE.md` §8.**

**Method note (anti-Goodhart, stated up front):** most decisions below are decided by *already-banked* measurements — those are adjudicated openly, citing the numbers. Criteria are genuinely pre-registered only where a decision waits on *future* numbers (the 64.3 recall smoke, the rung-1 vocab A/B, the MVE KD gate): those criteria are sealed **here, before their numbers exist**, and are not loosened afterward. One variable per stage. All hardware numbers parametric (Portability law).

---

## 0. What 64.2 binds — and what it does not

Binds: the v1 architecture family, the **ladder recipe** (one recipe, scaled by declared dims only), the per-decision verdicts D1-D9, the property-gate *definitions*, and the single 64.3 probe. Does **not** bind: the training plan and costs (64.4), the exact eval contract (64.4), the SCALEUP document upgrade to frozen spec (64.4, after the 64.3 probe returns).

---

## 1. D1 — Recipe invariance: the ladder contract

**Criterion (sealed):** the ladder exists to confirm that the *validated* architecture scales. Therefore every rung runs the **same recipe** — the code-verified foundation (Mamba-1 selective scan, `dt_rank = D/16`, `expand 2` → Dn = 2D, conv4; **one** SWA layer, window 128, H = D/32, placed as the last block of each 6-layer group; gated-dReLU ternary MLP → MoE-fied per D2; projections/SSM/head **fp32** per P61; decode hygiene locked) — scaled only by the declared dims (D, N, L, V, E). **Any structural deviation is a new variable and needs its own pre-registered gate before it enters a rung.** SWA density beyond L=8 (2 SWA layers at L≥12) is a declared extrapolation of the placement rule, record-only.

**Why this matters for the 10B argument:** the ladder's evidentiary value is *trend on a fixed recipe*. A recipe that mutates between rungs proves nothing.

## 2. D2 — Two-pool split & MoE config (the capacity lever)

**Criterion:** under the constraint inversion (64.1), MoE parameters are chosen by **quality-per-param**, not speed; engine dispatch overhead is engineering, not an architecture input.

**Verdict (from banked numbers):**
- **Granularity: E×h128, top-8, per-layer routing — sealed** (probe-4: granular E32×h128 top-8 BPB 0.8589 beat both the dense upper-bound arm and the coarse arm at matched total params; 64.1b: the 8.4 µs/expert dispatch overhead is *around* the kernel and engine-addressable — it is not an argument to coarsen experts, per the inversion).
- **Capacity scaling: E is the ladder's only capacity dial** — S0 E32 → S1 E128 → S2 E256 at fixed active (top-8 × h128 × L8 = 3.1 MB/token). S1↔S2 doubling is speed-free (64.1 grid).
- **Router: Switch-aux as apparatus** (probe-4-validated: 0 dead experts, max/mean ≤ 1.47 at E32). **Aux-loss-free (DeepSeek-V3-style) = a declared A/B** in the training plan, never a silent swap. **Rung gate (sealed):** router health (0 dead, max/mean ≤ 1.5) and **i.i.d.-union sanity** — measured expert unions must track E·(1−(1−topk/E)^K) as at 8.3M; a *departure toward locality at scale would be a finding, not a failure* (it would re-open Finding-7 and the hot-pool question — record, don't gate).
- **Two-pool layout sealed:** resident fp32 pool (backbone proj/SSM/SWA/head/router) + DRAM-streamed ternary expert pool, contiguous per-expert blocks (ρ-safe, granularity-bounded — probe-4 + 64.1b measured 2.5× i.i.d. penalty at 48 KB grain, not 14×).

## 3. D3 — Vocab & head, under the KD-cross-tokenizer constraint

**Criterion (sealed before the A/B):** (a) the head must stay resident-noise (≤ ~4-8 MB fp32); (b) the *unit* is decided by **BPB on code-val** (bytes-per-byte normalization — the mantra guard makes vocab choices comparable); (c) an offline KD path from a 256K-vocab teacher must exist and be **demonstrated at the $0 MVE before any paid run**.

**Verdict:**
- **Per-domain code BPE, V ∈ {2048, 4096}; the point is picked at rung-1 by code-val BPB** (cheap A/B, both heads cost 2-4 MB at D256 — criterion (a) is safe either way; the teacher's 256K tokenizer for the student stays REJECTED — banked: ~196M emb/head, ~10 ms/token head GEMV, keystone-killer).
- **Desk observation (marked as such):** larger V also compresses tokens/byte, and the W2 wall is per-*position* — so vocab is quietly a compute-floor lever in bytes/s. The A/B metric (BPB per byte + measured bytes/s) captures this automatically; no separate gate.
- **KD method: boundary-coincident top-K span mapping** (at student-token boundaries that coincide with teacher-token boundaries, map teacher top-K span probabilities onto student tokens by byte-prefix matching; plain CE elsewhere), **fallback = sequence-level KD** (+ the reverse-KL fine-tune stage, unaffected). **Pre-flight (instrumentation, no gate): measure the boundary-coincidence rate** teacher-vs-student BPE on code-val (CPU, free). **MVE gate (sealed): cross-tokenizer KD must beat plain CE-on-teacher-text at the 22M pilot** — else the fallback becomes primary and 64.4 re-costs before any spend.
  - **Implementation sealed 2026-07-13, BEFORE the MVE run — pushed immediately after the prereg, in the same pre-launch session, nothing yet executed (MM note: the prereg landed in `7db35ca` and this decision in the commit that follows it; the honest claim is decision-before-*run*, not decision-before-*prereg-push*) (Builder finding, Architect decision): the span mapping is CHAIN-RULE-FACTORIZED (`--kd span`), not anchor-projected.** The anchor-only projection (teacher top-K collapsed onto the *first* student token of each segment) was measured degenerate during apparatus construction: many-to-one collapse retains only **13-15% of teacher entropy** (H≈2.1 bits → 0.28-0.32 bits; targets near-one-hot on 74% of anchors; smoke: CE 2.081 beat KD-anchor 2.729) — correct and nearly empty, and raising V makes it *worse*. The chain-rule form conditions the same stored top-K rows on the bytes already emitted within the segment, supervising *every* student position in the span: **83-86% of teacher entropy retained, zero extra teacher compute or storage.** This is the faithful implementation of the span-mapping intent sealed above; the anchor variant is recorded as the degenerate case. Not gate-loosening: the D3 gate (KD must beat CE) is unchanged, and the decision predates the prereg push — a gate run against a variant already measured empty would teach nothing.
- **Probe-5 (compressed head): NOT engaged for v1** (V ≤ 4K keeps the head hot; probe-5 re-enters only if a future vocab decision exceeds ~32K).

## 4. D4 — Recall tier: IN for v1 (conditional-sealed) → the one 64.3 probe

**Posture carried from 64.1: PROMOTED to load-bearing.** The blueprint's thesis is the thinking/knowing split; a ladder that validates only the thinking half does not confirm *this* architecture. Capacity-without-params is also exactly what the P3 wall rewards.

**Criterion (sealed NOW, before any recall number at scale — including the demotion path):** recall stays IN v1 unless
1. the **64.3 engine-side smoke** fails: standalone C two-stage query (IVF + 4-bit ADC shortlist + exact top-16 rerank, dim 128, Hadamard partition, nlist ∝ √N) at 128K entries costs **> 50 µs/token** on the reference (P56 gate was 30 µs research-side, measured ~18 µs; ×1.7 integration allowance), or perturbs parity when disabled; **or**
2. the MVE / rung-1 shows the **InfoNCE warmup stage destabilizes the KD curriculum** (quality gate on the *rest* of the recipe regressing vs a no-recall control).

If either fires, recall demotes to **declared-v2** — decided *before* any paid run, with the ladder property-gates re-scoped honestly (the 10B claim then explicitly covers the thinking tier + MoE capacity only).

**Spec binding (what 64.4 freezes if the smoke passes):** one recall slot (SSM-state query, dim 128, value summed into the residual), placed late-mid like the SWA block; InfoNCE representation training (P55: load-bearing, λnce warmup per the promoted curriculum stage); values in RAM latency-class (ρ-safe, like the n-gram table); codes compact and sequential. **Known cost pulled in:** the sparse-slot training rewrite (SCALEUP §4, engineering) becomes a priced 64.4 line item — this is v1's single biggest integration risk, which is exactly why its demotion criterion is pre-registered above.

**VERDICT (64.3 registered run, HEAD 7672107, 2026-07-12): clause 1 CLEARED — recall stays IN v1.** Gate read at the operating point (t6 — interpretation sealed *before* the registered numbers, justified by the 64.1 budget's declared "t6 operating point" assumption + D8): **29.05 µs/token @128K ≤ 50, margin 1.72×**; zero perturbation (standalone probe, kselftest 5/5, P43 golden bit-identical). Records, no gates: t1 = 52.4 µs (scalar-fp32 ADC; research-side SIMD 4-bit measured ~18 µs → engine-v2 engineering item); two-stage as measured = shortlist ~64 → exact rerank → top-16 (spec language); footprint 1.69 MB searchable + 64 MB value-store RAM latency-class (~32 KB touched/token, ρ-safe). Scope honesty: this clears query *cost*, not integration *correctness* — the latter stays on the ladder's export gate. **Clause 2 (InfoNCE curriculum stability at MVE/rung-1) remains armed; the demotion path is unchanged.**

## 5. D5 — Block-verify: OUT for v1 (record)

Sealed by the Phase 63 law + the inversion: routed pools don't amortize (i.i.d. unions, 3rd confirmation), the operational win-condition (C/T ≈ 0.15-0.20 on the *shared* streamed class) is nowhere near met by S-class candidates, and speed is not the binding constraint. Chassis stays banked behind `--block K`, default off, zero cost on the AR path. Nothing new to decide; nothing re-opened.

## 6. D6 — Depth-reuse (K2): OUT of the ladder, deferred

**Criterion applied:** the ladder recipe admits only validated components (D1). K2 is unvalidated (no BPB measurement of tied-reuse vs untied) and S-class needs L=8 ≤ 10 — there is no depth pressure inside the trainable region. **Verdict: OUT of v1; the sandbox probe is engaged only if 64.4's quality modeling demands L-equivalent > 10** (M-class and beyond). Putting K2 into the ladder now would confound the very scaling trend the ladder exists to measure.

## 7. D7 — Finding-7 predictability/SKIP: DEFERRED (record)

Sealed at 64.1: it saves streamed bandwidth, which is not scarce in the trainable region. It re-enters at L1-class sizes or constrained silicon. Explicitly *parked, not abandoned* — it remains the highest-originality open claim (SCALEUP §3.6), and the D2 rung gate above records routing-locality at every scale point, which is precisely the data that would justify re-opening it.

## 8. D8 — Threads: IN (record, with the 64.1b corollary)

Sealed at P63 (2.45×/2.39×, bit-identical by construction). New from 64.1b: the expert path is **not** inherently thread-flat (kernel-pure ×2.29 vs engine ×1.06 — the flatness was dispatch overhead) → threading the expert path after the overhead fix is banked engine-v2 headroom, no architecture consequence.

## 9. D9 — Precision map: fp32 organs, sealed for the ladder

P61 measured (+0.018-0.022 BPB to ternarize projections — they stay fp32; head/SSM likewise). **Record-only at rung-2+:** re-measure the QAT gap trend with scale (prior-art expects it to shrink) — a *measurement*, not a design change; any precision-map change would need its own gate and would break D1 invariance for that rung.

---

## 10. The ladder (the goal reframe, made concrete)

| rung | config | total | role | property gates (definitions; exact eval contract = 64.4) |
|---|---|---|---|---|
| 0 | MVE 22M, TinyStories, 1×T4, $0 | 22M | **pipeline pilot** — KD-offline plumbing, cross-tokenizer gate (D3), FP16→ternary stability | pipeline-only + the D3 MVE gate |
| 1 | S0: D256 N96 L8 V{2048,4096} E32×h128 top8 | ~30M | first code rung; **vocab A/B (D3)**; dense→MoE upcycling validated; no-recall control for D4.2 | full gate set, first pass |
| 2 | S1: same, E128 | 105M | scale point 2 | full gate set |
| 3 | S2: same, E256 | 206M | **v1 product candidate** + engine export | full gate set + **export gate** |

**The per-rung property-gate set (what "the architecture holds" means, per point):** (1) code-val **BPB vs the matched dense arm** (probe-4 pattern; the pre-upcycling dense checkpoint is the natural baseline — exact matching protocol fixed in 64.4); (2) **sparsity bands** reproduce (hidden ~92% / gate ~79%, in-place recall 86-92%); (3) **router health + i.i.d.-union sanity** (D2); (4) **recall property** (retrieval diagnostic, MQAR-style — if D4 holds); (5) **QAT gap** recorded vs scale (D9). **Export gate at the top rung:** C-engine export, **bit-exact parity** (the P60 law: system-correctness, not kernel-correctness), tok/s ≥ the 64.1 grid floor for that config (grid = conservative floor, 64.1b) — measured per-protocol on the reference.

**Keystone, parametric restatement (folds 64.0/64.1b into §2 at the 64.4 upgrade):** active per-token slice ≤ **aggregate LLC at the operating thread count** (reference: 2×16 MB at t6, row-partitioned; measured ~185 GB/s resident, slope to ~34-36 GB/s streamed). S-class resident ~16.8 MB + active tern 3.1 MB sits inside the aggregate with margin; the single-CCX 16 MB figure remains the t1/floor story.

**Out of the ladder:** M1/M2/L1 (P3-bound; M1 only as a stretch row in the 64.4 cost table, never at the expense of a clean S2).

---

## 11. Handoff

- **64.3 (one probe engaged): the recall C-side query smoke** (D4 criterion 1, sealed above). Builder apparatus; **push-before-run**; STOP with latency/footprint/parity tables. No other probe is blocked-on.
- **Free pre-flights (CPU, instrumentation):** tokenizer boundary-coincidence rate (D3); teacher raw-LM BPB on code-val before any logit generation (banked owner decision D2/dossiers — reasoning-model calibration risk).
- **64.4:** costed training plan on this ladder (curriculum A→F adapted; envelope = 4 Kaggle accounts + sparing 5090 window + ~€100 conditional; MVE $0 first; sparse-slot rewrite priced; 2-GPU DDP validation before long runs; storage plan for offline logits) → **STOP: the owner decides the spend.**
