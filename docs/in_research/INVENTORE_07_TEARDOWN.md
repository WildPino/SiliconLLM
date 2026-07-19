# INVENTORE_07 — The Teardown: fuse / parallelize / split / invert

**Status: Inventor pass 2, desk-only (2026-07-19). No code touched, no runs launched.** Companion to the
pass-1 series (00–06). Pass 1 looked *at the organs* and found one (x_proj, S1 → adopted, plan §12).
Pass 2 looks *at the interfaces*: the owner's brief is to take the machine apart piece by piece and ask,
for every pair of pieces and every chain, whether a fusion, a parallelization, a decomposition, or an
order inversion makes it work better. External view only — the frozen v1 spec, the in-flight MVE
attempt-3, and every adjudicated decision are untouchable inputs here, not variables.

**Grammar applied at every edge:** FUSE (two pieces → one), PARALLELIZE (sequential → concurrent),
SPLIT (one piece → two with different laws), REORDER/INVERT (chain order → its inverse), ELIMINATE
(piece → nothing). Every proposed move names its falsification test and its cost. Moves that would
collide with a seal are recorded as **non-moves** (§4) — the discipline is part of the deliverable.

Sources: `SCALEUP_ARCHITECTURE.md` (frozen spec v1 §9), `PHASE64_TRAINING_PLAN.md` (§1–12 incl. the
2026-07-19 adjudication), `ENGINE_PLAN.md` (E1→Phase 63 + laws), `PHASE64_DECISIONS.md` (D1–D9),
pass-1 results (INVENTORE_00–06).

---

## 1. The machine, disassembled

Five planes, ~30 pieces. Arrows are the interfaces this pass interrogates.

### 1.1 Data plane (rung-1+; MVE runs on TinyStories)
```
Stack-v2 IDs (HF, gated) → SWH-S3 content fetch → license filter (permissive+no_license)
  → exact-hash dedup → MinHash dedup (5-shingle, 128 perm, J=0.7)
  → P62 decontamination (J=0.5, MANDATORY) → chunked storage (~40 GB corpus, ≥500 GB scratch)
  → per-domain BPE (V per D3 A/B) → Kaggle private-dataset shards (200 GB buffer)
```

### 1.2 Teacher plane
```
Qwen2.5-Coder-1.5B (Apache-2.0, sealed) → prefill scoring (measured 2283 tok/s on the 3060)
  → top-K=32 logits + row logsumexp, stored per TEACHER token (~180 B/tok)
  → chunked generate→train→delete → span-mapping chain-rule KD (`--kd span`),
    anchors ≈ 40% of student positions, CE elsewhere
```

### 1.3 Student plane (curriculum A→F per rung; token split desk C55/D20/E20/F5)
```
A data+tokenizer → B teacher logits → C KD pre-train fp16 → D QAT ternary (fp32 organs untouched)
  → E MoE sparse-upcycle (magnitude-matched = ε-identity law) + recall InfoNCE + progressive ctx
  → F reverse-KL fine-tune. Matched-dense control per rung = the pre-upcycle ckpt continued.
```

### 1.4 Model organs (frozen recipe D1)
```
embed → [Mamba block: in_proj → conv4 → x_proj (dt16 ‖ B96 ‖ C96; r26 structured arm queued at the
S0 boundary, plan §12) → dt_proj (16→512) → selective scan (fast-exp in engine) → out_proj]
+ 1 SWA (win 128) per 6-layer group
→ MoE-MLP: fp32 router → top-8 of E → ternary gated-dReLU experts (DRAM-streamed, i.i.d. unions)
→ recall slot: state-query d128 → IVF-Hadamard → 4-bit ADC shortlist 64 → exact top-16 (29 µs @128K, t6)
→ fp32 head, V ∈ {2048, 4096} (D3 A/B)
```

### 1.5 Engine + verification plane
```
exporter (deq_pack fp32 + packed ternary) → parity goldens (golden-trace / logit / BPB / tokenizer /
generation) → per-token loop: proj-GEMVs (~52%, memory-bound) → fast-exp scan → LUT+skip MLP → head
→ threads (2.4×, bit-identical) → block-verify chassis (`--block K`, correctness-proven, default OFF)
→ decode hygiene locked. Process organs: prereg + push-before-run, one-variable-per-stage,
end-to-end parity law, per-rung property gates, export gate at S2.
```

---

## 2. The moves

### M1 — FUSE: one teacher pass, four products  `[teacher ∘ data]`

**The gap found by the teardown:** plan §2 commits to *"KD-on-subset: KD tokens ≈ 1-2B highest-value
code"* — and no piece of any plane defines **"highest-value"**. Meanwhile stage B computes, for every
scored token, the full teacher top-K + logsumexp — i.e. an exact per-token teacher BPB — and keeps only
the KD tensors.

**Move:** declare scoring a multi-output stage. (a) KD logits, as now. (b) **Per-chunk teacher-BPB
profile = the KD-subset selection criterion**: one cheap sampling sweep (1-5% of candidate chunks)
ranks the corpus by teacher entropy; full scoring is then spent only on the chosen subset. (c) A
**quality filter for free**: teacher-perplexity outlier chunks (minified/generated/near-binary code)
flagged *before* they burn training tokens — the plan has license/dedup/decontamination filters but no
model-based quality filter at all. (d) The **S4 entropy statistics** (INVENTORE_01 §S4) computed on the
real corpus instead of the proxy, zero extra cost.

**Desk:** (a)+(d) cost nothing extra; (b) ≈ hours on the 3060 for the sweep. The teacher signal is *by
definition* aligned with KD value — no proxy heuristic can be better aligned than the teacher itself.
**Falsification:** declared A/B at rung-1 — entropy-selected KD subset vs random subset at equal budget.
**Status:** proposal; touches stage-B design only, decidable before rung-1 data work; MVE untouched.

### M2 — PARALLELIZE: the 3060 as standing scoring producer  `[teacher ∥ training]`

The plan already pipelines gen/train across Kaggle accounts. The idle piece: during rungs 1–3 the 3060
does nothing structural. At the measured 2283 tok/s, the whole 1–2B-token KD subset = **5.1–10.1 days
of continuous 3060 prefill** — the same order as the S1+S2 calendar (3–8 weeks).

**Move:** 3060 = dedicated producer (chunk k+1 scored while Kaggle trains chunk k); Kaggle = pure
trainer; the 5090 weekend becomes burst insurance instead of a load-bearing dependency on a loan.
**Respects:** never-2-trainers (scoring ≠ training, and it is the box's only job); the chunked-delete
pipeline unchanged. **Risk to measure, not assume:** thermals/availability — one 24h pilot chunk.
**Status:** pure scheduling, zero design change; a paragraph in the STOP-B re-price.

### M3 — FUSE: the control latent — x_proj ∪ dt_proj  `[inside the Mamba block]`

S1 ended with: x_proj low-rank (r26 adopted at sandbox), trained factor products at PR ≈ 7–9, and the
dt path *already* rank-16 **by design** (dt_rank = D/16). The teardown reading: x_proj and dt_proj are
**one mechanism split by convention** — x_proj emits (dt₁₆ ‖ B₉₆ ‖ C₉₆), then dt_proj re-expands 16→512.

**Move:** one shared encoder `V: 512→16` (= dt_rank); B and C become linear readouts `16→96`; the dt
slice **is the latent itself** (its 16×16 head folds into dt_proj). Statement: *the selection organ has
one 16-dimensional state; dt, B and C are three readouts of it.*

**Desk:** 11,264 params/layer (8,192 V + 3,072 B/C heads) vs dense 106,496 (**10.6%**) vs r26's 18,720
(17.6%). Three independent arrows point at r=16: the monotone fewer-ranks→better curve, PR ≈ 7–9 of the
trained products, and dt_rank itself. This gives the sealed "r=16 next point" a *shape*, not just a
number — and if it wins it **simplifies** the exporter case (one factored organ instead of factored
x_proj + separate dense dt_proj).
**Falsification:** phase57 apparatus + the sealed 3-seed protocol verbatim (C1-style criterion), one
new arm, hours/seed on the 3060, $0. **Status:** the natural S1 successor; Architect's call per §12
(r=16 was already listed as optional post-MVE).

### M4 — REORDER: QAT ↔ upcycle  `[stage D ↔ stage E]`

Current order: D (QAT ternary) → E (upcycle dense→MoE replicas). So the replicas are seeded from an
already-ternarized MLP and must diversify *inside the quantized landscape*. The inverse (upcycle fp16 →
diversify → QAT) lets experts separate in a continuous landscape first. Why the order might matter:
ternary rounding is a contraction — **replica pairs that differ by less than a quantization step
collapse to identical trits**; S3 measured *from-scratch* experts as fully independent, but whether
*upcycled* replicas ever reach that independence is exactly the open question.

**Move, measurement-first:** step 1 is already in the §12 pending ledger — the S3 rerun + **replica-
divergence telemetry on the stage-E MVE checkpoint** (minutes, CPU). That telemetry *is the
adjudicator*: replicas diverging freely → order fine, question closed; large trit-identical fractions
persisting → step 2, a declared order A/B (…C→D→E… vs …C→E→D…) at MVE scale. Note: the ε-identity law
is symmetric — magnitude-matched upcycle and α-QAT are both ε-identities, so the law itself does not
prefer an order; only measurement can. **Status:** no plan change until the telemetry reads.

### M5 — SPLIT: shared expert ∥ routed pool  `[MoE interior]`

The v1 streamed class is expert-dominated, and the Phase-63 law says block-verify amortizes **shared**
streamed weights only → the lever was scoped OUT and the chassis sits in the engine as a stranded
asset. A DeepSeek-MoE-style **shared expert** (always-on; k routed + 1 shared at matched active) splits
the pool into exactly the two classes the engine laws price differently: the shared expert is a SHARED
streamed class (amortizes at tpp 1.75–2.8 under `--block`); the routed pool stays i.i.d.

Two independent payoffs: (a) published quality evidence (common-knowledge factoring, DeepSeek-MoE);
(b) **it re-creates by design the only regime where the banked chassis pays.** Honest counterweights:
constraint inversion says v1 speed is slack — (b) is a v2/10B argument, not a v1 one; and S3 found *no*
common component across from-scratch experts (the shared expert would *impose* the factoring, not
discover it — the evidence cuts both ways). **Falsification:** sandbox arm at probe-4 scale (E32 top-8
vs E31+1-shared top-7, matched active), $0-class; rung-boundary A/B only if promoted.
**Status:** design note for the Architect; explicitly NOT a v1 proposal (D-decisions sealed).

### M6 — PARALLELIZE: within-token router-ahead — overlap streaming with resident compute  `[engine two-pool]`

Finding-7 (cross-token SKIP) was deferred because routing ≈ i.i.d. kills cross-token prediction. Its
**within-token cousin needs no clairvoyance**: layer ℓ's expert streams are consumed only after ℓ's
mixer; anything that pins the top-8 earlier opens an overlap window where DRAM streaming hides under
resident scan/proj compute — total → max(C_resident, T_streamed) instead of C+T at the limit.

Variants: **(exact)** issue the stream immediately post-router while the MLP prologue proceeds — small
window; **(predicted)** provisional router on the *pre-mixer* activation → prefetch → verify with the
exact post-mixer router, mispredicts fetched on demand — window = the whole mixer. The missing number:
in-place, same-token router predictability from the pre-mixer state (Phase 58's 86–92% is unit-level,
not expert-level). → **$0 probe on the existing `moe_gran.pt`: top-8 overlap between
router(pre-mixer x) and router(post-mixer h), CPU, minutes.** **Status:** engine-v2 lever, zero
training-side change; probe first, design note after.

### M7 — REORDER: export-parity smoke per rung, not only at the top  `[verification plane]`

The export gate sits at S2. But the adopted S1 path changes the **export format**: if the S0-boundary
A/B promotes a structured x_proj (or M3's control latent), the exporter meets *factored organs* for the
first time — and would discover any exporter/parity issue at the S2 gate, the most expensive possible
place. **Move:** a cheap export+golden smoke at every rung boundary that changes an organ (the P64-prep
carry-over "code-ckpt export + golden", generalized). Minutes per rung; catches format drift at the
boundary that introduced it. **Status:** process proposal; **no gate moves** — the S2 export gate stays
the gate, the smokes are instrumentation.

### M8 — FUSE: the lookup family — router ∪ recall ∪ SWA  `[10B trajectory]`

S6 unified router and recall as `lookup(q; K, V, k)` with the measured crossover E ≈ 2048. The teardown
adds the third member: **SWA is the same operator** with K = the recent window. Full statement: the
machine has exactly one non-SSM primitive — lookup against a key set — at three ranges (window-128
resident / expert centroids / 128K indexed). A v2 design can serve window ∪ retrieved keys in **one
softmax** (Memorizing-Transformer-style), removing the bespoke recall-injection surface and letting
retrieval compete inside attention — a milder form of the D4-clause-2 stability question, not a new
risk. **Status:** design note extending INVENTORE_04; no v1 action.

### M9 — PARALLELIZE (minor): data-plane shard jobs

Acquisition/dedup/decontamination/tokenization are shard-parallel CPU jobs that already run during GPU
training by construction (chunked pipeline). One real micro-fusion: compute MinHash shingles **in the
same pass** that tokenizes (one read per shard, two outputs). Logistics, recorded, not invention.

### M10 — ELIMINATE (confirmed): weight-entropy stage

S2 measured trits at 99.9% of max entropy → no entropy-decode stage for weights is ever built; the
5-trits/byte pack is the whole byte lever. Already adjudicated (plan §12); listed because a teardown
must also record what is *correctly absent*.

### M11 — INVERT (already in place, load-bearing)

The plan already contains three healthy inversions the teardown confirms rather than proposes:
reverse-KL at stage F (mode-seeking where mode-covering would blur); span-KD chain-rule (factorize
*through* the boundary mismatch instead of projecting onto anchors); ε-identity curriculum switches
(the inverse of "shock and recover"). No move — found structural, not accidental.

---

## 3. Ranked shortlist

| # | move | operator | cost | why first |
|---|------|----------|------|-----------|
| 1 | M1 teacher pass → four products | FUSE | ~$0 (+ sampling sweep, hours) | fills an **undefined** load-bearing criterion ("highest-value code") with the only signal aligned by definition |
| 2 | M2 3060 standing producer | PARALLELIZE | $0, scheduling | 5–10 days of idle capacity = the whole KD subset; de-risks the 5090 loan |
| 3 | M3 control latent (x_proj ∪ dt_proj, r=16) | FUSE | $0, hours/seed, apparatus ready | three arrows point at r=16; sealed protocol reusable verbatim |
| 4 | M6 router-ahead probe on `moe_gran.pt` | PARALLELIZE | CPU, minutes | one number unlocks/kills an engine-v2 lever |
| 5 | M4 QAT↔upcycle order | REORDER | already-pending telemetry | the §12 S3 rerun doubles as this adjudicator — two questions, one measurement |
| 6 | M7 per-rung export smoke | REORDER | minutes/rung | forced by r26/M3's exporter implication; catches drift early |
| 7 | M5 shared-expert split; M8 lookup family | SPLIT/FUSE | design notes | Architect discussion; v2/10B doors, not v1 claims |

## 4. Non-moves (seals respected)

- **Vocab × x_proj factorial at rung-1** — adjudicated sequential (vocab first, x_proj on the winner);
  one variable per stage stands.
- **Anything touching MVE attempt-3 in flight** — reading its gates is the next event, not this pass.
- **Block-verify re-litigation at v1** — constraint inversion says speed is slack; M5 is the *v2 door*,
  not a v1 claim.
- **K2 depth-reuse, 256K student tokenizer, Finding-7 cross-token SKIP** — sealed OUT/deferred;
  nothing here reopens them.

## 5. $0 probe queue (pending the owner's nod)

- **P-a (M6):** router-ahead top-8 overlap on `moe_gran.pt` — CPU, minutes.
- **P-b (M3):** control-latent arm — 1 seed first; escalate to the sealed 3-seed protocol on a
  C1-gray result — 3060, hours/seed.
- **P-c (M4):** replica-divergence telemetry the moment a stage-E MVE ckpt exists — already in the
  §12 ledger; this pass adds the order-question reading to the same numbers.
- **P-d (M1):** teacher-BPB profile sweep on ~1% of a candidate corpus slice — 3060, hours
  (blocked on the HF-token/data-path items, like all rung-1 data work).

---

**Closing note.** Pass 1 hunted organs and found one. Pass 2 hunted interfaces and found: an undefined
criterion (M1), an idle producer (M2), one organ split in two by convention (M3), an ungated stage
order (M4), a stranded asset with a designed rescue (M5), an unmeasured overlap window (M6), and a
gate placed one rung too late (M7). None of it requires touching the frozen spec today; all of it is
falsifiable for $0 or scheduling alone.
