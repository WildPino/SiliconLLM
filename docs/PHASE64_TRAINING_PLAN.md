# Phase 64.4 — The Training Plan (the ladder, costed)

**Status: DELIVERED 2026-07-12 (Architect) → STOP-A: owner approves the plan and launches stage-0 ($0). Inputs: `PHASE64_DECISIONS.md` (D1-D9 + ladder + 64.3 PASS), `PHASE64_BUDGET.md`, the adjudicated training dossiers (curriculum skeleton promoted; hazards corrected), the owner's sealed goal and envelope. All GPU runs are launched by the owner; gate-bearing runs follow push-before-run (config + gates pushed before launch).**

**What the spend buys (the sealed goal):** ~€100, conditional, buys **confirmation that the architecture holds toward 10B** — a measured property *trend* on one fixed recipe across MVE 22M → S0 ~30M → S1 105M → S2 206M, with the engine export at the top. Not a 10B run; not a vanity headline. Every prediction below is **desk-model, not yet trained** — brackets are wide on purpose and get re-priced once at the MVE (costs re-priced; gates never).

---

## 1. Envelope (declared; quota numbers verified at stage-0)

| resource | what | role |
|---|---|---|
| Kaggle ×4 accounts (4 real people, ToS-clean) | ~30 GPU-h/wk each ≈ **120 session-h/wk aggregate**, sessions = 2×T4 16GB fp16 (Turing: no bf16 → loss-scaled AMP), 12h cap, Linux (**regional torch.compile viable** — dead only on Windows) | the workhorse: training + bulk logit generation |
| RTX 3060 12GB (local, bf16 eager) | always-on | dev, smokes, MVE fallback, small logit jobs |
| RTX 5090 (friend's loan — **one window, sparing**) | ~a weekend | the **premium-teacher logit burst** (the one stage that wants big-VRAM fast inference) |
| vast.ai ~€100 **conditional** | released only at STOP-B | premium-logit top-up and/or re-run buffer |
| Local scratch | **≥500 GB needed** (corpus ~40 GB + rolling logit chunks ~2×90 GB + ckpts) — logistics line for the owner; Google-Drive-as-storage (dossier) rejected | chunked generate→train→delete pipeline |

Deviations from the dossiers, declared: **one bulk teacher + optional premium slice** (not the 3-teacher ensemble; Mamba-2 hidden-state distillation OUT — unvalidated extra variable); **top-K = 32** with 8-bit quantized probs (vs 64 — halves storage, A/B-able at MVE); optimizer = **AdamW 8-bit** (Muon = blog-tier ref, out); Kaggle quota figures above are assumptions to verify, not facts.

## 2. Teachers & the offline-KD pipeline

**Pre-flight T1 (sealed, $0, before ANY logit generation):** measure **raw-LM BPB on our pinned code-val** for the shortlist — Gemma 4 12B (*base/non-thinking variant if it exists; chat template OFF* — the reasoning-model calibration risk is why this gate exists), Gemma E4B-class, and 1-2 compact open code-specialist models (1-3B). Rank; check each license permits distillation of outputs (Researcher brief item). **Rule: a teacher that does not clearly beat the next-cheaper teacher on our code-val does not earn its generation cost.**

**Pre-flight T2 (sealed, $0):** tokenizer **boundary-coincidence coverage** teacher-vs-student BPE on code-val (D3) — instrumentation for the span-mapping KD; the D3 MVE gate (cross-tokenizer KD must beat plain CE at 22M) decides mapping vs sequence-level fallback.

**Owner input (2026-07-12, adjudicated IN):** with 30-206M students, the default teacher posture flips to a **compact primary teacher (1-4B, code-capable)**. Three independent legs: (i) the owner's point — reasoning ability doesn't transfer through teacher-forced logits on a fixed corpus, raw-LM code calibration does; (ii) the **capacity-gap literature** — a too-strong teacher can *hurt* a small student; (iii) generation economics (3-8K tok/s on T4 vs hundreds for 12B). Gemma 4 12B stays in the T1 shortlist as the premium *reference* (evaluating its BPB is cheap; only generation is expensive) and earns a generation slice only per the sealed rule; if T1 is ambiguous, a **declared teacher A/B at the MVE** decides. The 5090 window is reallocated accordingly if the compact teacher wins outright (candidate use: the S2 logit tail or the premium slice of whichever teacher T1 ranks first).

**TEACHER SEALED (T1 registered, 2026-07-13): Qwen2.5-Coder-1.5B.** See §10 for the full stage-0 adjudication.

**Generation economics (desk, the honest crunch):** logit generation costs more GPU than training at these model sizes. Therefore **KD-on-subset + CE-on-rest**: KD tokens ≈ 1-2B highest-value code (premium slice from the 5090 window ≈ 0.5-1.4B at fp8 3-8K tok/s desk-bracket; bulk from a compact teacher on Kaggle T4s at 3-8K tok/s Q8; vast.ai top-up conditional), CE for the remaining corpus. Storage per the K=32 format ≈ ~180 B/token → chunked (gen chunk i+1 on one account while training chunk i on another; delete behind).

## 3. Data (the known gap — Researcher brief goes out at stage-0)

Need: **~6-10B licensed code tokens** (The Stack-v2-class permissive subset, license-filtered, dedup'd) + our pinned P62 code-val for all BPB gates. Rules carried: corpora never committed (manifest + script + hash only; `data/` gitignored); logs eval-only (P62); LogHub research-only stays out of release assets. Brief asks: exact subsets & licenses, dedup recipe at 40 GB scale, hosting path to Kaggle datasets, teacher-output license terms. TinyStories only at the MVE.

## 4. The curriculum per rung (dossier skeleton A→F, adapted; order itself is MVE-checked)

**A** data+tokenizer ready (per-domain BPE; V per D3) → **B** teacher logits (chunked, offline top-K=32) → **C** KD pre-train fp16 (KD-on-subset + CE-on-rest) → **D** QAT ternary (KD-first-then-QAT; P61 precision map: fp32 organs untouched) → **E** MoE **sparse upcycling** from the dense checkpoint + recall tier (InfoNCE warmup, λnce per P55) + progressive context — *prereq: the sparse-slot training rewrite (engineering item, local dev, before rung-1)* → **F** reverse-KL fine-tune (small, S-rungs only). Token split desk: C 55% / D 20% / E 20% / F 5%.

**The matched-dense baseline per rung (gate 1 of DECISIONS §10):** the pre-upcycle QAT-dense checkpoint is the natural arm. At **S0**: full paired protocol (cheap — dense arm continued at equal step budget, probe-4 pattern). At **S1/S2**: pre-upcycle checkpoint + a short continued-dense control (declared cheaper proxy, ~10-15% extra compute, priced in).

## 5. The ladder, costed (desk-model brackets; MFU 5-15% assumed on 2×T4, scan-heavy — measured at MVE)

| rung | tokens (desk) | train cost (2×T4 session-h) | logit cost | calendar @120 h/wk |
|---|---|---|---|---|
| **0 — MVE 22M TS** | 0.3-0.5B | ~15-40 h (1×T4/3060 class, 3 days) | tiny (3060/T4) | **$0, ~1-2 wks incl. pre-flights** |
| **1 — S0 30M** | 0.8-1.5B | ~10-25 h ×1.5 curriculum ≈ 15-40 h | subset share | ~1 wk |
| **2 — S1 105M** | 2-4B | ~60-180 h | subset share | 1-2 wks |
| **3 — S2 206M** | 5-8B | ~240-700 h | subset share + 5090 window | 2-6 wks |
| gates/export/controls | — | ~10-15% overhead | — | in-line |

**Total desk-bracket: ~350-1000 session-hours ≈ 4-10 weeks at full aggregate quota, $0-path feasible.** The single re-pricing event: MVE measures real train tok/s and gen tok/s → this table is re-issued with measured numbers at STOP-B. Costs re-price; **gates do not move**.

## 6. Stage-0 ($0 — launches on STOP-A approval)

1. **Pre-flights T1+T2** (teacher BPB ranking; tokenizer coverage) — CPU/3060, briefs ready.
2. **Researcher data brief** out (§3).
3. **DDP validation** (sealed P3 item, never used in this project): 2×T4 DDP smoke ≤1h — loss-curve parity vs single-GPU within declared tolerance, scaling ≥1.6× — **before any long run**.
4. **MVE (22M, TinyStories, $0, pre-registered)** — gates sealed: (i) **D3**: cross-tokenizer KD beats plain CE-on-teacher-text; (ii) full pipeline A→F executes end-to-end incl. resume-from-preemption; (iii) **D4 clause 2 first read**: InfoNCE/recall stage does not destabilize the curriculum vs the no-recall control; (iv) KD-then-QAT stability (no divergence at the QAT switch); (v) measured throughputs → the STOP-B re-priced table. Quality at 22M is *recorded*, not gated (TS anchors are for property shape, not SOTA).

## 7. Eval contract (fixed here; per-rung gates defined in `PHASE64_DECISIONS.md` §10)

Canonical: **BPB per-domain on the pinned code-val** (P62 protocol) — never a blended mix. Checkpoint probes: sparsity bands (code-measured P62 reproduction), router health + i.i.d.-union sanity, in-place recall 86-92, QAT-gap record vs scale. Recall diagnostic: **MQAR-style**. Downstream: HumanEval **record-only** at S1/S2 (no gate — anti-Goodhart). Export gate at S2: C-engine export, **bit-exact parity**, tok/s ≥ the 64.1 grid floor (739 for S2) per-protocol on the reference. Decode hygiene locked everywhere.

## 8. Risks (each with its owner)

fp16-no-bf16 on T4 → loss-scaled AMP, MVE-validated | preemption → ckpt ~30 min + tested resume | storage 500 GB → owner logistics before rung-1 | teacher calibration → pre-flight T1 gate | recall destabilization → D4 clause 2, demotion path sealed | Kaggle ToS → 4 real people (owner-confirmed), fallback = single-account stretch | ckpt loss → promoted ckpts as release assets (R1 rule) | never 2 trainers on one box (local rule) | Windows-vs-Linux train divergence → code stays Linux-friendly, compile only on cloud.

## 9. Spend scenarios (STOP-B decides, after the MVE re-price)

| scenario | spend | buys | evidence delta for the 10B claim |
|---|---|---|---|
| **$0** | 0 | full ladder on Kaggle + 5090 window; compact bulk teacher; KD subset ~1-1.5B | complete trend, 3 points + export — the claim stands on this alone |
| **~€50-60** | vast.ai gen | +1-2B premium-teacher logits (better KD signal at S1/S2) | stronger quality-per-param at the top points |
| **~€100** | + buffer | re-run insurance for one failed rung stage without a quota stall | schedule/risk, not evidence |
| M1 402M stretch | out of €100 | — | explicitly not bought; S2 is the top |

**Sequence: STOP-A (now: approve plan → stage-0 $0) → MVE + pre-flights return → re-priced table → STOP-B (spend go/no-go + rung-1 launch) → rungs with per-rung gates → export gate → the 10B dossier (trend curves + banked laws).**

---

## 10. Stage-0 returns, adjudicated (2026-07-13: T1 + T2 registered, data brief received)

**T1 (teacher BPB on the pinned code-val, 1.5 MB CPython, chat-template OFF, all on the 3060):** Qwen2.5-Coder 7B(4bit)/3B/1.5B/0.5B = 0.3513/0.3641/0.3819/0.4408; starcoder2-3b 0.3724; deepseek-coder-1.3b 0.4240. deepseek and starcoder2 are strictly dominated (worse BPB at higher cost) — out. Returns collapse after 1.5B: 0.5→1.5B buys −13.4% BPB at 3× cost; 1.5→3B only −4.9% at 2×; 3→7B −3.5% at 2.35×. **By the sealed read-rule: the teacher is Qwen2.5-Coder-1.5B.** Reinforcements: capacity-gap (§2) favors the smaller teacher for 30-206M students; one family tokenizer across 0.5/1.5/3/7B means the span-mapping is built once; and (verify live, stage-0) **Qwen2.5-Coder-1.5B is Apache-2.0 while the 3B variant sits under the Qwen *Research* (non-commercial) license** — the "only defensible upgrade" carries a license restriction the 1.5B doesn't. The 3B remains a *recorded* upgrade path, engaged only if the MVE shows teacher-limited KD signal. Gemma premium reference: unmeasured (gated), now **moot for the decision** — optional record-only run if a Kaggle session sits idle. **Rejected as economic input: the batch-1 gen tok/s column** (physically inverted ordering 0.5B < 1.5B = per-token overhead, not model cost); real generation economics measured **batched (vLLM)** at the MVE stage-B → feeds the STOP-B re-price. Interim proxy: params × tokens. **5090 window re-role (desk-bracket):** a 1.5B teacher batched on the 5090 plausibly generates the *entire* 1-2B-token KD subset in one weekend — potentially freeing the whole Kaggle quota for training; measured at MVE.

**T2 (boundary-coincidence, student V ∈ {1024, 2048, 4096} × teacher tokenizers):** with Qwen, 39-44% of student boundaries coincide with a teacher boundary; exact 1:1 token alignment covers only ~8-10% of bytes → **pure token-level KD is dead, span-mapping is alive**: coincident boundaries tile 100% of bytes into segments of ~6.3 B ≈ 2.5 student ≈ 1.4 teacher tokens. **KD design sealed: teacher top-K distilled at segment-start anchors (~40% of student positions), CE elsewhere**; the sequence-level fallback stays armed behind the D3 MVE gate exactly as pre-registered. Vocab is NOT the alignment lever (V 1024→4096 moves coincidence 4 pts) → the V decision stays at the rung-1 BPB A/B; recorded: V4096 shortens the student sequence 17% (compute saving both in training steps and in bytes/s at inference). The T1↔T2 tension is understood and accepted: Qwen's big tokens = cheapest logits, fewest anchors — at 2.5-token span granularity this is the *canonical* span-mapping case, not a pathology.

**Data brief (Z.ai, `docs/research/training/SiliconLLM_Data_Sourcing_Brief.pdf`, 24 pp) — adjudicated: ~90% adopted, four corrections:**
1. **ADOPTED:** The Stack v2 (`the-stack-v2-dedup` variant) as source, permissive+no_license pre-filter with the strict-permissive option; MinHash dedup at BigCode defaults (5-shingle, 128 perm, J=0.7) after exact-hash pass; **P62 decontamination J=0.5 as a MANDATORY pipeline stage** — not theoretical: our code-val is CPython source and CPython is *in* The Stack → contamination is certain without it; P62 itself never modified; manifest+hash discipline (§7 of the brief) verbatim, hash-verify at every training start.
2. **CORRECTION (load-bearing): the content-acquisition path.** The brief's scripts read a `content` column from the HF parquets — that is Stack-*v1*-shaped; **v2 on HF ships file IDs/metadata, with contents fetched from Software Heritage S3**. Stage-0 smoke (Builder): live-verify the v2 content path on one shard before committing; fallbacks if painful: strict-permissive Stack-v1-class content-bearing subsets (we need only ~40 GB).
3. **CORRECTION: storage strategy.** The brief prices the full logit set at rest (1.2-1.5 TB, top-64) and proposes HF-Hub hosting ("no hard limit" — stale for private repos, and our no-redistribution rule keeps every shard private anyway). Superseded by this plan's **chunked generate→train→delete** pipeline + **K=32**: rolling ~2 chunks on the ≥500 GB local scratch; Kaggle private datasets (200 GB, verified quota) as the transfer buffer. The brief's parametric calculator is kept.
4. **CORRECTION: the teacher-license section analyzes the superseded panel** (Gemma/Mamba-2/StarCoder2 — written pre-T1). Standing live-checks for stage-0 instead: Qwen2.5-Coder-1.5B Apache-2.0 confirmation (+ distillation terms), and the brief's Gemma-4-is-Apache-2.0 claim (April 2026, multiply sourced, post-cutoff) noted as resolving the old D2 wrinkle but now decision-moot. Cosmetic flag: the brief's P62 manifest *example* invents HumanEval/MBPP contents — our P62 is the pinned CPython byte-slice; ignore the example, keep the format. Kaggle session length (9h vs 12h) verified live at first use; non-load-bearing.

**Stage-0 remaining:** the MVE apparatus (Builder, with `MVE_PREREG.md` pushed before launch) + the data smoke (item 2 above) + the two license live-checks. DDP smoke folds into the MVE apparatus validation. Then: MVE run (owner, ~3 days, $0) → re-priced table → STOP-B.
