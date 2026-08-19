# Phase 64 — Rung-1 (S0) Builder Brief

**Status: STOP-B DECIDED 2026-07-19 — GO.** Spend envelope **≤€100 authorized** (deployed only where a rung proves calendar-binding; the default path stays $0). **Hardware-targeted optimization mode ON, quick-wins tier** (owner call at STOP-B). Context: `PHASE64_TRAINING_PLAN.md` §13 (gates read, re-priced ladder), `PHASE64_DECISIONS.md` D1-D9 + §12.

Rung-1 recipe baseline after the MVE verdicts: **CE-primary** (D3 FAIL consequence), span-KD as **challenger arm** on a bounded subset, recall IN with the pre-registered rung-1 demotion clause, **α-QAT applied** at stage D, upcycle continuity checked per the ε-identity law.

The Builder builds, smokes, and STOPs with tables; registered runs are launched by the owner after the Architect's prereg is pushed (push-before-run, no exceptions).

## 0. Owner actions (blocking, restated)

- **HF token, classic Read type** (or fine-grained with "Read access to contents") on the account with `bigcode/the-stack-v2-dedup` terms accepted. Env `HF_TOKEN`, never committed. The token pasted in chat earlier must be revoked.
- **AWS credentials** for the Software Heritage S3 content fetch (`AWS_ACCESS_KEY_ID`/`SECRET`, env only, never committed).
- vast.ai account ready for when a window is called (cap €100 total).

## 1. Workstream 1 — Apparatus fixes (local, $0, first)

1. **Collective stop.** The time-budget stop decision must be all-reduced so both ranks exit at the same step (closes the NCCL race: rank 0 checkpointing while rank 1 waits in the guard all_reduce → watchdog SIGABRT). *Acceptance:* forced time-budget hit under 2-rank DDP → clean exit, no SIGABRT, resume clean.
2. **Dark-knowledge diagnostic** reports retention for the *active* KD mode (span number under `--kd span`). *Acceptance:* correct line in both modes. (It sits inside a gate-bearing log.)
3. **α-QAT** (§13 adjudication: trigger fired). α-ramp on stage-D entry per `INVENTORE_03` (~15 lines). *Acceptance:* end-state weights bit-identical to current QAT on the smoke (checksum), and transition-BPB continuity replaces the +0.32 step (ε-identity law check).

## 2. Workstream 2 — Sparse-slot training rewrite (cost-model-critical)

Replace compute-all MoE training with active-only (top-k) compute + slot-gathered expert parameters. This defines rung-2+ economics (§13: tok/s ≈ flat across rungs only if active compute is what we pay).

*Acceptance:* (a) gradient equivalence vs compute-all on a fixed micro-batch (routed-through experts receive identical grads, allclose at fp16 tolerance); (b) stage-E full config fits memory without the accum workaround, or with accum as a declared config; (c) measured tok/s at the S0 shape ≥ the 3860 floor; (d) the 13-interruption resume/surgery-replay test still passes. *Report:* measured tok/s at E32 and at a synthetic E128 shape (rung-2 forecast).

## 3. Workstream 3 — Data pipeline (blocked on §0 tokens; smoke-first)

One-shard live smoke of the v2 content path (SWH S3 via boto3, **massively parallel fetch** — report sustained MB/s: the download must not be the rung-1 bottleneck). Then the full pipeline: permissive+no_license filter → exact-hash → MinHash (5-shingle, 128 perm, J=0.7) → **P62 decontamination J=0.5 MANDATORY** (P62 files never modified) → per-domain BPE tokenizers at **both V=2048 and V=4096** (the vocab A/B needs both) → manifest+hash, verified at every training start. Target ~40 GB source per plan §3.

Rules restated: corpora never committed or redistributed (manifest+script+hash only, `data/` gitignored, Kaggle datasets private).

## 4. Workstream 4 — Rung-1 apparatus (arms & switches)

- Trainer switches: `--xproj-rank R` (structured x_proj arm, ported cleanly from the in_research S1 code); CE-primary default with KD only in the challenger-arm config; challenger span-KD subset arm; no-recall control; dense-paired continuation arm (matched-dense gate).
- **MQAR-style recall diagnostic on code data** (eval contract §7) — this feeds the pre-registered recall demotion clause.
- Kaggle notebooks: bitsandbytes availability solved (bundled wheel or install; else fp32-AdamW as a *declared* fallback), 2×T4 DDP default, collective stop integrated.
- Challenger logits production plan on the 3060 (producer role, off-Kaggle): ~0.2-0.3B teacher-tok at the measured ~2283 tok/s.

## 5. Workstream 5 — HW-targeted optimizations (quick-wins tier; each one measured)

Order: (1) **AdamW-8bit on Kaggle** (MVE fell back to fp32); (2) **regional `torch.compile` on Linux** (scan region first) — measure per-session compile warmup vs steady-state gain, adopt only if net-positive over a 12h session; (3) input-pipeline streaming only if the profile shows it. **Not now:** Triton fused kernels (bigger lift — revisit after rung-1 measures). **MXFP8: only inside the 5090 window.**

Rules: every optimization A/B'd on the same protocol; applied to ALL arms symmetrically; must not change model math (loss-curve overlay check); resume-compatible. Nothing optimization-related lands inside a registered run undeclared.

## 6. CPU probes (anytime, minutes each — feed the D→E investigation)

S2 rerun on the MVE stage-D (QAT) checkpoint; S3 rerun + replica-divergence telemetry on the stage-E (upcycled) pool. These serve the new head of investigation: the D→E upcycle's +0.07 arm-independent ε-identity violation and the KD arms' failure to recover inside E.

## 7. Sequence and STOPs

1. WS1 + WS2 smokes → **STOP** (tables + tok/s).
2. WS3 one-shard smoke on token arrival → **STOP** (schema + fetch throughput).
3. WS4 apparatus + WS5 quick-wins → **STOP** (measured deltas per optimization).
4. Then the Architect writes the rung-1 prereg (sealed A/B sequence: **vocab → x_proj → main run CE-primary + arms**), the MM pushes it, the owner launches. Push-before-run for every gate-bearing run; gates never move.

## 8. Returns — adjudicated (2026-07-19)

**WS1: 3/3 PASS.** Collective stop proven by logical invariant (autonomous exit-0 at 47.8s vs ~600s watchdog); diagnostic reports the active KD mode with the PER-POSITION label (anchor 11% ≈ the banked 13-15%; span per-position 42% is a *different normalization* from the 83-86% summed-over-span figure — never compare them); α-QAT refactor-invariant at default (weights+logits sha identical), α=0 ≡ fp32 exactly, C→D shock cancelled *exactly* (0.0000 vs +0.0179 hot-swap at smoke scale). **Rung-1 prereg rule fixed: α ON (per §13) with ramp ≤10% of stage-D steps.**

**WS2: PASS, with a premise correction that re-scopes acceptance (b).** Grad-equivalence max rel 1.1e-06 over 124 tensors; determinism restored after removing the atomic path (index_select-with-repeats backward → restructured to fixed-axis sum + unique-index permutation, no atomics); 29-interruption resume identical. Layer-level: sparse 1.51× at E32, **4.27× at E128** (memory 4.06×); end-to-end at the comparability config **+1.1%** — because, measured stage-by-stage, **the SSM scan is the memory/compute eater (stage C alone: 4.86 GB peak), the MoE is 2.9% of peak**. The recorded "MoE compute-all caused the OOM" attribution is corrected: stage E was the last drop, not the load. Acceptance (b) ("stage-E fits without accum") is therefore **not deliverable from this workstream and is re-scoped**: accum stays as the *declared comparability choice* (effective batch 16, micro-batch 8/rank — already ruled); the memory question belongs to the scan lane. The 3860 floor check moves to the first Kaggle session (per-protocol: the +1.1% ratio transfers, the absolute does not). **Cost-model note: §13's flat-tok/s-across-rungs assumption survives** — sparse keeps the MoE share ≈3% of step time at any E (vs ~4× per E-quadrupling under compute-all); the rewrite is the rung-2/3 investment it was priced as.

**Decisions:**
- **`--sparse-moe` ON at rung-1, declared, arm-symmetric** (as is α-QAT). Rationale: rung-2 requires it; its first long run must be the cheap rung, not the expensive one. Grad-equivalence + determinism + resume are its warrant; the run's own bit-identity sanity checks stay active.
- **The scan lane is NOT entered now.** Rung-1 runs on the current scan (recipe invariance; memory suffices at the comparability config; costs are priced on measured numbers with this scan). Two instruments parked, pre-classified: *gradient checkpointing on the scan* = numerics-identical memory knob (~30% compute), to be pulled **only if** a config OOMs (e.g., context extension) — declared, no gate needed; *chunked/associative scan* = a model-touching change → **rung-boundary declared A/B** with the WS2 gate set (grad-equivalence, determinism, tok/s, resume) — candidate if rung-2 calendar bites.
- **Order confirmed: WS6 before WS5** (minutes, feeds the D→E investigation — the largest open discontinuity). In WS5, the scan-region `torch.compile` is promoted to headline item (the bottleneck is now measured to be the scan); AdamW-8bit stays first as the trivial win.

## 9. WS6 returns — adjudicated (2026-07-19)

Validity check accepted: checkpoint-reconstructed baselines reproduce the run-3 finals exactly (0.7015 / 0.6902 / 0.7022), so the ablations measure the real model. The replica-collapse ablation replicated across two independent processes to all digits (unplanned, but it is a free determinism replication).

- **S2 — CLOSED NEGATIVE.** Pool trits at H1 = 1.584 b/w vs flat 1.585, no order-1 structure along rows; the 5-trit/byte pack (1.600) is at the floor. **Product-relevance note that removes the sting of the end-of-F caveat: end-of-F is what ships** — the artifact whose bytes we pay for is the final one, so the product claim ("no entropy-decode stage for weights, ever") is unaffected. The stage-D question is scientific, and `--save-stage-ckpt` resolves it next run.
- **S3 — POSITIVE on the upcycled pool** (56.9% NN-match vs 42.6% shuffled control; +14 pts), opposite sign to the from-scratch pool, exactly as S3's scope note predicted. Builder's self-correction of a blind nearest-neighbour (2000-row subsample on 32768 → a row and its replica co-sampled with ~6% probability) is on record. **Desk estimate of the byte prize, marked as desk:** at 64.6% same-slice trit agreement, a reference+residual code costs ≈1.29 b/w for residual rows → (1.585 + 3×1.29)/4 ≈ 1.36 b/w vs the 1.600 pack ≈ **15% upper bound**. **DO NOT BUILD.** Two reasons: the redundancy is *scale-fragile* (replicas differentiate with steps and with E; measured only at E32/S0-scale budget) and *recipe-dependent* (the ε-identity upcycle below would change the replica structure it feeds on). Re-measure at S0 and S1 before any engine work is scheduled.
- **Replica telemetry — the Builder's own "8 experts in 32 hats" hypothesis is falsified by his own numbers, and this is the right outcome.** Collapsing replicas costs +0.069/+0.075/+0.068 (control 10× that) → **the pool is not degenerate; replicas carry load**, and the router has learned to separate them (row cosine −0.128 same-slice = anti-correlated, not degenerate).

**Decision 1+2 — they collapse into one: the proposed arm IS the test.** The +0.07 ↔ D→E-shock coincidence is a strong clue, not an identity (two deltas measured on different objects — a transition vs an ablation), and the discriminating experiment is exactly the ε-identity upcycle. **It enters the rung-1 prereg as a declared A/B arm at the E boundary**, cheap because both arms **branch from one shared stage-D checkpoint** (E+F ≈ 25% of curriculum tokens → ~+25% of one run, not a second full run). Implementation must be a **ramp, not a hard identity** (the α-QAT analog): router logits = (1−α)·constructed-identity + α·learned, with the **Switch aux loss ramped from 0 in step** — a hard identity routes every token to the same 8 replicas, so (a) the other 24 receive no gradient and (b) the load-balancing loss would demolish the identity on step 1.

**Pre-registered NOW, before the arm exists — three outcomes, all informative:**
- **H1 (Builder's "runs to stand still")**: E repays its own insertion cost → predicts transition shock ≈0.000 **and** end-of-E better than the baseline arm by an amount of order the shock (~0.07).
- **H2 (Architect's alternative)**: the cost is intrinsic to symmetry-breaking, not to insertion → predicts shock ≈0.000 but end-of-E converging to the baseline arm. Then the E budget, not the insertion, is the target.
- **H3 (dead replicas)**: predicts shock ≈0.000, end-of-E no better or worse, **and replica-divergence telemetry staying flat** — the instrument for this already exists (built in WS6). The α-ramp is the mitigation; the telemetry is the detector.

**Decision 3 — `--save-stage-ckpt` APPROVED** (apparatus, not recipe; ~120 MB/stage). The stronger argument than diagnostics: it is the **prerequisite for the cheap branch-from-D A/B above**. Requirements: separate from the resume checkpoint (which is deleted on completion — do not entangle the two), and Kaggle working-dir budget checked.

**Decision 4 (raised by the Architect, not asked): this finding is a probe into D2 itself.** (continues below) D2 sealed "E is the only capacity dial". At E32 the upcycle nets ≈zero: 4 replicas per slice, +0.07 of realized capacity paid for by +0.07 of insertion. But replication grows with the dial (E128 → 16 replicas/slice, E256 → 32), so **if differentiation cost scales with E as fast as the capacity gain, the ladder's core dial does not deliver**. This is not a reason to slow down — it is the reason rung-1→rung-2 matters. **Pre-registered read:** measure the replica-differentiation gain (the collapse ablation, now a standing instrument) at **every rung**; if the gain at S1/E128 is not materially larger than the +0.07 measured at S0/E32, D2 is re-opened at that boundary. Recorded in `PHASE64_DECISIONS.md` terms: this is a *reading rule*, not a gate change.

## 10. WS5 returns + `--save-stage-ckpt` — adjudicated (2026-07-19)

**`--save-stage-ckpt` accepted as delivered.** Both requirements verified with measurements, not estimates: permanent stage artifact (model+cfg, no optimizer state) separate from the transient resume file (which is confirmed deleted at completion while the four stage files survive); 314 MiB/arm on a ~20 GB working dir. Sizes check out against model-only math (11M×4 ≈ 42 MiB, 30M×4 ≈ 115 MiB).

**Open item before the branch-from-D A/B can be declared valid:** does the trainer carry optimizer moments *across* stage boundaries, or rebuild the optimizer at each stage? If it carries them, the stage-D artifact needs them for the branch to reproduce baseline conditions; if it rebuilds per stage, model-only is sufficient. Either answer is workable — **but it must be declared, and both arms must be byte-identical at the branch point.** Builder to confirm before the prereg is written.

> **MM, 2026-07-19 — answerable from the source, no Builder round-trip needed.** The trainer **rebuilds** the optimizer at every stage entry: `stage_surgery(st)` is followed by `net = wrap(model); opt = make_opt(STAGE_LR[st])` (`mve_train.py`, stage loop), and the same factory is re-called on resume. No moments cross a stage boundary — by construction, since the D/E surgery changes the parameter set the moments would be indexed against. **Model-only stage checkpoints are therefore sufficient for the branch-from-D A/B**, and both arms forking from one file are byte-identical at the branch point trivially. Still to be *declared* in the prereg, but there is nothing to confirm.

**Item 1 — AdamW-8bit: NEGATIVE, verdict accepted, and the WS4 line item is struck.** −6.0% throughput at BPB −0.0006 (≪ σ_seed), loss overlay mean 0.0005 → the model-math rule is satisfied and the answer is still no. The dimensional reasoning is correct and I verified it: 30M params → 240 MB fp32 optimizer state vs 60 MB at 8-bit = 180 MB saved on a ~5 GB peak (3.6%), paid for with per-step quantize/dequantize. 8-bit optimizers sell memory, and memory is not where this configuration hurts. The n=1 caveat is accepted with his own framing (run-to-run ~1% observed in WS2; the decision is invariant between −3% and −6%). **Consequence adopted: the "solve bitsandbytes on Kaggle" item is removed from WS4** — runs 1-3 used fp32 AdamW as an unplanned fallback and it was in fact the winning option; nothing to justify, work saved. **Re-open at S2 (206M), where fp32 state reaches ~1.65 GB** — and note the share grows faster than it looks: activation memory is rung-invariant under D2 (D, N, L fixed; only the expert pool scales), so at S2 the optimizer goes from ~3.6% to ~25% of peak. Trigger recorded.

**Item 2 — regional compile: implementation accepted, measurement deferred to Kaggle, with a gate added.** Two implementation choices are ratified as correct decisions, not details: compiling `.forward` as a function rather than the module (avoids the `_orig_mod.` state_dict prefix that would have silently broken every resume and cross-run checkpoint comparison — resume-compatible by construction), and **failing loudly without Triton instead of a silent no-op** (an unmeasured "compiled" label on an uncompiled run is exactly the class of error this project bans).

**Measurement placement: the Builder's option is adopted — folded into the first rung-1 session as a declared A/B on the first N steps — for the *project-law* reason rather than the quota one.** P61's law (microbenches that are compute-bound do not compose onto a memory-bound engine) says a standalone microbench of a scan kernel would not settle the question; the real load on the real machine will. Honest correction to the framing: this is **not** "zero quota burned" — Kaggle bills session-hours, so 30 minutes of measurement costs the same 30 minutes wherever it sits; what folding actually saves is the session-setup overhead, and what it buys is validity. Say it that way in the log.

**Gate added before compile may run inside any gate-bearing arm — WS1/WS2 bought determinism and we do not give it back silently.** Inductor fuses and may autotune, so: (a) two fresh processes, same seed, compiled → **bit-identical**; (b) resume across a session boundary stays numerically continuous (pin the compile mode / cache or disable autotuning so session 2 compiles the same way as session 1); (c) compiled-vs-eager loss overlay within the same declared tolerance applied to AdamW-8bit; (d) net-positive over a 12 h session *including* warmup — and count the recompiles, since C→D→E→F change the model structure and each transition can trigger one. **If (a) or (b) fails, compile is out of gate-bearing runs entirely** and becomes a rung-boundary adoption item; the arms run eager. The A≡C bit-identity diagnostic is load-bearing apparatus and outranks a throughput gain.

**Item 3 — input streaming: does not fire.** The gate was "only if the profile shows it"; the profile shows 4.86 of 5.0 GB in the scan's two (B,L,Dn,N) tensors. Correctly not opened.

**Scan lane — the parked knob is refined and re-ranked (still parked).** The 768 MiB×2 per layer are now named precisely, and the Builder's variant — computing `dA`/`dBx` inside the loop instead of precomputing them — supersedes generic gradient checkpointing as the *preferred* form of that knob: the operations are elementwise, so it is **numerics-identical (bit-identical, not merely equivalent)**, taking memory from O(L·Dn·N) to O(Dn·N) at the cost of recompute. It stays parked under the standing rule (pull only if a config OOMs) — **and it is now explicitly the first instrument to reach for if the S2 optimizer-state question above turns into an OOM**, ahead of reducing batch.

## 11. Stage-1 screening verdict — adjudicated by the Architect (2026-07-22)

**Decision: V=2048 is ADOPTED, by the sealed decision rule — not by tie-breaker.**

The recovered decider reads: arm1_V2048 **P62 [DECIDING] = 1.1377**, arm2_V4096 **1.1443**, both with the byte invariant holding (1,499,998+2 = 1,499,997+3 = 1,500,000 B). Margin = **0.0066**. The prereg §4 rule, sealed at v1 and untouched since: *"adopt the lower code-val BPB if the margin > σ_seed; if ≤ σ_seed, adopt the pre-declared tie-breaker."* σ_seed = 0.005 (§3, R1 calibration). **0.0066 > 0.005 → the lower arm is adopted and the tie-breaker never fires.** The "2σ bar" quoted in the Builder's report is a mis-citation: 2σ_seed thresholds exist in §4.3 (KD trend) and in claim-grade language, but the screening *adoption* rule is σ_seed. Applying a 2σ bar here — after the numbers exist, in the direction that happens to agree with the forbidden tail-val metric — would be exactly the post-hoc renegotiation the seal forbids. Symmetrically: 1.3σ single-seed is **not a claim**; both numbers carry the screening label, and no "V2048 is better" claim is published. Adoption ≠ claim, by §4's own wording.

**The recovery is VALID as the stage-exit decider.** The metric is a pure function of (saved fp32 weights, pinned P62 stream, tokenizer); stage C is the only stage these arms ran, so the final checkpoint is the model whose P62 the trainer would have printed. Equivalence of the offline path is proven, not asserted: the same pipeline reproduces the Kaggle-printed tail val to |Δ| = 0.0000 on both arms, and the planted byte invariant holds on the P62 read itself.

**The tail val contradiction is v7's vindication, recorded with its caveat.** The in-corpus tail val pointed the other way (V4096: 1.1328 < 1.1452) — the metric the prereg forbids as decider would have selected the arm the decider puts higher. Direction matches the sealed mechanism (in-distribution fit ≠ temporal generalization), but at 1.3σ single-seed this is an observation, not a confirmed mechanism.

**Incident, on the record:** the uploaded data packages predated the P62 addition (6 files, not 7 — `p62_s0.u16` absent); the trainer treated the decider as optional (MVE legacy) and declared STAGE1-DONE without it. 18 GPU-hours produced weights — which is why nothing was lost — but the run answered only the forbidden question until the offline recovery. Root cause: the code/data separation froze "data packages are immutable" into practice while the one needed change *was* a data change. **Armor adopted: `--require-p62`** — refusal at startup, verified in both directions (flagless run reproduces the mute completion; flagged run refuses immediately), inherited by stages 2-3. A run can no longer declare itself done without its deciding metric. **Rider for stage-2/3 packaging: the cell pins the expected DATA manifest sha next to CODE_SHA** — `--require-p62` catches this incident's specific form; a generation-pin catches the whole class.

**Operational notes:** floor 3860 verified on V4096 (3907 tok/s); V2048 ran at 3653 — −5.4%, within the ±5-10% per-protocol session variance, and both arms cleared the 3,125 tok/s single-session threshold (~9 h each). bitsandbytes absent → fp32 AdamW, which is the declared recipe anyway.

**Next: stage 2 = one arm** — x_proj r=26 on V2048, fresh seed-paired init, same recipe, its chained control being arm 1's run (P62 1.1377). Pre-registered reading unchanged (§4.2): a null is the expected fading of a regularization benefit, and r26 is still adopted on byte grounds at ≤ σ_seed.

## 12. Stage-2 verdict + stage-3 branch decision — adjudicated by the Architect (2026-07-22)

**Stage 2 CLOSED: x_proj r=26 ADOPTED.** arm3 (r=26 on V2048) P62 [DECIDING] = 1.1376 in-run, byte invariant holds; arm1 (dense V2048) control = 1.1377. Δ = −0.0001 = **0.02σ** — inside σ_seed, so §4.2's byte tie-breaker adopts r=26: identical quality at 17.6% of x_proj's parameters. This is not even the "expected fading of a regularization benefit" the prereg tolerated — it is dead-identical, r26 marginally under dense. It confirms the Inventor's S1 finding (r=26 beat dense 3/3 seeds) at rung-1 scale, on code. Equivalence cross-check clean: arm3 offline reproduces in-run at |Δ| = 0.0000, so the arm1(offline) vs arm3(in-run) comparison is apples-to-apples. The P62 printed **in-run** this time — `--require-p62` + the DATA_SHA pin did their job; the stage-1 incident's armor paid on first use. Throughput 3938 tok/s (above the 3860 floor), stage C in 523 min, single-session. On the record: the first cross-check attempt ordered `.to(dev)` before the low-rank surgery, leaving new layers on CPU — benign (it touched only the cross-check, not the run), corrected, re-run.

**Stage-3 branch decision: the α arms INHERIT THE CHAIN (V2048 + r=26) and run as stage-C screening. They do NOT branch from the main run.**

The Builder offered two candidate parents — arm3, or "the main-run CE-primary". **The second is a category error, and naming it is worth more than dismissing it:** the main run is §5, it does not exist yet, and it runs the full D/E/F curriculum (α-QAT, MoE upcycle, context extension). Branching the α screening off it would measure KD-vs-CE in the presence of every other moving part, off stage C, which is the opposite of what a screening does. Every screening in this block isolates one variable on stage C; the α screening isolates α. So the parent is the chain's current winner, **arm3 (V2048 + r=26)**, exactly as the chained-controls discipline dictates.

**The α = 0 point of the curve is arm3 itself — but that identity must be VERIFIED, not assumed, and this is the stage-3 branch-point assertion.** arm3 already ran CE, on the logit-covered slice, at the seed the α arms will pair to, with the winning config. In principle it *is* the α = 0 point of §4.3's curve, at no extra cost. But the α arms run inside the KD harness (logit load + window sampling), and arm3 ran in the plain CE path. If the KD machinery perturbs the training RNG even at α = 0 — the exact failure class of the shared-`kdc` and unrestored-α bugs — then arm3-plain is not the control the curve needs. **Requirement: the KD sampler must draw from an RNG separate from the training RNG (init, dropout, batch order), and this is proven by a planted control — run the KD harness at α = 0, a few steps, and assert bit-identity with arm3.** Two outcomes: bit-identical → arm3 is the α = 0 point, nothing re-runs, the curve is {arm3, α0.25, α0.5}; not bit-identical → the bug is the RNG coupling, and it is **fixed in the harness, not papered over with a separate control arm**. The budget does not grow either way; this is the ε-identity discipline applied to the KD path, and it is a prerequisite before the α arms are believed, not a reading after.

**The two α arms (0.5, 0.25) run in parallel, seed-paired to arm3, config V2048 + r=26, on the same logit-covered slice**, consuming the 17.976 GB of teacher logits (the stage-3 packages are large where stages 1-2 were id-only). Read as the pre-registered §4.3 trend over α ∈ {0, 0.25, 0.5}, never a best-of. "Both winners" in §4.3 resolves to the single chain config now that V2048 has won stage 1 — there is one vocabulary winner, so one α curve, not two. The record-only stratified span-vs-CE-by-segment-length diagnostic is built at stage-3 packaging and reported from the production logs, per §4.3's mechanism check.

**Stage summary so far:** stage 1 → V=2048 (σ_seed rule, 1.3σ); stage 2 → x_proj r=26 (0.02σ, byte tie-breaker). The chain config entering stage 3 is **V2048 + r=26**.

## 13. Stage-3 α=0 control — the planted check produced a THIRD outcome (2026-07-22)

**§12 posed two outcomes for "is arm3 the α=0 point?"; the planted control produced a third, and it is the correct one. My "the budget does not grow either way" was wrong — it grows by one arm. Recorded as a correction, not absorbed.**

What the control found, three facts:
1. **The KD harness is bit-deterministic** (KD-α0 run twice → bit-identical weights) and batch positions are `default_rng([seed, rank, gstep, micro])`, independent of α (line 538). So the α curve is internally clean — α is the only variable across its points. This is exactly what the KD ε-identity asked for; this side is sound.
2. **arm3 (CE) ≠ KD-α0**, diverging at the very first tensor (`emb.weight`, max|Δ| 0.54) — not an RNG perturbation but training on entirely different positions.
3. **Cause, diagnosed not assumed: sampling domain.** `KDChunks(resident=2)` holds 2 of 121 chunks resident → sampling is a sliding window over the slice, while arm3 (`--arm ce`, no `--restrict-to-slice`) sampled the whole corpus. Different positions by construction.

**Why this is neither of my two outcomes.** The harness has no bug — it is deterministic and α does not touch sampling, so "fix the RNG coupling" does not apply. And arm3 is not the α=0 point — but not because of a fixable defect: arm3 runs CE-on-whole-corpus while the KD harness runs CE-on-logit-window, and the window is the streaming mechanism, unavoidable with 18 GB that cannot be resident. There is nothing to fix; it is how it must work. **This is, in fact, what §4.3's "CE as the α=0 point of the same curve" already required** — "the same curve" means the same harness and sampling domain, and my "arm3 = α=0" was the economical hope the control has now falsified. Using arm3 as the anchor would confound the α trend with a change of sampling domain precisely at the anchor point.

**Decision: ACCEPT the three-arm curve {KD-α0, KD-α0.25, KD-α0.5}** — three arms through the same deterministic harness, seed-paired, α the only variable. **arm3 does NOT anchor the curve.** Cost: **+1 arm** over the "arm3 = α=0" hope (~15% of stage C, ~9 GPU-h). This is the honest curve; the alternative saves an arm by confounding the reading it exists to produce.

**arm3 is retained as a record-only cross-check, not the anchor:** does CE-on-whole-corpus (arm3, 1.1376) match CE-on-logit-window (KD-α0)? If yes, restricting to the window introduces no domain bias and the α trend is read on a representative domain; if no, the window itself is a variable and that fact must be known before the trend is believed — a free positive control on the domain restriction.

**Re-price, declared in the brief (the prereg §9 is post-launch and not amended):** screening goes 5 → **6 arms** = 6×15% = 90% of stage C ≈ **49.5% of a rung** (was 41%); the rung inventory to ≈ **2.25 rung-equivalents, +125%** (was 2.2 / +116%); S0 to ≈ **250 session-hours** at the 3860 floor (was ~240). Still $0-path on the 4-account quota. Costs re-price, gates do not.

**Packaging notes carried:** the stage-3 package must contain `logits_s0/` (17.976 GB, slice_frac=1.0) and those bytes must be inside `data_sha256`; `pack_vocab` does not copy logits today and must be extended (the local control used a junction to the real logits, then removed — no 18 GB copy). **One planted check I want before launch: does the resident=2 sliding window traverse all 121 chunks ~uniformly across the step budget** (each chunk resident for a comparable fraction of steps), or does it dwell on early chunks? At ~0.67 passes the curve must still see a representative sweep of the slice, not a biased prefix — assert the per-chunk residency histogram before the arms are believed.

## 14. Stage-3 chunk-residence gate FIRED pre-launch — chunk_steps derived, not passed (2026-07-22)

**The §13 planted residence check fired before launch and is correct.** At the trainer default `chunk_steps=200`: 15106/200 = 75 advances = 0.62 sweeps → **44 of 121 chunks never resident** (the final 36% of the slice, chunks 77-120). The sparkline shows a clean full block to chunk 76 then empty — the α trend over {α0, α0.25, α0.5} would have been read on the first 64% of the corpus, confounding α with "which slice region": the exact failure the check exists to catch, caught pre-launch. Tool validated both directions: `chunk_steps=63` → 121/121 resident, CoV 0.058, PASS. No past result is touched — stages 1-2 were CE arms (`kdc=None`, whole-corpus sampling); only KD arms use the window, so V=2048 (1.1377) and r=26 (1.1376) stand.

**Decision: DERIVE chunk_steps from the budget, K=3 sweeps — do not pass it by hand.** `chunk_steps = steps // (n_chunks × K)` with **K=3** → 15106 // (121×3) = **41** for this stage (368 advances = 3.04 sweeps, below the CoV taper at the ring ends). Chosen over the terse `--chunk-steps 50` for the STEPS-grade reason: a screening parameter that the launcher must remember is one the launcher will forget, and the silent fallback (default 200) is precisely this failure. Deriving takes it out of the launcher's hands and **self-scales if the budget changes**. Second benefit that clinches it: since all three α arms share one budget, derivation guarantees they share one chunk_steps → one sampling domain → the one-variable curve, with no way for a per-arm edit to desync the domain. (Note: the proposed 50 is ~2.5 sweeps, not the "~3" quoted — a minor slip flagged so it does not propagate; K=3 derived gives 41 and the true 3.0.)

**The residence gate stays as an independent pre-launch check, not replaced by the derivation.** `ws3_chunk_residence.py` asserts every chunk resident with CoV under threshold before the arms are believed — because *deriving is not verifying* (n_chunks misread, off-by-one in the advance schedule), the same discipline as "recording a hash is not verifying it". Belt and suspenders.

**Cleared for the stage-3 batch:** SPECS[3] (three arms {KD-α0, KD-α0.25, KD-α0.5}, arm3 record-only cross-check), `pack_vocab` extended to include `logits_s0/` (17.976 GB) in the package and in `data_sha256`, derived chunk_steps wired in, residence gate as a pre-launch check in the batch. Push comes as one lot when built.

## 15. Stage-3 logit packaging — SHARED, and the reason is rigor not just bytes (2026-07-22)

**Decision: the teacher logits are a SHARED dataset (18 GB, one physical artifact, attached to all three α notebooks), pinned by a separate LOGITS_SHA verified in the cell — not copied into each arm's package. My earlier "in the package and in data_sha256" is corrected; the Builder was right to flag it before 36 GB of redundant upload.**

The decisive reason is not the 3× upload saving — it is that **one physical artifact makes the logits byte-identical across the three arms by construction**, which is strictly stronger than three copies each verified against a hash. The α curve {KD-α0, KD-α0.25, KD-α0.5} is a one-variable comparison only if every input except α is identical; a single shared logit file removes any path by which the arms could diverge on logits — the exact same argument that put the trainer code in a shared bundle rather than copied per package (and the same argument the stage-1 stale-package incident taught: a copied artifact is a place a wrong generation can hide). Efficiency and rigor point the same way here.

**Pin discipline, explicit:** LOGITS_SHA is *verified* against the artifact by the cell (a `check_logits` beside `check_code`/`check_data`), not merely recorded in a manifest — the stage-1 lesson ("recording a hash is not verifying it") binds all three pins equally. The arm's data manifest declares the LOGITS_SHA dependency so an auditor sees the arm consumes an external artifact, and the cell's pin ties them.

**Follow-on the packaging should resolve: the three arms differ ONLY in α (0 / 0.25 / 0.5) — same config V2048+r=26, seed-paired, same slice.** So the arm *data* (V2048 slice ids, anchors, t2s/decomp, BPE) is also identical across the three. If nothing else distinguishes them, the data bundle is shared too — one data input, not three — for the same physical-identity reason as the logits, with α the sole per-cell delta. If there is a reason the three data packages must differ (per-arm output dir, arm_id in the manifest), that belongs in the cell as metadata, not in three near-duplicate 360 MB datasets. Builder to confirm.

**Accepted as built (independent of the shared/per-arm call):** `chunk_steps` derived in-trainer (`steps // (n_chunks × K)`, K=3 → 41, `--chunk-steps 0` = derive, explicit value = test override, cells leave it 0); `--logits-dir` with default `--data-dir` (nothing breaks if logits are in the package); residence gate re-run on the real logits, derives 41, PASS 121/121. Minor observation, not a block: CoV rose 0.058 (at 63) → 0.064 (at 41) because 15106/(121·41)=3.045 sweeps leaves a fractional final sweep that over-covers the first ~5 chunks — the expected edge effect of a non-integer sweep count, benign and well under any threshold.

## 16. Stage-3 α curve read — D3 does NOT close; the domain cost is the finding (2026-07-22)

### The near-miss that has to be recorded first

Had arm3 anchored the curve — the economical hope of §12, which the planted branch-point check falsified — the curve would have read **1.1376 / 1.1677 / 1.1746: strictly monotone increasing, span 7.40σ.** That is an exact, high-confidence match to row 1 of §4.3's pre-registered table: *"KD hurts, and hurts more the more of it there is — D3 closes for good, with a mechanism rather than a single point."* We would have closed a design question permanently, with a clean monotone curve, on an artifact of sampling domain. **Two planted controls in series prevented it**: the branch-point assertion (born from the α bug) refused the anchor, and the record-only cross-check then priced what the anchor would have smuggled in. The extra arm accepted in §13 was not overhead — it was load-bearing.

### The curve, read as pre-registered

| α | P62 [DECIDING] |
|---|---|
| 0.00 | 1.1715 |
| 0.25 | 1.1677 |
| 0.50 | 1.1746 |

Span 0.0069 = 1.38σ. Pairs: α0.25−α0 = −0.0038 (0.76σ, KD side lower); α0.5−α0.25 = +0.0069 (1.38σ); α0.5−α0 = +0.0031 (0.62σ). Shape: shallow U, interior minimum at 0.25.

**Adoption: CE — and it is over-determined.** §4's rule (best 1.1677 vs the α=0 point 1.1715 → margin 0.0038 ≤ σ_seed) fires the §4.3 tie-breaker, which is CE at no logit cost. The trend reading lands in the same place. The main run stays CE-primary — which §4.3 had already sealed ("this screening does not flip the main run"). *Precision on the one pair above noise:* 1.38σ is above σ_seed (adoption-grade) but below the 2σ this document requires of a claim. "More KD than 0.25 hurts" is a weak directional observation, never quoted as a result.

**D3 does NOT close.** The Builder is right that no table row matches, and he was right to refuse to force one. The gap is mine: I enumerated non-monotonicity only in the convex-down direction (row 3's parenthetical, "α=0.5 better than α=0.25") and never enumerated the **interior-minimum shape with a sub-σ best point**. Where the table is silent, the **governing sentence above it governs** — and it is pre-registered prose, not a reading invented now: *"closure now requires a coherent trend, where before a single point sufficed."* A shallow U with everything but one pair inside noise is not a coherent trend. D3 therefore carries to rung-2 as **unresolved**, which is row 3's disposition for the whole non-monotone family. This is the conservative resolution: the gap is filled in the direction that denies us the strongest claim, not the one that grants it.

**A second, independent reason D3 must stay open:** the challenger ran in a regime that costs 6.8σ relative to the recipe it was being compared against (below). The α contrast is internally valid — all three arms share the window — but its **transfer to the i.i.d. regime the main run uses is untested**. A challenger that loses while handicapped has not been shown to be worthless.

### The domain cost — the actual finding

arm3 (CE, i.i.d. over the slice) **1.1376** vs KD-α0 (CE, sliding 2-of-121 window) **1.1715** → **+0.0339 ≈ 6.8σ, i.e. 4.9× the entire α span the stage existed to measure.** *(Corrected post-push by the Architect, flagged by MM: this line first read arm3 = 1.1377 / +0.0338 — 1.1377 is **arm1**'s value, inherited from a slip in the first stage-3 relay and corrected in the Builder's own follow-up. The committed re-derivation log is authoritative: arm3 = 1.1376, in-run and re-derived at |Δ| 0.0000. No decision moves; both round to 6.8σ.)*

The Builder closed the obvious confound before reporting: **it is not "saw less data"** — slice 117808/117808 windows, residence gate 121/121 at 3.04 sweeps, CoV 0.064, and total tokens are equal, so expected passes (0.669) and unique coverage (~48.8%) match on both sides. Same pool, same tokens, same steps. The residual difference is **order**: blocked sliding window vs global i.i.d. shuffle. This refines §13's "different positions by construction" — true per step, but the pool was never different.

**Mechanism NOT asserted** — correctly, and I hold that line. Two alternatives must be separated before this is called an ordering law: (i) **recency bias** — under a sliding window the final ~82 steps see only chunks 119-120, so final weights tilt toward an unrepresentative ~1.6% of the slice, which an external val punishes; (ii) **position-grid restriction** — if KD sampling draws only positions aligned to teacher windows while arm3 draws any offset, context diversity differs independently of order. Both are cheap to distinguish.

**Design irony worth recording:** §4.3 chose full logit coverage deliberately, *"sized for maximum statistical power and maximum favorability to KD"*. Full coverage forces all 18 GB to stream, which forces the window. **The choice made to maximize KD's chances is what imported the handicap.** The deployment form specified in `PHASE64_TRAINING_PLAN.md` §2 (KD-on-subset + CE-on-rest) would dilute this roughly in proportion to the KD fraction — so the shipping form was never exposed to the full cost the screening paid.

### Decisions

1. **Re-derive arm3's P62 from its checkpoint — yes, now** (minutes, zero GPU). The 6.8σ anchor currently rests on a number that exists only in adjudication documents; `ws3_recover_p62.py` turns a citation back into an artifact. **New standing rule: every [DECIDING] number lives in a committed log under `results/`, and the brief cites the log path.** An adjudication is authority, not evidence. *Question back:* is arm3's log absent because it was never saved, or because an ignore rule swallowed it? The fixes differ, and the second is the `.gitignore` incident's family again — check it against the pending anchoring audit.
2. **Stratified span-vs-CE-by-segment-length diagnostic: DO NOT build it.** Its pre-registered purpose was to verify the chain-rule mechanism *if span won*. Span did not win, and the effect it would decompose is 0.76σ — slicing noise into strata yields noise. Recorded as consciously dropped with its reason, not silently skipped; it returns if rung-2 ever produces a separating KD result.
3. **Order probe: ONE record-only arm, in parallel with the main run, gating nothing.** Preferred form is a **dose-response, not a binary contrast**: KD-α0 at a wider resident window (e.g. resident=16, ~2.4 GB) against the existing resident=2 point (1.1715) and the i.i.d. point (1.1376). If BPB moves monotonically toward 1.1376 as the window widens, window width is the mechanism and we also learn how wide is wide enough. The main run occupies one account at a time, so this rides on idle parallel capacity — free in calendar terms.
4. **`--expect-slice-sha` closed at the next stage.** Printed but not enforced, identity confirmed by human observation rather than machine assertion — "recording is not verifying" a third time. Mitigating fact, and not luck: the shared-artifact packaging of §15 made the three arms read **one physical slice and one physical logit set**, so they *could not* have differed. The packaging choice supplied physically the guarantee the missing flag would have enforced. Close it anyway.
5. **Candidate remedy for rung-2, hypothesis not instruction: shuffle before chunking.** If the slice were shuffled at window granularity *before* the logits were chunked, each chunk would be a random sample rather than a contiguous block, a 2-chunk residency would be a random 1.66% rather than a contiguous region, and the recency tilt would largely vanish — one disk pass over 18 GB, no GPU. Builder to assess feasibility (it needs a position index); this is the cheapest known path to KD-without-the-handicap.
6. **Scale-up implication, held provisional until the probe reports:** if ordering is confirmed as the mechanism, it is the largest single effect measured at rung-1 — larger than vocabulary (1.3σ) and x_proj (0.02σ) combined — and it lands directly on `SCALEUP_ARCHITECTURE.md`'s data path, where 10B-scale training *must* stream from disk. It does **not** enter that document on today's evidence.

### Screening block CLOSED

| stage | question | verdict | metric |
|---|---|---|---|
| 1 | vocabulary | **V=2048** (σ_seed rule, not tie-breaker) | 1.1377 vs 1.1443 (1.3σ) |
| 2 | x_proj r=26 | **adopted** (byte tie-breaker) | 1.1376 vs 1.1377 (0.02σ) |
| 3 | span-KD α trend | **CE adopted; D3 unresolved, carried to rung-2** | best-vs-CE 0.76σ; domain cost 6.8σ |

Chain recipe for the main run: **V=2048 + x_proj r=26 + CE-primary**, i.i.d. sampling, 1.5 B student tokens, full A→F curriculum with the §2 declared deviations. **The main run is cleared to launch.**

## 17. Returns on §16's five decisions + the hardcoded-threshold incident (2026-07-22)

**(B) The missing table row was already adjudicated in §16, before this report arrived: D3 does NOT close.** Recorded here because the Builder reached the same disposition independently, from the numbers, without having seen §16 — he tested all four rows, found none matching, refused to force one, and volunteered that he would be cautious about permanent closure for the domain-cost reason. Two independent paths to the same reading is worth more than either alone. The formal ground remains §4.3's governing sentence where the table is silent ("closure now requires a coherent trend"), and the enumeration gap — interior minimum with a sub-σ best point — is mine.

**(A) ε-identity check before the probe: APPROVED, run it.** Adding `--kd-resident` changes CODE_SHA away from `232267f2`, and the probe's entire value is a comparison against arm4, which ran under `232267f2`. By inspection the flag is neutral (default 2 = the previously hardcoded value); but the standard here is not inspection, and this report contains the reason why in its own body — a hardcoded constant that looked fine and was not. New code at `--kd-resident 2` vs `232267f2`, same seed, a few hundred steps, bit-identical weights. Minutes, and it converts an assumption into a fact.

### The hardcoded-threshold incident — banked, with its counterfactual

`ws3_recover_p62.py` carried `'decisive' if abs(b1-b2) > 0.010 else 'INSIDE noise'` — the 2σ bar I corrected on 22 July, still wired into the tool, printing on every run that the stage-1 delta of 0.0066 was *not separable*. **Counterfactual, computed: under that rule stage 1 would have been a tie → the §4.1 tie-breaker fires → V=4096 adopted. The tool encoded a rule that selects the opposite vocabulary from the sealed one.** Nothing downstream consumed it, because adjudication reads the document and not the tool's opinion — but the tool would have supplied a confident, quotable second opinion contradicting the seal.

Note the shape: this is the *same* mis-citation that appeared in the stage-1 report and was corrected there. Correcting a person's reading left the tool repeating it. **Law adopted, in the Builder's words: a lesson written only in a document does not protect the code that violates it. When a threshold is corrected, grep the hardcoded thresholds in the tooling before calling the correction closed — a threshold inside a tool is a paraphrase-from-memory that repeats itself forever.**

The fix is right in structure, not just in value: **ADOPTION (§4, σ_seed) and CLAIM-GRADE (2σ) separated into two questions**, because conflating them is what produced the bug; and validated against three already-adjudicated known-positives, all three reproducing the Architect's calls. Grep extended across `benchmarks/phase64` — it was the only wired decision threshold.

### Decision 1 — re-derivation: DONE, anchor holds

arm1 offline tail-val 1.1452 = Kaggle 1.1452 (|Δ| 0.0000); arm2 1.1328 = 1.1328 (|Δ| 0.0000); **arm3 P62 [DECIDING] 1.1376, |Δ| 0.0000 against the brief-recorded value**, byte invariant holding. Log: `results/phase64_rung1/p62_rederivation_arm123.txt`.

**The verification design deserves naming, because it is stronger than what was asked:** the two known-positives were reproduced *in the same invocation that produces the unknown*, so the pipeline is proven faithful in the session that generates the number, not in a separate session whose equivalence would itself be an assumption. That is the planted-control discipline applied to a recovery path. The 6.8σ anchor is now an artifact: **1.1715 − 1.1376 = 0.0339 = 6.78σ.**

**The `.gitignore` diagnosis — third occurrence, and the two-layer analysis is the valuable part.** The log was never written to disk *and* its home was invisible to git at two layers (`kaggle_rung1/*` plus unanchored `results/`). His reading is exactly right: **the second layer is why nobody noticed the first — in a tracked tree, a missing log is an obvious hole.** Invisibility does not merely hide a file; it removes the contrast that would have made its absence visible. Fix `results/*` + `!results/phase64_rung1/` accepted, blast radius verified zero. **Standing instruction: finish the whole anchoring audit now** — the pending list still names patterns unaudited, and three incidents of one family is past the point where fixing them one incident at a time is a strategy.

### Decision 3 — the probe, and its reading pre-registered BEFORE the number exists

Built as arm7 (`--kd-resident 16`), identical to arm4 in every other respect, bundles not rebuilt. Residence CoV 0.035 vs 0.064 at resident=2 — coverage stays full, so a BPB difference cannot be attributed to seeing less. **[Correction, MM 2026-07-29: this paragraph describes the arm as PLANNED. It executed at r=8 (OOM, §18), so the residence figures here — and with them the load-bearing "coverage stays full" argument — are anchored to a gate run at r=16, not to the configuration that produced 1.1446. The gate must be re-run at r=8 and the number recorded here. It is expected to pass (the advance schedule is independent of residency width: chunk_steps stays 41, the ring is still swept 3.04 times, so CoV should land between 0.064 at r=2 and 0.035 at r=16) — but "expected to pass" is not the standard, and the run costs seconds of CPU. **Reading fixed in advance: if r=8 residence PASSES, the coverage explanation stays excluded and §18 is unchanged; if it FAILS, coverage reopens as a rival explanation for arm7 specifically and the 79% recovery figure must be re-read before it is quoted again.**]** RAM 2.39 GiB/process, 4.77 GiB under DDP×2; fallback resident=8 is a legitimate third curve point, not a failure.

**Economy worth naming: this gives three points, not two — arm3 is the r→∞ (i.i.d.) limit of the same curve.** So one new arm buys a dose-response across r = 2 / 16 / ∞ = 1.1715 / ? / 1.1376.

**Pre-registered reading, fixed here so the probe cannot be narrated after the fact** (record-only, gating nothing):
- arm7 **more than σ_seed below 1.1715** (i.e. < 1.1665) → window width is a mechanism; the dose-response is real; **shuffle-before-chunking becomes the preferred rung-2 remedy**, since it buys the same effect at O(1) RAM instead of O(width).
- arm7 **within σ_seed of 1.1715** → width is NOT the mechanism; the cost lies elsewhere in the KD path, with **position-grid restriction** the leading suspect (KD may sample only teacher-window-aligned offsets while arm3 samples any offset), and the remedy changes accordingly.
- arm7 **materially above 1.1715** → unexpected; record it, do not theorize.

**Cost declared, not absorbed:** the probe is record-only and is **not** a screening stage — the screening block is closed at three stages / six arms. Bookkeeping only: 7 arms × 15% = 105% of stage C ≈ 58% of a rung; rung inventory 2.25 → **≈2.33 rung-equivalents**; S0 ≈ 243 → **≈252 session-hours** at the 3860 floor. It runs on capacity that would otherwise idle while the main run occupies one account.

### Decisions 2, 4, 5 — accepted as reported

2. Stratified diagnostic **not built**, dropped by decision with the reason on the record. 4. `--expect-slice-sha` wired as the fourth pin; the collateral discovery (the flag existed, the cells simply never passed it) is the same family as the manifest hashes that were recorded and never read. **His placement argument is correct and I adopt it: the slice-sha belongs to the logit manifest, not the data manifest — "which windows carry teacher signal" is a property of logit generation.** 5. Shuffle-before-chunking registered as a rung-2 hypothesis; his observation that it pairs with the probe outcome is right and is now folded into the pre-registered reading above.

### One question back (cheap, not a blocker)

Throughput reads 4017 / 4055 / 3687 tok/s for α = 0 / 0.25 / 0.5. The α=0 arm is *not* the fastest, which is what one would expect if the KD block were genuinely skipped at α=0. Either it is computed and multiplied by zero — in which case all three arms share one code path exactly, which strengthens the one-variable claim — or it is skipped, in which case the α=0 arm differs from the others by a (tiny) second variable. The 9% spread is inside the declared ±5-10% per-protocol session variance and **does not touch the comparison**, which is read at equal steps and not equal time. Worth one line of confirmation, not a re-run.

## 18. Order probe — the pre-registered reading fires; promoted to architecture (2026-07-29)

| resident window | P62 [DECIDING] | vs i.i.d. |
|---|---|---|
| r = 2 (arm4) | 1.1715 | +0.0339 = 6.8σ |
| r = 8 (arm7) | 1.1446 | +0.0070 = 1.4σ |
| r = ∞ (arm3) | 1.1376 | — |

**Branch (i) of §17's pre-registered reading fires: window width IS a mechanism.** Threshold was < 1.1665; the measurement is 1.1446, i.e. **4.4σ past it on a monotone three-point curve**. Recovery 0.0269/0.0339 = **79% of the domain deficit at 6.6% of the ring resident**; residual to i.i.d. is 1.4σ, inside single-seed noise. Consequently, and as pre-registered rather than as a post-hoc preference, **shuffle-before-chunking is the rung-2 remedy of choice** — same property at O(1) memory instead of O(width) — and we now also know *how much* width suffices, which was the half of the question a binary contrast would not have answered.

**The OOM deviation is conservative and makes the result a fortiori.** The probe ran at r=8 instead of the planned r=16. A narrower window should recover *less*; crossing the threshold anyway strengthens the conclusion rather than weakening it. Declared, not absorbed.

**Attribution closed separately, and in both directions.** The concern that the −0.0269 might come from the code change that exposed `--kd-resident` rather than from width is answered three ways: the diff is one of nine bundle inputs, 9+/2−, substance = an argparse entry plus passing the already-hardcoded default; the pinned ε-identity gate shows the new trainer at r=2 **bit-identical** to `d145a01`; and the same gate shows r=4 vs r=2 **measurably different** (max|Δ| 0.18). Bit-identical where it must be, different where it must be — an instrument validated on both a known-negative and a known-positive. `kd_resident` is not caching and not KD mathematics (at α=0 the KD term does not enter the loss); `KDChunks.window()` delimits which positions are sampled, so it is data ordering by construction. **arm7 is therefore not a substitute for a missing r=16 point — arm7 is a legitimate point of the curve.**

**Retraction handled correctly, and the distinction is the valuable part.** The Builder withdrew his own earlier PASS on discovering three defects in the gate apparatus (moving `HEAD` reference; anti-tautology control comparing raw bytes under `core.autocrlf`, so it could never fire; `subprocess(text=True)` decoding a UTF-8 file as cp1252, producing a byte-corrupted "old trainer" that still parsed — and which was *also* the reason the second defect stayed invisible). His framing is the one to keep: *"I am not saying it is wrong — I am saying it is not evidence."* A verdict that would probably survive re-derivation is still not a verdict. The clean re-run reproduces the same `max|Δ| 0.1806`, which is confirmation and not vindication. Lessons banked in `feedback_planted_controls`, including the fourth, on judgement: a risk dismissed must be dismissed **on a declared side** — the shared resume path was correctly judged harmless on the read side and wrongly on the write side.

### Consequence 1 — D3's disposition is reinforced, and nothing moves

§16 kept D3 open for two reasons; the second was that the challenger ran in a regime whose transfer was untested. That reason is now **quantified and shown to be largely removable**: the α curve was measured under a handicap that is ~79% recoverable at near-zero cost. This changes no decision — CE ships (sealed, and §4.3 always denied this screening the power to flip the main run) and D3 was already unresolved — but it converts "untested transfer" from a caveat into a specified, cheap experiment. The α *contrast* remains internally valid: all three arms shared r=2, so the handicap is common-mode. What is untested is whether a teacher signal's value survives — or is masked by — a degraded-ordering regime.

**Rung-2 form, specified now while it is cheap to specify: re-run the α trend at r=8 (or on shuffled chunks) — and it costs TWO arms, not three, because arm7 *is* the α=0 anchor of that curve** (α=0, V2048, r=26, r=8, same bundles). The probe arm does double duty. Not ordered now: the main run is the priority and §4.3 sealed D3 as a rung-2 question.

### Consequence 2 — promoted into `SCALEUP_ARCHITECTURE.md` as §5.1

§16's decision 6 held this finding provisional *"until the probe reports"*. It has reported, and the promotion condition is met, so it is now written into the architecture as a constraint on the training data path: at 10 B the corpus must stream, the naive form of streaming *is* the blocked-residency pattern measured here, and the guarantee to engineer is that **the data path presents an i.i.d.-equivalent sample at every point in the run — obtained by randomizing membership when the artifact is written, not by buying residency at run time.** Scope stated in place: measured at rung-1, single seed per point; the mechanism is beyond reasonable doubt at this scale, the magnitude at 10 B is an extrapolation and is not claimed.

### Push and one flag

To push: `.gitignore` (anchoring audit closed), `ws3_epsilon_identity.py` (three fixes), `epsilon_identity_kd_resident.txt` (**replaces** the retracted log — correct: a withdrawn artifact should not survive next to its replacement), `arm7_w8_console.txt` (the probe's decider — and the first file to land under the newly-anchored `results/phase64_rung1/`, which is the rule from §17 working on its first real use), plus this brief and `SCALEUP_ARCHITECTURE.md` §5.1.

**Flag, not a block: `scripts/kaggle_run.py` is modified and unowned by the Builder.** It is presumably the owner's launch-side edit. An unattributed modification riding along in a gate-bearing push is the shape that has bitten us repeatedly this week — **the owner confirms what it is before it goes, or it stays out of this push.** Cheap either way.

## 19. Main run — launch record, decisions A and B (2026-07-29)

### Residuals closed

Residence at **r=8: CoV 0.048, 121/121, PASS** — between r=2 (0.064) and r=16 (0.035), the ordering the mechanism predicts. The coverage explanation stays excluded and §18 is unchanged; the pre-registered FAIL branch of §17 does not fire. Log `results/phase64_rung1/chunk_residence_arm7_w8.txt`. Rename done, no pin broken, `SPECS[4]`/`KD_RESIDENT` updated with the deviation declared beside them (**w16 attempted → OOM → w8; w16 remains UNMEASURED and is never quoted as if it were**).

**The two rename residuals are handled correctly and the reasoning is the right one:** `cfg['out']`, `cfg['save_stage_ckpt']`, `cfg['resume_ckpt']` inside the checkpoint and the `RUN:` line in the log still read `w16` — **they are the record of what was invoked, and rewriting them would falsify it.** A record is not a label. And the artifact is self-describing on the variable that matters: `cfg['kd_resident'] = 8` is inside the checkpoint, so an auditor reads the true width without trusting a filename. That is the correct resolution of a naming/record conflict: fix the label, never the record.

### Decision A — corpus: 5.5 GB, APPROVED, with the contract made load-bearing

Arithmetic verified independently: 5.5 GB ÷ 2.571 B/tok = **2.14 B token pool**; 1.5 B ÷ 2.14 B = **0.70 expected passes**, unique coverage 1 − e^(−0.70) = 50.3%. That sits alongside the screening's declared 0.669 / 48.8%, so the main run trains in the same repetition regime the recipe was validated in — the reason to prefer it over 6.5 GB, which buys headroom we have no declared use for at the cost of ~1.3 h of CPU and ~0.8 GB of upload. 4.2 GB is correctly rejected: 0.92 is too close to the wall for a run that will be resumed a dozen times.

**Addition, and it is the week's own lesson applied: the single-epoch contract is currently prose in the prereg, and prose does not enforce.** Wire an assertion at startup — `steps × tokens_per_step / n_train_tok < 1`, refuse otherwise, and **print the computed ratio in the run header** so every session's log carries it. A declared contract that no consumer makes load-bearing is documentation, not defence; this one costs one line and guards the whole run.

### Decision B — conditional KD apparatus: APPROVED, and the gate must run in CE mode

~9.5 GB of anchors/t2s/decomp are dead weight for a CE-primary run, and they are dead weight **for the §6 branches too** — none of the three (ε-identity upcycle, no-recall control, dense-paired) uses KD — so the saving applies to the whole remaining rung, not just one upload. Three lines against 10 GB per upload, on a path that repeats across relay accounts, is the right trade.

**Specification of the gate run, because the obvious way to run it would prove nothing.** The changed behaviour is *skipping the load*, so the informative test is a **CE-mode run**: KD-mode still loads the arrays and is therefore trivially unchanged — testing that branch would be testing the code that did not change. Required: (a) CE-mode bit-identity, old vs new, pinned immutable sha, extracted in bytes, VOID control exercised; (b) KD-mode bit-identity as the cheap second point. The hypothesis being falsified is specifically *"loading those arrays touches nothing else"* — plausible, and this week has shown what plausible is worth. **Confirm before you write it that the condition is keyed on KD usage alone**, and that nothing in the recall tier or the P62 path reaches those arrays.

### Main run — launch record

**STEPS = 183,105** at 8,192 tok/step (micro 8/rank × world 2 × accum 1, seq 512; effective batch 16). Curriculum C 100,707 / D 36,621 / E 36,621 (seq 2048, B·L constant) / F 9,155; α-QAT ramp 3,600 (≤10% of D ✓, 3,662 being the cap). Config **V2048 + x_proj r=26 + CE-primary**, `--recall on`, `--sparse-moe`, `--require-p62`, `--save-stage-ckpt`, four pins, `--expect-slice-sha`.

**Calendar, stated so it is committed to with open eyes:** 1.4996 B tokens ÷ 3,860 tok/s = 108 h, **+34% for the stage-E extension → ≈145 h ≈ 14 sessions** at `--time-budget-min 660`. Measured throughput ran 3,900-4,050, so 13 is likelier than 14. **This exceeds one account's weekly GPU quota, so the resume chain must rotate across accounts** — the apparatus already supports it (`kernel_sources` relay, resume proven bit-identical). The Kaggle operator confirms the quota arithmetic and plans the rotation before session 1, rather than discovering it at the quota wall mid-run.

**Two things wired at build time, not later:** `--save-stage-ckpt` must emit the stage-D checkpoint that §6's three branches fork from; and the **branch-point assertion is automatic from the start** — fork entry BPB must equal parent exit BPB, asserted, any branch failing it is VOID. That rule exists because an unrestored α once produced an fp32 control against a ternary arm with an entirely plausible BPB, and the branches are where it would happen again.

### `scripts/kaggle_run.py` — classified operational, cleared, with one sharpening

The operator's classification is accepted and his method is right: he grepped rather than asserted, and he **named the one seam instead of returning a clean "no"** — `machine_shape="NvidiaTeslaT4"` → `device_count=2` → `--accum 1`. Fixed, identical across arm4-arm7, effective batch invariant.

**Sharpening, because "math-neutral" is very slightly stronger than what holds.** The *step grid and effective batch* are invariant across (world 2, accum 1) and (world 1, accum 2) — that is the comparability claim the prereg makes and it stands. **Floating-point summation order is not invariant**: DDP gradient averaging and local accumulation sum in different orders, so a session on a different accelerator is *comparable but not bit-identical*. Over a 14-session resumed run this matters: **the accelerator must stay T4×2 for the whole main run, and any session that gets something else is a declared deviation, not a silent one.** Make it load-bearing — assert `device_count == 2` and the device name in the cell, and refuse otherwise. Kaggle substituting an accelerator is precisely the kind of silent substitution the cell should catch in two seconds.

Commit line approved as drafted; it names the accelerator, which is what keeps the seam on the record.

---

## 20. Main-run corpus build — the memory wall, and why both proposed exits die (2026-07-31)

### The measured wall is real; it is not the only one

The Builder's diagnosis of `teacher_tokens` is correct and independently verified: `tids`/`tstart`
accumulate as **Python lists** (`code_data.py:96-105`), ~76 B per token where numpy holds 12. At the
5.5 GB target that is 1.47 B teacher tokens × 76 B = **104 GiB**, against 9.2 GiB at the 0.5 GB slice
where the component was validated. Two kills at the same point is the signature of a memory ceiling,
not of time, and it is why warming the corpus cache did not help. All of that stands.

**Reading the rest of the surviving path shows two further allocations of the same or larger size, both
on the branch that Option 2 keeps.** Projected at 5.5 GB (2.139 B student tokens at the measured
2.5706 B/tok):

| line | allocation | at 5.5 GB | on which path |
|---|---|---|---|
| `126` | `pool.map` returns `parts` = list of ~2.14 B Python ints, then a list comprehension builds a **second** full list before `np.array` consumes it | ~77 GiB **× 2** | **student** — survives Option 2 |
| `135` | `b2j = np.full(N+1, -1, dtype=np.int64)`, `N` = corpus **bytes** | **41.0 GiB** | teacher/anchor — removed by Option 2 |
| `130` | `ends = np.cumsum(exp_len[ids])`, plus the `exp_len[ids]` temporary it materialises first | 15.9 GiB × 2 | student, but only feeds the anchor read and one assert |
| `121/126` | `ids` held as **int64** rather than uint16 | 15.9 GiB (vs 4.0) | student |

Consistency check against the one run that worked: at 0.5 GB the student path costs ~14 GiB and the
teacher path 9.2 GiB, peak ~23 GiB — which is why it completed. Scaled to 5.5 GB the **student** path
alone exceeds the box, and it exceeds it by more than the teacher path does, because there are more
student tokens than teacher tokens.

**Consequence for the two options as written.** Option 1 (memmap the teacher accumulator) leaves
`b2j` at 41.0 GiB and line 126 at ~77 GiB: it dies. Option 2 (drop the teacher apparatus) removes
`b2j` and the teacher lists but leaves line 126 untouched: it also dies, later and for a different
reason. The stated Option-1 peak of "~22 GiB instead of 110" is a projection for one array, not for
the path.

### Decision — Option 2, extended to the student path

**Option 2 is adopted**, for the Builder's three reasons — the arrays are exactly the ones Decision B
(§19) already ruled we would not load; the main run and all three §6 branches are CE, so no consumer
reads them; and a code change is now unavoidable either way, so "let it run" is no longer on the
menu — **plus a fourth that closes the option-value argument rather than deferring it.** Any rung-2
reopening of KD needs the corpus **rebuilt regardless**: `SCALEUP_ARCHITECTURE.md` §5.1, promoted two
days ago, requires membership randomised **when the artifact is written**, and these arrays are built
on exactly the sequential r=2 ordering the probe priced at +0.0339 (6.8σ). Preserving them preserves a
version we have already declared unusable. The option value is ≈ 0, not merely cheap to re-pay.

**Scope of the change, which is larger than "don't build the teacher arrays":**
1. remove the teacher tokenisation, `b2j`, `anchors`, `t2s`, `decomp` (Option 2 as proposed);
2. stream the student encode — `pool.imap` (ordered) writing each block as uint16 to the ids file,
   then `np.fromfile`, so neither `parts` nor the comprehension is ever fully materialised;
3. keep `ids` as **uint16**, and replace `ends = np.cumsum(...)` with a **chunked sum** — under
   Option 2 `ends` has no consumer except `assert ends[-1] == N`, and a sum answers that.

Projected peak after all three: `raw` 5.1 GiB + `ids` 4.0 GiB + one block ≈ **10 GiB**.

### Two gates, both pre-registered here before the code is written

**(a) Peak, projected then measured.** Before writing the fix, produce the projected peak for **every**
surviving allocation at 5.5 GB — the table above is the form — and during the build print a **running
peak RSS** line. Rationale is the Builder's own method note applied one step further than he applied
it: the "~22 GiB" estimate was a model of one array at a scale where nothing had been measured, made
in the same message that correctly named that habit as the defect. A projection that covers one array
is how we got here; a build that dies silently a third time teaches nothing.

**(b) Bit-identity of the student side, old vs new, at 0.5 GB.** The hypothesis being falsified is
*"removing the teacher branch touches nothing the student path produces"* — plausible, and this week
has priced what plausible is worth. Compare `ids_V2048_*.u16`, the document boundaries, **the P62 val
split and the slice sha** between the old and the new builder at the 0.5 GB slice, where the old path
is known to run and the reference artifacts already exist from s0. Planted control: corrupt one byte
of `raw` and confirm the comparison fires; a comparison never seen fire is not a comparison.
The P62 decider must be inside the compared set explicitly — `--require-p62` is the deciding metric of
the whole rung and it lives in the same builder as the code being cut.

### Method

Banked in [[feedback_planted_controls]]. The Builder retracted his own "let it run" twice-given
recommendation, named the cause correctly (a component validated at 0.5 GB and assumed at 5.5, the
same form as the 2.12×), and did so unprompted. That is the behaviour the process wants. The
correction here is not to the diagnosis but to its reach: **the scale-assumption law applies to the
remedy as well as to the defect**, and the second wall was visible in the same file, twelve lines
below the first.

---

## 21. Main-run corpus built; trainer authorised for the last O(corpus) site (2026-07-31)

### Both gates pass, and gate (a)'s agreement is not a coincidence

**(b) bit-identity.** Arm A (new code, `--kd-apparatus`): 10/10 artifacts bit-identical to s0, both
vocabularies — the rewrite changed how accumulation happens, not what comes out. Arm B (default):
2/2 ids identical, 0/8 KD files built — the flag removes work without altering the artifact. Arm C
(planted control): one corrupted corpus byte → ids differ. The control fires, so A and B inform.
Log `results/phase64_rung1/code_data_identity_gate.txt`.

**(a) peak.** Projected 11.0 GiB, measured 11.2. **The Builder supplied the argument that makes this
evidence rather than agreement:** the token count was extrapolated from s0 and landed within 0.45%
(2.2957 e9 measured vs 2.306 e9 projected), so the peak matches because the model is right, not
because two errors cancelled. That is the correct way to read a projection that lands.

### Corpus constants — recorded here because a re-derivation from "5.5 GB" will not reproduce them

`--gb` is **GiB**: 5,905,582,016 B = 5.50 GiB = 5.906 GB. 1,163,820 docs, **2,295,693,322 student
tokens** (2.5725 B/tok, against 2.5706 measured on the disjoint sample), `raw_sha256 dec356ca…`.
Expected passes 1.4996 B / 2.2957 B = **0.653**, unique coverage 47.9%.

Decision A (§19) was approved on a projected 0.70 / 50.3%, priced from 5.5 GB **decimal**. The
delivered corpus is ~7% larger, so the realised figure moves **toward** the screening's 0.669 / 48.8%
rather than away from it: the premise the approval rested on — main run trains in the repetition
regime the recipe was validated in — is strengthened, not weakened. No re-approval needed. The unit
is recorded explicitly because "5.5 GB" and "5.5 GiB" differ by more than the margin we reason with.

### `mve_train.py` — authorised, with the gate the file deserves

`mve_train.py:325` loads `ts_*.u16` and casts to int64: 4.28 GiB on disk → **17.10 GiB in RAM, per
rank**, so 34.2 GiB resident under DDP×2 and up to 42.8 at peak, on a ~29 GiB Kaggle session. It does
not start. This is the last O(corpus) site on the inventory and the Builder was right to stop rather
than patch it unasked — the trainer is the file every number of this rung comes out of.

**The alternative is not available, and the reason is stronger than "it breaks the contract".**
Cutting the corpus to ~1.6 GB moves expected passes from 0.653 to ~2.4, which **trips the
single-epoch assert wired at Decision A** (`steps × tokens_per_step / n_train_tok < 1`, refuse
otherwise). The run would refuse to launch. Independently of that, it trades a memory bug for a
data-repetition bug, and the second one surfaces in the metric rather than in the log.

**Change:** keep the ids resident as **uint16** and cast **per batch** — `nn.Embedding` needs int64
for the batch, never for the corpus. Do not add memmap: it would make the resident copy shared
between ranks via page cache, which is strictly better, but it is an optimisation against a budget we
already meet, and it introduces a new variable into the file we are being careful with. Hold it in
reserve for the case where the measured peak exceeds projection.

**Gate — reuse the validated apparatus rather than build a weaker one.** The proposed reading
("identical loss trace and P62 BPB to the last decimal") is the right bar but a printed number is a
rounded one. Use `ws3_epsilon_identity.py`, whose defects were closed on 2026-07-29: run K steps at
s0 scale on one seed, old vs new, and compare the **saved checkpoint sha256**. If the data path is
unchanged the trajectory is deterministic-identical, so one hash answers the whole question and
answers it in bytes. Conditions carried over unchanged: reference pinned to an **immutable sha, never
`HEAD`**, and printed in the artifact; extraction in **bytes**; anti-tautology control exercised in
both directions. s0 scale is the correct choice for the reason the Builder gave — it is where the old
code still runs.

**Three additions, each closing a failure this rung has already produced once:**
1. **Project the host total, not the per-rank figure.** Both ranks share one machine, so the number
   that matters is `2 × per-rank`. Same table form as §20: every O(corpus) allocation, plus torch,
   CUDA host allocations, the P62 val array and the recall index — including the ones that fit.
2. **Assert RAM headroom at startup**, beside `--expect-gpus` / `--expect-gpu-name`. The arm7 incident
   is exactly this: all guards pass, then a host SIGKILL with no diagnosis attached to the cause. Over
   a 14-session resumed run a refusal in two seconds is worth more than a correct projection.
3. **Report tok/s old vs new on the gate run**, declaring any regression > 2%. The 3,860 tok/s figure
   carries the 145 h calendar, which carries the account-rotation plan.

**Consequence that must not be discovered later: the trainer changes ⇒ CODE_SHA changes ⇒ the stage-5
package already built (`main_V2048_r26_CE`, DATA_SHA `79c58b30…`) carries the old trainer and must be
rebuilt.** This is the stale-package incident of 2026-07-22 in its exact original form, and it is
cheap only if it is remembered now. Push before run, gate log included.

### Two instrument failures, opposite in sign — and the sibling that is still open

`peak_gib()` reported **0.0 GiB for an 11 GiB build**: `GetProcessMemoryInfo` called without
`argtypes`, so the `-1` pseudo-handle was truncated on a 64-bit `HANDLE`, the call failed, and the
zeroed struct was read back as a measurement — no exception. Second blind instrument of this rung
after the anti-tautology check neutralised by CRLF; neither raised, both returned a plausible number.
Handled correctly: fixed, validated on a known positive **before** the real build (0.029 → 3.029 GiB
on a touched 3 GiB allocation), and the correction **appended to the log rather than rewritten**.

Opposite sign, same day: `--gb 0.5` no longer reproduces `raw_s0.bin`, because the WS3 shards finished
downloading after 21/07 — the same command now yields 512 MiB over 105,920 docs instead of
483,190,011 B over 95,564. The gate would have reported a difference having nothing to do with the
code under test. Seeding the file and verifying its sha is the right fix, and the general rule is
worth stating: **a tag is not a reproducible specification; a content hash is.**

**Still open, found by applying the fifth case's rule — when you correct one instance, grep for its
siblings.** `code_data.py:122 total_gib()` calls `GlobalMemoryStatusEx` with no `argtypes`, **no
return-value check**, and an `except → NaN`. `byref` marshals correctly so it likely works, but a
failure leaves the struct zeroed and reported as a measurement — the identical failure mode 55 lines
below the one just fixed — and **every comparison against NaN is silently False**, so a headroom
assert built on it can never fire. "Probably fine" is the verdict class this project retired:
*"I am not saying it is wrong — I am saying it is not evidence."* Add `argtypes`/`restype`, raise on
failure, and validate against the known 80 GiB.

### `ts_s0.u16` containing m0 bytes — change it

Reversible for the cost of one constant, and it should be paid. The manifest records `src_tag: m0`
and the size differs 12×, so the truth is available — but only to someone who reads the manifest,
which is the *declared-but-not-load-bearing* shape this rung has been bitten by four times. The
specific harm is sharper than a confusing name: the `RUN:` line and the checkpoint `cfg` will say
**s0** while the run consumed **m0**, and §19 settled that a record must never be falsified. That
section kept `cfg['out'] = w16` precisely *because* it was the record of what was invoked; here the
record would state something that did not happen. Rename the artifact and set the tag. If it ever
turns out to cost more than a constant, the floor is a startup assert that the ids file size matches
the token count expected for the declared tag — the mismatch must not be able to stay silent.

---

## 22. Trainer gated, chain rebuilt — two checks before the main run departs (2026-07-31)

### The gate passes, and the deviation from my instruction is an improvement

**PASS against immutable sha `bce0e89d63ff`, CE mode, 200 steps, one seed.** CLAIM A(old) vs B(new):
every tensor bit-identical, weights sha256 `3b577140caa1…` MATCH. CONTROL A(old) vs C(new, seed+1):
DIFFERS, worst on `emb.weight` at 6.306e+00 — the comparison fires, so the null informs. Throughput
2177.0 → 2177.5 tok/s, +0.0%.

**Deviation accepted, and the reasoning corrects mine.** I specified the sha256 of the saved
checkpoint; he hashed the **weights**, not the file, because the checkpoint also carries the argparse
cfg and this very change adds `--min-host-ram-gib` — a file hash would have come out DIFFERENT for a
reason having nothing to do with behaviour. I chose the file because it was one hash and singleness
was the virtue; but the file bundles the *invocation* with the *product*, and the claim is about the
product. **General rule, worth keeping: a gate must be scoped to the object it makes a claim about.
If a benign change can flip it, it will be argued with instead of believed** — and a gate that gets
discussed has already lost the property we built it for. The question stays answered in bytes.

### Three defects the gate would not have caught — the first is the serious one

**(1) The accelerator guard was inverted.** `log()` used at line 373, defined at 401: `--expect-gpus 2`
would have raised `UnboundLocalError` on a **correct** 2×T4 session and exited clean on a wrong one.
It never surfaced because no arm before the main run passed the flag. His summary is exact: *the guard
ordered to protect the main run worked only when it failed.*

This is my order from §19, and the order was right; what I did not require was the thing that would
have caught it. **Standing rule from here: no guard ships without being exercised in BOTH directions —
must-pass on a known-good input, must-refuse on a known-bad one — and the exercise goes in the log.**
The remedy is demonstrated in the same message: `--min-host-ram-gib` was validated 10 → passes,
999 → refuses. A guard that has never executed is untested code that looks like protection, which is
a new member of the declared-but-not-load-bearing family: it *was* load-bearing, it was simply never
run.

**(2) The chunked bincount overran into the validation band** — `range(0, ntr, 2^24)` with no clamp on
the last slice — and was refused by the val-split assert the trainer already carried. **The control
caught a defect in the repair, not in the original.** That is the third time this rung that the defect
was in the remedy (the memory fix that left two larger walls; the ε-identity gate's own three
defects; now this), and it is the argument for gating fixes as strictly as features. Verified after:
chunked == direct, 473,531,343 against the 473,531,343 declared.

**(3) A moving reference from the opposite side.** He edited the trainer while the gate was comparing
it: run A had started against the commit, B and C would have read a different working-tree file.
Killed, relaunched from a frozen tree. **This is the dual of the 2026-07-29 law and it completes it:
pinning the reference is not enough if the OBJECT moves.** Both endpoints must be stationary for the
duration of the comparison. Cheap closure, and it should go in: **hash both endpoints at gate start
and re-hash at gate exit; if either moved, the verdict is VOID** — two hashes, and it makes the
failure impossible to produce by accident rather than merely unlikely.

### The three additions — two of them better than what was asked

**(i) Host-total table** (`main_run_trainer_ram_projection.txt`): 2 ranks × 7.23 + 2.30 notebook
process = **16.76 GiB**, including torch, host-side CUDA context, P62, the bincount temporary and the
negligible entries. The 20 GiB floor leaves ~3.2 GiB for the recall index and ~9 GiB of slack on a
~29 GiB session.

**The best decision in the message is that the recall index is declared NOT MEASURED rather than
absorbed into a margin** — stages 1-4 all ran `--recall off`, so no local number exists, and folding
an unmeasured quantity into "margin" is how a projection becomes a guess wearing a number. One
consequence follows and costs a line: the main run is `--recall on`, so **session 1 is the first time
this quantity is observed at all.** Print the host peak immediately after the index is built. No
refusal — the true risk is the index exceeding ~12 GiB, which is not credible at rung-1 scale — but
the unknown should stop being an unknown within minutes of the first session rather than being
carried into the 10B design.

**(ii) `--min-host-ram-gib`, and the cgroup catch is one I had not anticipated.** On Kaggle
`/proc/meminfo` reports the physical host, so a check against `MemTotal` reads a number the process
cannot spend and **passes always**. Reading the cgroup limit and taking the min with MemTotal is the
difference between checking the machine and checking the allowance — precisely the "plausible number
that is not the right quantity" class, and one that no test on the dev box would have exposed. It
also raises instead of returning a sentinel, applying the `total_gib()` lesson forward into new code.

**(iii) tok/s ratio, 2% threshold — with the right correction to my instruction:** the absolutes are
CPU-local and are not the 3,860 tok/s that carries the calendar; the **ratio** is the measure, and it
is device-independent.

`total_gib()`, the sibling flagged in §21: argtypes, return check, raise instead of `except → NaN`,
validated against the known 80 GiB (79.95, matching `host_ram_gib()`).

### Chain rebuilt

CODE_SHA `befac334…` → `8b05126b4fda0d7b…`, DATA_SHA `79c58b30…` → `171e481379f54ba3…`, with the
package directory **deleted first so no previous-generation file could survive beside the new one** —
which removes the failure mode rather than detecting it, the same hierarchy as extraction in
`mkdtemp()`. DATA_SHA moved because of the SRC_TAG→TAG collapse, which renamed the shipped files and the
meta. *(Corrected 2026-08-01: this line first also credited "the new trainer". It does not —* `DATA_SHA`
*covers the four data files only, the code lives in a separate block. The discriminator was a number that
stayed still where it should have moved: `CODE_SHA` changed twice while `DATA_SHA` held, which is the
diagnostic corollary read in reverse. No decision moves — check (a) verifies the trainer bytes directly and
never depended on `DATA_SHA`.)* The package now carries `ts_m0.u16` / `p62_m0.u16` / `meta_m0.json`
and the cell passes `--tag m0`, so the `RUN:` line and the checkpoint cfg will record the corpus the
run actually consumes. Floor assert: the ids file must be exactly 2× the declared token count
(2,295,693,322 × 2 = 4,591,386,644 B).

### Two checks before launch, then it goes

**(a) Confirm the trainer bytes inside the shipped package are byte-identical to the trainer the gate
certified.** Not a suspicion — a one-command sha comparison. The gate ran, then the tag collapse
touched the packaging, and "the artifact that departs is the artifact that was gated" is exactly the
invariant the 2026-07-22 stale-package incident violated. It is verified on the thing that ships, or
it is not verified.

**(b) Exercise the new slicing once at seq 2048.** The gate covered stage C only. At **stage E** the
sequence length goes 512 → 2048 with B·L held constant, so the micro-batch reshapes 8 → 2 and the new
uint16-slice-then-cast path meets a geometry it has not seen — roughly 137k steps into a 183k-step
run, across a dozen resumes. A handful of steps at the stage-E geometry, or at minimum an assert on
the slice shape. This is not a second gate; it is the one untested shape on the changed path.

Then: MM pushes the lot (push-before-run, gate logs included), the Kaggle operator's rotation plan for
~14 sessions lands before session 1, and the Capo launches.

---

## 23. Stage E runs at seq 2048 — a sealed parameter that was never implemented (2026-07-31)

### The finding

The Builder found, one step from launch, that **stage E runs at seq 512 like every other stage**. The
comment asserting otherwise is the written justification of the single-epoch invariant — which holds
regardless, and a fortiori. So the extension exists in the documents and not in the code.

### This is not an architect's judgement call; the seal already decided it

`PHASE64_RUNG1_PREREG.md` line 226, body text, sealed at v1:

> "**Consequently, stage E's context extension is declared, not left implicit:** … rung-1 extends to
> **seq = 2048**, with micro-batch reduced to keep B·L — and therefore scan activation memory and
> tokens per optimizer step — constant. Gated scale then: **128 / 512 / 1024**, plus d=8 as the
> standing calibration sanity, plus 2048+ record-only. **Without the extension the recall tier would
> be judged in a regime where the state still reaches, which is not the question the tier exists to
> answer.**"

And line 220 fixes the consequence: gated distances are **bounded by the trained context, longest
gated ≤ half of it**. At a trained context of 512 the gated scale collapses from 128/512/1024 to
**128 alone** — 512 is the trained context, not half of it. The pre-registered 2×2 recall read at 512
and 1024 would then be **null by construction**, which is precisely the extrapolation artifact WS4 was
built to prevent, and the pre-registered recall demotion would fire **on that artifact**. That is the
stage-3 danger in its exact original shape: a sealed rule matching a clean number produced by the
apparatus rather than by the model.

**So restoring seq 2048 is not a change to the plan; it is the plan.** Two further points make it
non-discretionary:

1. **The omission ran in the cheap direction** — 512 saves ~37 h — which is the direction we are most
   obliged to police. A declared parameter that quietly did not happen, in the direction of less cost,
   is the anti-Goodhart case whether or not anyone intended it.
2. **The budget was approved with the cost inside it.** §19's calendar is 108 h **+34%** → ≈145 h ≈ 14
   sessions, and the +34% is exactly the stage-E extension. The arithmetic is self-consistent at
   B·L constant: prereg line 73 measures the seq-2048 penalty at **2.69×**, and 0.8 + 0.2 × 2.69 =
   1.338. Micro-batch 2 at seq 2048 (2 × 2 ranks × 2048 = 8,192 tok/step) keeps tokens per optimizer
   step, STEPS = 183,105 and the effective batch all invariant. **The rotation plan for ~14 sessions
   stands unchanged** — which matters, because at 512 it would have been over-provisioned and someone
   would eventually have "corrected" it.

The VRAM question raised in §22 is already answered by the prereg and does not need re-measuring:
memory is **linear in L, +1.4% at constant B·L** (line 73), and 9,988 MiB of 12 GB was the batch-4
ceiling on the 3060, so batch 2 on a 16 GB T4 is comfortable. §22's check (b) therefore narrows to
what it always was: exercise the geometry once, confirm the slice shape and report tok/s.

### Order of operations — killing the gate now is compliant, not a second violation

The gate currently running certifies a trainer that lacks this behaviour. **Kill it, make the one-line
change, relaunch on a frozen tree.** The frozen-tree law is *"do not move the object while the gate is
comparing it"* — kill → change → relaunch is exactly what it prescribes. Letting the gate finish and
then re-gating costs two runs to reach the same place, and shipping a trainer we already know needs a
change would break the invariant the last two days were spent establishing.

### Two wiring requirements that would otherwise fail the same way this one did

**(a) Persist the sequence length and assert it on resume.** Stage E begins ~137k steps into a 183k-step
run, across roughly a dozen resumes. If seq is set at stage entry rather than derived from the stage at
every step, a resume landing mid-E silently restarts at 512 — and this is the family that has already
bitten us three times with quantities living outside the checkpoint: `α`, `kdc.pos`, the GradScaler.
Put seq in the checkpoint cfg and assert on restore that it matches the stage's declared value.

**(b) Wire the pre-registered early read.** Prereg lines 236-241 already seal a decision tree, so the
question "is +34% worth it" is not reopened — it is answered by measurement at near-zero cost:

> "It becomes measurable … on **the first stage-E checkpoint that exists with the extension applied**:
> run the calibration there, before the remaining arms pay for it."
> | 2048 fires | proceed as declared; gated scale 128 / 512 / 1024 |
> | 2048 flat, 512 fires | tail extension transfers only partially → cut the gated scale to what is
> demonstrated and drop the extension to that level on the remaining arms |

"The remaining arms" are the three §6 branches, which fork from D and run through E. So
`--save-stage-ckpt` must emit the **stage-E** checkpoint as well as stage D, and the MQAR calibration
must run on it when it appears. A pre-registered read with an unwired prerequisite is the same failure
class as the parameter this section exists to restore.

### Check (a): the Builder's reading is right, and it implies one thing he did not draw

His statement of what today's PASS means is exact and should be kept as written: **"the invariant is
now verifiable, and this time it was verified"** — not "the invariant is protected". With the gate
running before the bundle is built, the link is by construction and (a) confirms rather than discovers.
Its value is the *next* time someone touches packaging between gate and upload, which is the 22/07
sequence.

**What follows: a check whose entire value is in future runs must be automated, or it will not exist
in the scenario it was built for.** A manual step in a five-item list works today because we are paying
attention; 22/07 happened precisely when someone was not. Concrete form — `pack_kaggle` reads the
certified sha from the gate log and refuses to build when the trainer bytes do not match, which makes
the gate→package link load-bearing instead of incidental and expresses the existing policy (any trainer
change requires a new ε-identity gate) in code rather than in prose. The packagers are not in the
bundle, so this costs no CODE_SHA change and no new gate.

**Not now.** Today's PASS is genuinely valid by construction, and adding a build-path change minutes
before launch is the exact move that has cost us this week. First non-blocking item once session 1 is
running.

---

## 24. Main run — seed deviation at 41.9%, and the condition underneath it (2026-08-01)

### The deviation, and where it is recorded

The run trained under `--seed 0` to gstep 76,687, diverged in fp16 at ~94,379, was rolled back to the
76,687 checkpoint and resumed under `--seed 1`. The fork is verified on the checkpoints: history
bit-identical to 76,687. Current state gstep 96,012 (52.4%), weights finite, GradScaler 4.0.

**This is recorded here and not in the prereg. The prereg's amendment window closed at v9** — "the
first gate-bearing number now exists, so by the seal's own clause nothing below is amended" — and a
deviation is declared *beside* a sealed document, never by editing it. Same handling as the w16 → w8
OOM deviation in §19. A seal that gets amended when reality disagrees with it is not a seal.

### The deviation does not damage anything the plan rests on

Verified rather than accepted:

- **Distribution unchanged.** The sampler is i.i.d. uniform with replacement, so two independent
  segments drawn under different seeds have the same expected unique coverage as one segment of the
  combined length. The sealed 0.653 expected passes / 47.9% unique coverage already *assume* sampling
  with replacement — that is where 1 − e^(−0.653) comes from — so the figure is untouched.
- **The single-epoch contract holds on the correct reading.** The model performs 183,105 optimizer
  steps; the ~17,700 steps drawn under seed 0 after the fork point were discarded with the weights
  that produced them. Total draws across session history exceed the contract; total draws *the
  surviving model trained on* do not. Stated here because an auditor reading the session logs will
  count the larger number, and the answer belongs on the record before the question is asked.
- **No decision moves.** The main run is a single arm carrying no adoption. The three §6 branches fork
  from stage D at 137,328 — downstream of 76,687 — so all inherit the same history and the paired
  comparisons are unaffected. Against the screening, σ_seed = 0.005 already prices seed-to-seed
  variation; a reseed places this run inside the distribution that constant describes.

### The finding the report does not contain: GradScaler 4.0 is not a healthy reading

"Weights finite, scaler 4.0" is offered as evidence of health. The first half is; the second half is
evidence of the opposite, and it is the only quantitative statement in the report.

`torch.amp.GradScaler` initialises at 65,536 with `backoff_factor` 0.5, `growth_factor` 2.0 and
`growth_interval` 2,000. **4.0 is fourteen backoffs below init.** More sharply: the run is 19,324 steps
past the fork, so a scaler that had been quiet since the fork would have doubled up to nine times and
would read ≈ 2,048. Reading 4.0 means an overflow fired within roughly the last 2,000 steps, and the
scale is oscillating near the floor rather than recovering from a single event.

That reframes the incident. Changing the seed treats the divergence as a **data-order accident**; a
loss scale pinned at the floor is the signature of a **systematic condition** that the reseed moved
rather than removed. Two further facts narrow it: the divergence at ~94,379 and the fork at 76,687 are
both inside **stage C** — plain seq 512, no surgery, no QAT ramp, no MoE upcycle — which is the most
benign context available and leaves the event entirely unexplained; and the trainer changes gated
before launch (uint16 residency, per-batch cast, derived geometry) do not touch fp16 numerics and were
certified bit-identical at exactly this stage and sequence length, so they are not a credible cause.

**Near-term, and this is the actionable part: stage D begins at step 100,707, roughly 4,700 steps
away, and opens with the 3,600-step α-QAT ramp** — the transition that produced a +0.32 shock on the
MVE. The run is approaching a known-stressful boundary with the loss scale already at the floor.

### Ordered, and none of it stops the run

**Diagnosis, from logs already written, no GPU cost:** the GradScaler trace over the last ~20,000
steps — the step of the most recent backoff, the backoff frequency, and whether `found_inf` fires
regularly or in bursts. That single trace separates "one pathological batch, now behind us" from
"persistently on the edge of fp16 range", and those two have different correct responses.

**Pre-registered response to a second divergence, decided now while it is still blind.** Reseeding
after each crash selects for the seeds that happen to survive; at one occurrence the effect is
negligible, at three the run becomes "trained on the seeds that did not crash", which is a selection
effect that would have to be declared and could not be undone. **Rule: a second divergence is not
answered with a third seed.** It triggers the diagnosis above and a numerical remedy — gradient-norm
clipping, or a reduced LR for the remainder, declared as a deviation with the step it takes effect —
because a numerical remedy is a stated change to the recipe, while serial reseeding is an unstated
search.

**The run continues.** Weights are finite and the gradient signal is real; a low loss scale costs
precision, not correctness, and stopping a 145-hour run to think is worse than measuring it while it
moves.

---

## 25. Stage C closed; second divergence, and the loss scale is a quality problem (2026-08-01)

### The first gate-bearing number of the main run

**Stage C exit: P62 code-val BPB 1.0787 `[DECIDING]`, byte invariant HOLDS.**

Both caveats the operator attached are correct and were volunteered, not asked for. The stream-tail val
(4.0172 → 0.9520) **is not cited**: the sealed eval hierarchy (prereg v7) makes the pinned P62 code-val
the decider and the tail val apparatus — curves, divergence, liveness. And 1.0787 **is not compared to
1.242**: that comparison was declared invalid when the corpus was found temporally held out, so this is
a new number, not a measured improvement on an old one.

Against the screening's arm1 (V=2048, P62 code-val 1.1376) it reads −0.0589 ≈ 11.8 σ_seed, but the
screening arms trained a fraction of the tokens on a different slice. **Directional sanity only — the
main run is progressing as a ~10× token budget predicts. It is not a controlled comparison and must
never be quoted as one.**

**One caveat nobody has raised, and it attaches to this number.** Stage C exited with the GradScaler at
2.0. At that scale, with gnorm ≈ 0.50 over ~11M active parameters, RMS per gradient element is ≈1.5e-4,
so the scaled value sits at ≈3e-4 — above fp16's smallest normal (6.1e-5), but the lower tail of the
distribution is not: elements 10-100× below RMS land in subnormal territory and lose precision, and the
smallest go to zero. **1.0787 was therefore produced under mild-to-moderate gradient truncation.** The
number is not invalidated — it is what the run actually achieved — but it was measured in a degraded
numerical regime, and if the numerics are repaired the two regimes are not comparable. Recorded now,
before anyone needs it to be comparable. *(Inference from the arithmetic above, not a measurement.)*

### Second divergence — the pre-registered rule fires

Divergence at gstep 104,452 (last finite 104,402), stage D, under `--seed 1`. **The reseed moved the
condition from 94k to 104k; it did not remove it**, which is the outcome §24 pre-registered as the
reason not to answer a second divergence with a third seed. That rule now binds me as much as anyone:
the response is diagnosis plus a declared numerical remedy.

**The scaler series is the evidence §24 asked for, and it says the scale is descending, not resting:**

| gstep | scale | last overflow |
|---|---|---|
| 96,012 | 4.0 | 1,073 steps ago |
| 100,707 | 2.0 | 940 steps ago |
| 103,992 | 0.5 | 1,107 steps ago |

A growth tracker sampled at ~1,000 three times running implies an overflow interval near 2,000 — right
at the growth threshold — so the scale halves about as often as it doubles and drifts down. Three net
halvings over 7,980 steps is consistent.

**The operator's inference is sound and I adopt its direction while keeping his label on it (hypothesis,
not measurement).** The decisive part is not the implausibility of a 131,000× outlier: it is that
**backing off is the cure for magnitude overflow, and three backoffs into sub-1.0 territory have cured
nothing.** The medicine has been administered and the disease has not responded, so the disease is not
what the medicine treats. Below 1.0 the mechanism plausibly inverts — halving pushes more of the
gradient tail into subnormals and to zero, and a `0·inf` or `0/0` downstream manufactures the NaN that
triggers the next backoff. T4 is Turing: bf16 is not available, so the standard escape is closed.

**This reframes the urgency. A collapsed loss scale is not only a stability risk — it is silently
truncating the gradient signal, and has been since at least gstep 96,012.** Getting past the crash is
not sufficient; the run needs a healthy scale to be worth its remaining GPU-hours.

### A coincidence worth checking before any theory is built on the scaler alone

Stage D begins at 100,707 and the α-QAT ramp is 3,600 steps, so **the ramp completes at 104,307 and the
divergence is at 104,452 — 145 steps later, with α at full strength for the first time.** Close enough
to be a lead, far enough not to be trivially causal. It is the transition that produced a +0.32 shock on
the MVE.

It does **not** explain the first divergence at 94,379, which was in plain stage C with no QAT, no
surgery and seq 512. That inconsistency stays on the record unresolved rather than being smoothed over:
either there are two causes, or one fragility with two triggers.

### Ordered — and the code does not need to move

**The operator framed the scaler-reset test as "one line in the resume path, but it moves CODE_SHA — your
decision". It does not have to.** The GradScaler state lives *in the checkpoint*. Resetting it is a
**data** change, not a code change: an offline script that loads `resume_103992.pt`, sets the scale to
the 65,536 default and clears the growth tracker, and writes a **new file** — original never overwritten.
No CODE_SHA move, no ε-identity gate, no rebundling, no repackaging. The certified chain stands intact.

Conditions on the surgery, because a hand-edited artifact is exactly where plausible corruption lives:
every other tensor and field **bit-identical**, asserted; the changed fields printed before and after;
and a **planted control** — deliberately perturb one unrelated tensor and confirm the identity check
fires — before the edited checkpoint is trusted.

**Why 103,992 and not the stage_C boundary.** Restarting from `stage_C_100707.pt` would give a fresh
scaler for free (model-only checkpoint means no scaler state to restore) and re-enter stage D through the
designed path, at a cost of ~3,285 steps ≈ **1.9 GPU-hours**. But it restarts the sampler, so the batches
differ — and then a clean run cannot be distinguished from having dodged a pathological batch. Resuming
from 103,992 **replays the same batches**, which is what makes the test a controlled experiment. Keep the
stage-C route as the fallback if the surgery cannot be verified.

**Three-way readout, declared before the result exists:**

| observation | conclusion |
|---|---|
| immediate cascade back to the floor | magnitude overflow after all; the scale was tracking a real signal |
| NaN again at ≈104,452 with the scale still healthy | the source is external to the scaler — and check α |
| runs clean past 104,452, scale settles ≥1,024 | the descent was self-feeding; low scale was cause, not symptom |

**Two pre-registrations that stop this from being read after the fact.** First: **if the divergence
recurs within ~200 steps of 104,307 despite a healthy scale, the trigger is α-QAT reaching full strength
and the remedy belongs in the QAT schedule, not in the loss scale.** Second: **if the restart runs clean,
we adopt it and declare it as a numerical deviation with the step it takes effect — we do not re-label it
"a diagnostic that happened to work".** A scaler-state discontinuity is a change to the recipe whichever
box the result lands in.

**Zero-cost diagnostic that should be read while the restart prepares:** the gnorm trace over the ~200
steps before *each* divergence, 94,379 and 104,452. If gnorm is flat at ≈0.50 and `found_inf` fires with
no excursion, that is NaN and not overflow, measured rather than inferred — and if the two divergences
share a signature across different seeds and different stages, that is the common cause the scaler series
cannot name. It is already in the logs.

**When a trainer change does become necessary** — a scale floor, gradient-norm clipping, or a QAT
schedule change, whichever the diagnosis selects — **it carries one more item into the same gate rather
than a second one later: the checkpoint writer must refuse to write, or must mark, a checkpoint taken
inside a non-finite sequence.** This time the 20-minute timer happened not to fire during the NaN run and
no artifact was poisoned. That was luck, and a poisoned checkpoint is the plausible artifact par
excellence: finite-looking, resumable, wrong.

### The guard

`every account holding a checkpoint ended in MVE-DIVERGED / NOT pushing, and NOT burning another session
/ NEEDS A HUMAN` is the relay guard firing on a real event, in the correct direction, refusing to
propagate a poisoned state and refusing to spend. With the previous code it would have propagated the
checkpoint and burned three sessions. **This is the first guard in the project to pay off on an
unplanned production incident rather than on a planted control**, and it paid back the quota it cost.

---

## 26. Divergence forensics, checkpoint surgery, and the live discriminator (2026-08-01)

### The diagnostic I ordered was not executable, and saying so was the right move

`gnorm` is printed only at `i == i0` and every `budget//5` — three times per session — so the nearest
sample to the second divergence is **3,744 steps away**. The zero-cost gnorm trace ordered in §25 did
not exist as formulated. The operator reported that rather than substituting something that resembles
it, which is the distinction this project runs on: a diagnostic that cannot be executed is not a
diagnostic that came back empty.

### The substitute instrument is better than what it replaced

`hist` records the loss at **every** step, and the poisoned checkpoint from the *first* divergence
(94,419) contains the onset:

| window | mean | sd | max |
|---|---|---|---|
| baseline, 1,900 steps before | 1.8602 | 0.1550 | 2.5315 |
| last 25 finite | 1.8630 | — | 2.3245 — **below the baseline max** |
| step immediately before | 1.6133 | — | **below the mean** |
| next step | **NaN** | | |

**Zero excursion, zero precursor — no rise, no trend, not even a local record.** A magnitude-overflow
cascade driven by exploding activations would announce itself in the loss first. It did not. The NaN
signature is now measured rather than inferred, which was the point of the exercise.

The operator's two limits are correctly stated and are kept: it measures the **forward**, not the
backward, so it does not replace *found_inf-without-gnorm-excursion*; and it covers only the first
divergence, because at the second the last checkpoint (103,992) precedes onset (~104,403) and no
per-step record exists. **The shared signature across two seeds and two stages therefore remains
unmeasured**, and is not claimed.

One reading to add: the last finite loss, 1.6133, is **1.59 sd below** the baseline mean — not merely
"below the mean". In a window of 25 that is unremarkable on its own (one sample beyond ±1.6 sd is
expected), so it is a lead and not a finding. It is cheap to chase later because the sampler is seeded
and deterministic: the batch drawn at that step is reproducible offline on CPU.

### The next order: open the poisoned checkpoint

**94,419 is not only evidence that the hazard is real — it is the forensic artifact, and nobody has
looked inside it.** Zero GPU, one file already in hand. Four questions, in order:

1. **Which tensors are non-finite**, and are they **NaN or inf**?
2. Is the pattern **localized** (one module, one layer) or **global**?
3. Are the **optimizer moments** non-finite as well, or only the weights?
4. What is `scaler.scale` in that checkpoint?

These discriminate mechanically between forward-activation NaN, backward-gradient NaN, and — the case
worth naming explicitly — **something updating outside the GradScaler's protection**. That last one
matters because the scaler's contract is that non-finite gradients cause the step to be *skipped*, so
weights should never become non-finite at all. **That weights in this checkpoint are corrupted at all
means either the corruption entered through a path the scaler does not guard, or the run continued
executing after onset** — it wrote this checkpoint ~40 steps after the divergence began. Either answer
is worth more than the hypothesis it replaces.

### Surgery — the planted control applied at its sharpest

Perturbation of `model.blocks.4.mlp.down.weight[0]`: 0.013754970394074917 → 0.013754971325397491,
delta 9.3e-10. **Verified: for a float32 in [2^-7, 2^-6) the ULP is 2^-30 = 9.3132e-10, so this is
exactly one ULP — the smallest perturbation the format can represent.** That is the minimal-significant-
corruption rule from the val_split case at its limit: not a catastrophe planted to make the detector
look good, but the smallest thing that could ever be wrong. The comparator flagged exactly one
difference and fired; only then was the real edit made. The comparison runs on **raw storage bytes, not
values**, which also immunises it against NaN ≠ NaN and dtype coercion.

Edit: `scaler.scale` 0.5 → 65536.0, `scaler._growth_tracker` 1107 → 0. Exactly those two fields differ,
everything else bit-identical; gstep 103,992 / stage_idx 1 / step_in_stage 3,285 / seq 512 / 0
non-finite weights. Original never overwritten.

### Experiment placement and relay state — both correct

Running on **giggio253**, deliberately not wildpino, so wildpino's output remains the cloud copy of the
original 103,992 and of `stage_C_ce_anchor_on_m0.pt` alongside the local copies. That is
*original-never-overwritten* extended from the local file to the cloud replica, and it is the right
instinct: the experiment cannot destroy its own baseline.

Relay stays paused — **"an experiment with a declared read step is not a resumption"**. An auto-relay
would run straight past the read point and turn a discriminator into a training run.

### Revised read — one row the §25 table was missing, and a declared horizon

The three-way table stands. It had a gap: it assumed any recurrence would land near 104,452.

| observation | conclusion |
|---|---|
| immediate cascade back to the floor | magnitude overflow after all |
| NaN again at ≈104,452, scale still healthy | source external to the scaler → **and 104,452 is 145 steps past α-QAT completion at 104,307, inside the ±200 window, so the sealed reading names α-QAT as the trigger and the remedy belongs in the QAT schedule** |
| **NaN at a step far from 104,452, scale healthy** | **the same batches are being replayed, so a batch-triggered NaN would recur at the same step. A different step means the trigger is not the data — it is a recurring numerical fragility, and the remedy is a scale floor plus gradient clipping regardless of what fires it** |
| clean past 104,452, scale settles ≥1,024 | the descent was self-feeding; low scale was cause, not symptom |

**Declared read point: gstep 106,500** — 2,048 steps past the decision point, one full `growth_interval`
beyond it. The criterion is not "did it survive" but **"did the scale grow"**: surviving at 65,536
without a single doubling opportunity observed would not distinguish a cured run from a lucky one.
Cost from 103,992: 2,508 steps ≈ **1.5 GPU-hours**.

### Correction for the record

The numerical-regime caveat on 1.0787 — that stage C exited at scaler 2.0 and the number was therefore
produced under gradient truncation — is in **§25**, not §24. It is written, and it was written before
anyone needed the two regimes to be comparable, which was the point.
