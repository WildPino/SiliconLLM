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
