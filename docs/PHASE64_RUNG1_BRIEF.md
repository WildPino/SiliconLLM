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
