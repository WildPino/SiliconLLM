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
