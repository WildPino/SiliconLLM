# Inventor Pass — Report to the Architect

**Status: DRAFT — finalizes when the 3-seed S1 escalation lands (chain running on the 3060,
started 2026-07-18, ~15 h; every other number below is final).** Prepared by the Inventor side-lab
(owner-mandated creative review, 2026-07-17/18). All work on branch `inventor/s2-s3-probes`,
`benchmarks/in_research/` (read-only toward the repo) + `docs/in_research/`; the frozen v1 spec, the
MVE apparatus, and the Kaggle run were never touched. This report consolidates `INVENTORE_00..05`;
the owner may attach further research to it before sending.

## 0. Mandate and method

Mandate: look at the whole project from above and hunt for architectural improvements by substitution,
fusion, splitting, inversion — "the CPU is deterministic; apply mathematics; dream big."
Method: the project's own discipline applied to ideation — every spark either dies by a controlled $0
measurement or graduates toward a pre-registered probe; desk-vs-measured always labeled; negatives
published with their controls; one gate-bearing A/B run under a sealed brief with push-before-run.

## 1. Executive verdict table

| # | Spark | Verdict | The load-bearing number |
|---|---|---|---|
| **S1** | **structured x_proj** | **GATE PASSED; 3-seed escalation in flight** | from-scratch r=52: Δ+0.0035 (G1 PASS); **r=26: Δ−0.0074, beats dense at 17.6% of x_proj bytes** — where post-hoc predicted +0.053 |
| S2 | entropy-coded ternary weights | DEAD (from-scratch pools) | trits at 99.9% of max entropy (H0 1.5835/1.585); the queued 5-trits/byte pack is the whole 2.5× |
| S3 | delta-coded expert pool | DEAD (from-scratch pools) | NN-match ≈ shuffled control; delta loses to direct everywhere; rerun owed on upcycled ladder ckpts |
| S4 | entropy-patch tokenizer (K4) | v2 FEASIBILITY POSITIVE (input side) | BPE code tokens: CV 0.75, 16.9% carry <2 bits; constant-entropy patches: CV 0.33 at 31% fewer units; 3× shorter at B=16 |
| S5 | α-QAT (ε-identity law) | BANKED, trigger-armed on MVE gate iv | mechanism + code ready; end-state bit-identical to current QAT |
| S6 | unified lookup (recall ∪ router) | BANKED + precondition measured | full-scan router crosses the ANN cost at **E ≈ 2048**; v1 (E≤256, ~25µs) untouched |
| S7 | SEE-experts as drafter | RECORDED (parked with block-verify) | no action until a shared-streamed-dominated design point |
| S8 | fuse projection GEMVs | DEAD (premise) | already fused where fusible; rest is a sequential chain |

## 2. The main result — S1: the selective-control pathway is structurally compressible

**Claim chain, each link measured:**
1. *Spectra:* trained x_proj concentrates to PR-rank ~53 of 208 (random control: 148). Reproduced
   independently on a fresh control run (PR 52.9) — **the concentration is a stable training attractor,
   not a checkpoint accident.** in/out_proj concentrate far less (PR/min 0.47–0.55).
2. *Post-hoc function:* SVD-truncating x_proj to half rank costs +0.0001 BPB (real val eval); in/out
   cliff (+0.033/+0.045 at half rank) — they are rank-hungry on top of P61's precision-hungry.
3. *From-scratch A/B (sealed brief `INVENTORE_02`, gates frozen before numbers, seed 0):*

| arm | params | x_proj bytes | val BPB | Δ vs ctl |
|---|---|---|---|---|
| ctl dense | 8.312M | 100% | 0.8757 | — |
| r=52 | 7.966M | 35.2% | 0.8792 | +0.0035 → **G1 PASS** |
| r=26 | 7.873M | 17.6% | **0.8683** | **−0.0074** |

   The control reproduces P61's fresh-base 0.8757 to four decimals (apparatus sanity for free).
4. *Secondary reads (`s1d`):* foundation properties intact on all arms (dReLU sparsity 79/92
   reproduced; ternary zero-frac 32%) — the one-variable claim holds. Cross-arm top-1 agreement ~83%
   (≈ seed-change distance: genuinely different models, equal quality). **The trained low-rank products
   concentrate further still — product PR ≈ 7–8** — with the architecture's own dt_rank=16 precedent,
   r=16 is a plausible next point (not probed; noted).

**Pending (sealed protocol §6b):** 3-seed escalation, C1 = the "r26 better" claim adopted only if
paired Δ<0 in all 3 seeds AND mean ≤ −0.005; C2 = quality-neutrality at 3 seeds. Table lands here:

*(3-SEED TABLE — pending, chain running)*

**Why it matters beyond the sandbox:** x_proj scales as (dt_rank+2N)·Dn — its share grows with state
size; the projections are the dominant engine compute component (proj-GEMV ~52%) and the largest
resident fp32 class at the S-ladder shapes. A structurally-thin control pathway is also a *statement
about selective SSMs* worth one line in any scaling dossier. Adoption path: v1 frozen spec untouched;
candidate = recipe A/B at a rung boundary (v1.1/v2), butterfly form (arm C) legitimized as second stage.

## 3. The negatives (bought cheap, each closes a door with a control)

- **Ternary weights are incompressible** (from-scratch): near-uniform trit marginals, no pair/Markov/
  per-row structure, on both MoE pool and dense MLP. Consequence: the engine-v2 denser trit-pack
  (5 trits/byte) is the *entire* byte prize; no entropy-decode stage should ever be built for weights.
  One conditional remains: re-measure on the ladder's KD→QAT checkpoints (different training pressure).
- **No cross-expert redundancy** in a from-scratch pool (NN ≈ shuffled control; mode-centroid ≈ control).
  The i.i.d.-routing "license" has nothing to license on these pools. BUT the ladder's upcycled models
  are *born* 4×-replicated — replica-divergence telemetry on MVE/S0 checkpoints is free and owed.
- **No projection-GEMV fusion left** (in_proj and SWA qkv already merged; the rest is a dependency chain).

## 4. Banked designs (no v1 impact, ready when their trigger fires)

- **The ε-identity law** (`INVENTORE_03`): "every curriculum switch is a functional ε-identity at switch
  time" — generalizes the magnitude-matched-upcycle law and the recall zero-init gate; stage D (QAT
  hot-swap) is the one violator. α-scheduled QAT (~15 lines, end-state bit-identical) is armed on the
  MVE gate-iv verdict: shock → the fix exists; clean → parked, nothing changes.
- **Unified lookup** (`INVENTORE_04`): recall slot and MoE router are one operator `lookup(q;K,V,k)`;
  measured crossover E≈2048 where router-as-ANN becomes a speed win; one C code-path + one scaling law
  at engine-v2; fast-weights extension recorded as a bounded research door.
- **Entropy-patch tokenizer** (S4, v2): input-side feasibility measured (see table); open half = the
  variable-byte emission side (BLT local-decoder problem) + D3's sealed vocab for v1.

## 5. Pending-on-MVE ledger (nothing blocked on the Inventor side)

| item | trigger | cost |
|---|---|---|
| S2 rerun (weight entropy) | MVE stage-D (QAT) checkpoint exists | minutes, CPU |
| S3 rerun + replica-divergence telemetry | MVE stage-E (upcycled) checkpoint exists | minutes, CPU |
| S5 α-QAT apply-or-park | MVE gate (iv) verdict | ~15 lines if triggered |
| S1 rung-boundary adoption A/B | 3-seed verdict + owner/Architect decision | one rung-scale A/B |

## 6. Recommendations (ranked)

1. **Read the 3-seed S1 verdict** (lands with this report's final version) and, if C1/C2 hold, put the
   structured-x_proj A/B on the rung-1 boundary agenda alongside the vocab A/B (D3) — same discipline,
   one extra arm, materially fewer resident bytes.
2. **Keep the trit-pack at the top of engine-v2** — S2's negative upgrades it from "queued optimization"
   to "the only byte lever on the pool, measured".
3. **Adopt the ε-identity law as written project law** (it has already been paid for twice); α-QAT is
   its free application if gate iv fires.
4. r=16 x_proj probe and butterfly arm C: cheap follow-ups, only after the 3-seed verdict.

---
*Raw data: `docs/in_research/*_out.txt`, run logs `s1c_run_*.txt`, checkpoints `results/phase57/s1c_*.pt`
(local — promote to release assets if kept). Scripts: `benchmarks/in_research/`. History: 8 commits on
the branch, sealed briefs pushed before their runs (GitHub received-time witness).*
