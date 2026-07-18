# The Inventor — 02: S1 pre-registration brief (SEALED) — structured x_proj, from-scratch A/B

**Status: SEALED 2026-07-17 — owner approved in-session ("Voglio che procedi") and explicitly delegated
the launch to the Builder for this run (recorded deviation from the owner-launches rule, owner's own
instruction). Push-before-run: this brief + apparatus pushed to origin BEFORE the arms launch; gates
below are frozen as written, before any training number exists.** Nothing here touches the frozen v1
spec or the MVE: this is a **sandbox model-side probe** in the P61 lineage, feeding v1.1/v2 recipe
decisions only.

## 1. Question (one variable)

Can `x_proj` — the selective-SSM control projection (Dn → dt_rank + 2N: the input to B, C and the
already-rank-16 dt path) — be **re-parameterized as low-rank at training time** with no BPB cost?

**Prior (measured, `INVENTORE_01`):** post-hoc SVD truncation of the trained x_proj costs +0.0001 BPB at
r=104, +0.0008 at r=78, +0.0078 at r=52 (its trained PR-rank), cliff below r≈26. Post-hoc is the
pessimistic bound (P61 lesson); from-scratch should hold r≈52 or lower. Spectra: trained PR-rank 53/208
vs random-init 148. in/out_proj are NOT candidates (rank-hungry + precision-hungry, two independent axes).

## 2. Arms (matched; the R1 lesson: within-script fresh control, never the historical anchor)

Foundation recipe = the sp58 class (8.3M Arch-A D256 N96 L6 SWA@5, gated-dReLU ternary MLP, fp32
organs), phase57/59 training recipe (4k steps, seq 512, batch 16, lr 3e-3, bf16, CE only), same data
order, same seed. ONE variable = the x_proj parameterization:

| arm | x_proj | new params | rationale |
|---|---|---|---|
| 0 control | dense (208×512), fp32 | — | fresh matched baseline |
| A | U·V low-rank, **r=52**, fp32 | 52·(208+512) = 37.4K vs 106.5K (35%) | the PR-rank point; post-hoc +0.0078 → from-scratch should recover |
| B | U·V low-rank, **r=26**, fp32 | 18.7K (17.6%) | the aggressive point; measures the from-scratch recovery margin |

Deferred second stage (only if A passes): arm C = Hadamard-sandwich / butterfly form (zero-weight
transform + O(D) diagonals) — higher implementation risk, kept out of the first A/B (one variable).

## 3. Gates (sealed at approval, before any numbers; anti-Goodhart: never loosened)

- **G1 (pass):** BPB(A) ≤ BPB(0) + **0.005** (= σ_seed from R1: indistinguishable from seed noise).
- **G2 (gray zone, pre-declared):** +0.005 < ΔBPB(A) ≤ +0.010 → escalate to 3 seeds before any verdict
  (R1 calibration rule: single-seed deltas < 0.005 unresolvable).
- **B is reported as the curve, not gated** (its post-hoc prior says it should cost something; the
  interesting number is how much from-scratch recovers vs +0.053 post-hoc).
- Secondary recorded (no gate): sparsity bands, top-1 agreement vs arm 0, per-layer x_proj spectra of
  the trained A/B (does the r=52 factor itself concentrate further?).

## 4. Cost & scheduling

~2-3 h/arm on the 3060 (P61 precedent) × 3 arms ≈ one evening, $0. The 3060 is currently free (MVE runs
on Kaggle) and the never-2-trainers rule is per-box → schedulable now at the owner's discretion. Owner
launches; Builder delivers apparatus + CPU smoke + ready commands and STOPs.

## 5. Payoff if PASS (why this is worth an evening)

- **Bytes:** x_proj at r=52 = 35% of its bytes; ~14% of total projection bytes at 8.3M shapes. Scaling
  direction favors it: x_proj is ((dt_rank+2N)·Dn) — its share grows with state size N.
- **Compute:** same fraction off the x_proj GEMV share of the per-position floor (proj-GEMV = the
  dominant engine component, 52.7%).
- **Design knowledge:** completes the projection map — P61 said "no fewer bits"; this says whether the
  control pathway tolerates "fewer dimensions". A pass legitimizes structured forms (arm C, and the
  recall/router query projections) as a v2 axis; a fail seals ALL projections as dense-fp32 with two
  independent measurements — either way the map closes.
- **Not v1:** the frozen ladder recipe is untouched; any adoption is a v1.1/v2 decision at a rung
  boundary with its own A/B.

## 6. RESULTS + ADJUDICATION (run 2026-07-17 23:08 → 07-18 06:38, 3060, launch delegated by owner)

| arm | params | x_proj bytes | val BPB (200K) | Δ vs ctl | gate |
|---|---|---|---|---|---|
| 0 ctl dense | 8.312M | 100% | **0.8757** | — | (reproduces P61's fresh base 0.8757 exactly — free apparatus sanity check) |
| A r=52 | 7.966M | 35.2% | 0.8792 | **+0.0035** | **G1 PASS** (≤ +0.005; below σ_seed → clean, no gray zone, no escalation owed) |
| B r=26 | 7.873M | 17.6% | 0.8683 | **−0.0074** | curve (not gated): BETTER than dense by ~1.5σ |

**Verdict: S1 PASSES its sealed gate.** From-scratch low-rank x_proj at r=52 is quality-neutral within
seed noise; the aggressive r=26 point — where post-hoc truncation cost +0.053 — not only recovers fully
from scratch but lands 0.0074 BELOW the dense control (5.7× fewer x_proj bytes, −5.3% total params).
The P61 lesson quantified in reverse: post-hoc was pessimistic by ~0.06 BPB at r=26.

**Honest caveats (declared with the numbers):**
- Single seed. The non-monotonicity (r52 +0.0035 vs r26 −0.0074, a ~2σ spread) is unexplained: plausibly
  seed noise, possibly a regularization effect at 4k steps. The sealed rules require nothing further
  (G1 passed clean), but **claiming the r26 improvement as real requires the 3-seed escalation** —
  recommended before any recipe adoption.
- Sandbox scale (8.3M, TinyStories). Scale transfer is the usual open question; the architecture
  precedent (dt_rank = D/16 by design) argues the control pathway's low-rank nature is structural,
  not scale-accidental.
- Adoption path unchanged: v1 frozen spec untouched; candidate = v1.1/v2 recipe A/B at a rung boundary
  (and the arm-C butterfly form is now legitimized as a second-stage probe).

## 7. Apparatus

`benchmarks/in_research/s1c_structured_xproj.py` (this branch): trains any arm with `--xproj-rank {0|52|26}`
(0 = dense control), same script/data/seed for all arms, val BPB at 200K protocol + top-1 vs a dumped
control-logit slice. CPU smoke: `--smoke`. Ready commands (owner):

```
.venv/Scripts/python.exe benchmarks/in_research/s1c_structured_xproj.py --xproj-rank 0  --steps 4000 --seq 512 --batch 16 --bf16 --save results/phase57/s1c_ctl.pt
.venv/Scripts/python.exe benchmarks/in_research/s1c_structured_xproj.py --xproj-rank 52 --steps 4000 --seq 512 --batch 16 --bf16 --save results/phase57/s1c_r52.pt
.venv/Scripts/python.exe benchmarks/in_research/s1c_structured_xproj.py --xproj-rank 26 --steps 4000 --seq 512 --batch 16 --bf16 --save results/phase57/s1c_r26.pt
```
