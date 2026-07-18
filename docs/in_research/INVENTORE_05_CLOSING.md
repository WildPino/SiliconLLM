# The Inventor — 05: Investigation closed (2026-07-17)

**What this is.** The closing ledger of the Inventor pass opened in `INVENTORE_00` (big picture, 8 sparks)
and executed same-day at **$0, CPU-only, on existing checkpoints, on branch `inventor/s2-s3-probes`** —
zero shared files touched, MVE run untouched. Every spark now carries a number or a documented
disposition. Detail: `INVENTORE_01` (probe results), `02` (S1 brief), `03` (S5 proposal), `04` (S6 note);
raw outputs in `*_out.txt`; scripts in `benchmarks/in_research/`.

## Final disposition — all 8 sparks

| # | Spark | Disposition | Key number / artifact |
|---|---|---|---|
| S1 | structured projections | **RUN + GATE PASSED (2026-07-18 overnight, sealed brief, owner-delegated launch)** | from-scratch low-rank x_proj: r=52 Δ+0.0035 (G1 PASS, sub-noise); **r=26 Δ−0.0074, BETTER than dense at 17.6% of x_proj bytes** (post-hoc predicted +0.053 there — pessimistic by ~0.06). ctl reproduced P61's 0.8757 exactly. Single-seed; 3-seed escalation before adopting the r26 improvement as real; v1 untouched, candidate = v1.1/v2 rung-boundary A/B. Full table `INVENTORE_02` §6 |
| S2 | entropy-coded weights | **DEAD (from-scratch pools); conditional rerun on QAT'd ladder ckpts** | trits ≈ max-entropy (H0 1.5835/1.585, zeros 31%); the whole 2.5× is the queued 5-trits/byte pack |
| S3 | delta-coded experts | **DEAD (from-scratch pools); conditional rerun on upcycled ckpts (born 4×-replicated)** | NN-match ≈ shuffled control, delta loses to direct everywhere; side-find: fp32 rows correlate (|cos| ~0.47 vs 0.26 random) → fed S1 |
| S4 | entropy-patch tokenizer | **v2 RESEARCH, feasibility measured POSITIVE (input side)** | BPE CV 0.75 & 17% of tokens <2 bits vs patch CV 0.33 at 31% fewer units; 3× shorter at B=16. Output side (variable-byte emission) = the open half |
| S5 | α-QAT (ε-identity law) | **IN THE POCKET, trigger-armed** | mechanism + exact code in `INVENTORE_03`; applies only if MVE gate (iv) shows a C→D shock; end-state bit-identical to current QAT |
| S6 | unified lookup operator | **v2 DESIGN NOTE banked** | recall ∪ router = one `lookup(q;K,V,k)`; router-as-ANN becomes real at E≥~1024; preconditions listed in `INVENTORE_04` |
| S7 | SEE drafter | **RECORDED, parked with block-verify** | no action until a shared-streamed-dominated design point exists (10B story) |
| S8 | fused projection GEMVs | **DEAD (premise)** | already fused where fusible; rest is a sequential chain |

Score: 3 clean kills (S2, S3, S8), 1 strong lead with a ready apparatus (S1), 1 positive feasibility card
for v2 (S4), 2 banked designs (S5, S6), 1 recorded synergy (S7). Plus one law proposed for the books:
**every curriculum switch is an ε-identity** (generalizes the magnitude-match law + zero-init gate; `03`).

## New knowledge fed back to the frozen program (no action required)

1. **The projection map is now complete on two axes:** P61 (precision) + this pass (rank) agree that
   in/out_proj are information-dense organs, while x_proj is structurally compressible — a fact about
   *selective-SSM control pathways* worth one line in any future scaling dossier.
2. **Weight-trit incompressibility** closes the "entropy-code the pool" question the compressor DNA kept
   whispering: pack density (engine-v2 trit-pack) is the whole prize. No decode stage, no complexity.
3. **The BPE information-unevenness numbers** (17% of code tokens < 2 bits) are independent ammunition
   for any future adaptive-compute argument (K2 depth-reuse, patching, halting) — measured on our
   pinned val, our tokenizer.

## Standing recommendations to the owner (ranked, all $0 until a launch)

1. **S1 A/B** — approve + seal the brief (`INVENTORE_02`), then launch the 3 arms on the free 3060
   (~1 evening). Highest knowledge-per-hour available right now; does not touch the MVE (per-box rule).
2. **MVE-checkpoint reruns** (when stage-D/E checkpoints exist): `s2` on the QAT'd weights, `s3` on the
   upcycled pool (replica-divergence telemetry) — minutes each, free, and they close both conditionals.
3. **S5**: hold until the MVE gate (iv) verdict; apply only on a measured shock, as a declared deviation.
4. **S4 output-side sketch + S6 preconditions**: v2 backlog, revisit at the v1 export gate.

## Method note (for the record)

Discipline held: read-only side-lab, separate branch, every negative published with its control
(shuffled baselines, random-init spectra), every positive stated with its pessimistic-bound caveat
(post-hoc vs from-scratch), no gate language attached to exploratory numbers, apparatus + brief prepared
for the one promotable item with push-before-run intact (nothing launched). Two implementation bugs found
by their own outputs (self-match sentinel; uint8 overflow) — both fixed and rerun before any reading.

*The Inventor pass is closed. Reopening criterion: new checkpoints (MVE rungs) or a new design gate.*
