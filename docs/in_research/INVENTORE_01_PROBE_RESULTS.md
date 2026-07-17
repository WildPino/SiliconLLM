# The Inventor — 01: First probe results (2026-07-17, $0, CPU-only)

**Scope & discipline.** Exploratory side-lab measurements (branch `inventor/s2-s3-probes`, scripts in
`benchmarks/in_research/`, raw outputs in `docs/in_research/*_out.txt`). Read-only toward the repo: existing
checkpoints only, no training, no GPU, nothing touched in the MVE run. These are **desk-inputs for future
pre-registered probes, not claims** — no gate discipline applies here, and any promoted idea re-measures
under prereg rules. Checkpoints probed: `moe_gran.pt` (probe-4 promoted MoE, from-scratch),
`sp58_base.pt` (the E1 anchor). Ternarization everywhere = engine-exact (`e4_export.py` semantics).

## TL;DR

| Spark | Verdict | The number |
|---|---|---|
| S2 entropy-coded weights | **NEGATIVE** (from-scratch pool) | H0 = 1.5835 vs log2(3) = 1.585 — the trits are ~maximum-entropy; entropy coding adds ~0 over the queued 5-trits/byte pack (which alone is the real 2.5×) |
| S3 delta-coded experts | **NEGATIVE** (from-scratch pool) | NN-match ≈ shuffled control; delta-coding loses to direct on every matrix (net 0.0%) |
| S8 fused projection GEMVs | **NEGATIVE** (premise) | already fused where fusible; the rest is a sequential chain, not same-input GEMVs |
| S1 structured projections | **SPLIT — and it found the organ** | global post-hoc truncation cliffs (+0.087 at ρ=0.5), BUT **x_proj at half rank = +0.0001 BPB, free**; spectra confirm x_proj is the low-rank organ (PR 53/208 vs 148 random) |

Three clean falsifications and one live, sharpened lead. The falsifications are cheap and final for the
from-scratch pool; the S1 lead graduates toward a pre-registered probe.

---

## S2 — entropy of the ternary weights (`s2_weight_entropy.py`)

Hypothesis: post-QAT ternary weights are sparse/structured → ~1 bit/weight reachable → another ~1.6×
over the 5-trits/byte pack on the streamed pool.

**Measured (moe_gran, 18.87M pool trits, engine-exact):**
- Marginals p(−1/0/+1) = 0.346/0.313/0.341 — near-uniform; weight-zeros only ~31% (the 79-92% sparsity of
  the architecture lives in the *activations*, not the weights — now measured, not assumed).
- H0 = 1.5835 b/w vs log2(3) = 1.585 (99.9% of max). Per-row H0 = 1.567. Pair entropy /2 = 1.5832.
  Order-1 Markov H1 = 1.5830. **No axis of structure: not marginal, not per-row, not adjacent-pair, not
  Markov.** Flat across layers and matrix classes; the dense sp58 MLP reads the same (1.5842).
- Storage consequence: current pack 4.0 b/w → 9.44 MB pool; 5-trits/byte 1.6 → 3.77 MB (**2.50×, all of
  it from the pack, zero from coding**); the best entropy bound 1.567 → 3.70 MB (+2% — noise).

**Verdict: from-scratch ternary training produces incompressible trits.** The queued denser trit-pack is
the whole prize; an entropy-decode stage would add complexity for nothing. **Conditional remnant:** QAT'd
ladder checkpoints (KD-then-QAT pressure, different from from-scratch CE) could in principle sparsify
weights — re-run this script on the MVE stage-D checkpoint when it exists (cost: minutes, CPU).

## S3 — cross-expert redundancy (`s3_expert_residuals.py`)

Hypothesis: experts share structure → store as shared centroid (resident, amortizing) + sparse ternary
residuals (streamed) → move bytes into the weight class every measured law favors.

**Measured (moe_gran, all 6 layers × {gate, up, Wd}):**
- Aligned mode-centroid agreement 42.6-44.3% vs shuffled control 42.1-42.3% — no aligned sharing (expected:
  permutation symmetry; measured anyway, falsifiable).
- Pool-wide nearest-row match (any expert, any position): median 49-52% agreement vs control 44-51%;
  a few points of real but tiny excess on gate/up (matches mostly intra-expert, 60-67%); Wd exactly at
  control. **Delta-coding (match-id + residual) loses to direct coding on every matrix** — residual
  disagreement ~49%, residual entropy ~1.78 b → 461-470 b/row vs 405 direct. Net pool saving **0.0%**.
- fp32-domain check (after the self-match bug fix): median NN |cos| 0.41-0.51 vs ~0.26 expected for random
  256-dim (max over 4096) → **the fp32 rows DO correlate geometrically** — there is subspace structure in
  the trained weights, but ternarization does not surface it as near-duplicate rows.

**Verdict: no delta-codable structure in a from-scratch pool.** Two live threads out of the wreckage:
(a) the fp32 correlation pointed the search at *spectral* structure → became the S1 probe below;
(b) **the ladder's upcycled models are born with 4× replica seeding** (8 slices × 4 copies + small noise)
— cross-expert redundancy there is a birth property whose *decay under training* is itself an interesting
measurement (how fast do replicas diverge? what survives at stage-F?). Re-run on MVE/S0 checkpoints.

## S8 — projection GEMV fusion (closed by reading, no run)

`in_proj` is exported merged (2·Dn×D), SWA `qkv` merged (3D×D); `x_proj`/`dt_proj`/`out_proj` form a
sequential dependency chain with different inputs. Nothing to fuse. Premise falsified; closed.

## S1 — spectra + functional truncation of the fp32 projections (`s1_spectral_probe.py`)

The P61 verdict ("projections are precision-hungry, stay fp32") tested the *precision* axis. This probes
the orthogonal *structure* axis at zero training: SVD spectra vs random control, then post-hoc rank
truncation with a REAL val-BPB eval (100K tokens, within-script matched; ref BPB 0.8412 on this slice —
consistent with E2's G3 figure on the same protocol, an accidental cross-check that the harness is sane).

**Part 1 — spectra (trained vs random-init control, mean over layers):**

| matrix | shape | PR-rank | PR/min | E@r/4 | E@r/2 | random ctl PR |
|---|---|---|---|---|---|---|
| in_proj | (1024, 256) | 121.1 | 0.47 | 59.1% | 83.9% | 204.6 |
| x_proj | (208, 512) | **53.2** | **0.26** | 82.3% | 96.3% | 147.6 |
| out_proj | (256, 512) | 140.3 | 0.55 | 57.6% | 83.8% | 170.4 |

All three concentrate vs control — training carves real spectral structure — but **x_proj is qualitatively
the low-rank organ** (a quarter of its nominal rank carries 82% of the energy).

**Part 2 — functional truncation (the honest test):**

Global (all three classes together): ρ=0.75 → +0.0097 BPB; ρ=0.5 → **+0.0872**; ρ=0.25 → +0.85. A cliff —
post-hoc global low-rank is a far worse trade than ternary was in P61 (+0.02 at ~20× fewer bits). The
factored-bytes column even shows ρ=0.75 costs MORE bytes than dense (r(m+n) > mn) — global low-rank is
dead as a byte lever at these shapes.

Per-class at ρ=0.5 (50K-token slice, matched ref):

| class | dBPB | top-1 vs ref |
|---|---|---|
| in_proj | +0.0330 | 92.0% |
| **x_proj** | **+0.0001** | **99.5%** |
| out_proj | +0.0452 | 89.9% |

**The split is the finding:** in/out_proj are genuinely rank-hungry (their PR ~0.5·min was load-bearing),
while **x_proj — the selective-SSM control pathway (input to B, C and the already-rank-16 dt path) —
tolerates rank-halving post-hoc at zero cost.** The architecture itself set the precedent: dt is rank-16
by design (Mamba's dt_rank = D/16); the measurement says the *rest* of x_proj shares that nature.

### S1b — how deep does x_proj go? (`s1b_xproj_sweep.py`)

x_proj-only rank sweep (50K tokens, matched ref BPB 0.8131; full rank 208, trained PR-rank ~53):

| r | x_proj bytes | all-proj bytes | dBPB | top-1 vs ref |
|---|---|---|---|---|
| 104 | 70.3% | 93.7% | +0.0001 | 99.5% |
| **78** | **52.7%** | **89.9%** | **+0.0008** | 98.6% |
| 52 | 35.2% | 86.2% | +0.0078 | 95.5% |
| 26 | 17.6% | 82.4% | +0.0534 | 88.5% |
| 16 | 10.8% | 81.0% | +0.1419 | 81.7% |

**The free post-hoc rank is ~78-104 of 208** (dBPB ≤ +0.0008, sub-noise vs the R1 seed-σ 0.005); the
PR-rank point (52) costs +0.008 post-hoc — exactly the territory a from-scratch structured arm should
recover (P61 lesson in reverse: post-hoc is the pessimistic bound). The cliff starts below r≈26.

**Honest sizing:** at the 8.3M shapes this saves ~10-14% of projection bytes — real but modest. The value
is (a) the *existence proof* that the selective-control pathway tolerates structure (unlike in/out_proj,
which two independent axes — P61 precision, this probe's rank — both call information-dense); (b) the
scaling direction: x_proj is ((dt_rank+2N) × Dn), so its share grows with state size N, and a from-scratch
structured arm can plausibly hold r near the PR-rank (~53) rather than the post-hoc-free 78.

## What survives, sharpened

1. **The trit-pack (engine-v2 queue) is confirmed as the only byte lever on the pool** — 2.50×, no coding
   stage on top. S2/S3 die for from-scratch pools; both get one conditional re-run on ladder checkpoints.
2. **S1 narrows from "sandwich everything" to "structure the control pathway":** a from-scratch structured
   x_proj (low-rank or butterfly) has a strong measured prior (post-hoc free at r=104; from-scratch can
   only do better). in/out_proj stay fp32-dense per P61+this probe — two independent axes now both say
   those organs are genuinely information-dense.
3. **New measurement idea banked from S3(b):** replica-divergence tracking across the ladder's upcycle —
   free telemetry on checkpoints that will exist anyway.

**Proposed next step (owner's call, $0):** draft the pre-registered brief for a from-scratch
structured-x_proj A/B at sandbox scale (P61 discipline: matched arms, sealed BPB gate, one variable),
queued for whenever a GPU window is free — NOT during the MVE.
