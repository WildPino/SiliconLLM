# E68 — a rank-64 shared linear residual recovers local error, but not complementarity across layers

**Result:** `PARTIAL_LOCAL_SIGNAL`, 2026-09-16. The frozen
[brief](../briefs/BRIEF_E68_SHARED_RESIDUAL.md) fixes the donor, D0c labels,
splits, ranks, ridge rule and primary rank before the run. E68 is a CPU-only,
local FFN-output reconstruction diagnostic. It is not BPB, generation, engine,
cache, bandwidth or tok/s evidence.

## Question and frozen object

E67 held fixed a sparse FFN approximation: for each token it chose three of
256 D0c groups by float64 activation mass, retaining 105 of 8,960 neurons.
E68 asks a different local question: can a rank-`r` linear map of the MLP
input `x` explain the FFN output omitted by that same sparse choice?

For calibration only, it fits `S_r(x) = (x A_r) V_r^T` to
`y - y_sparse`, using the frozen float64 reduced-rank ridge construction and
`r={32,64,128}`. A shared-only comparator fits the same form to `y` and omits
the selected groups when scored. Primary adjudication is rank 64. The fit
slice is the disjoint `calib` 8×512 slice (4,096 positions/layer); the score
slice is E67's exact held-out sequences 0–3, positions 384–511 (512
positions/layer), at layers 1, 7, 14, 21 and 27.

The sparse selection still uses dense `z = SiLU(W_gate x) * W_up x` before it
can choose groups. Therefore this is a capacity diagnostic for a proposed
shared-plus-block-sparse operator, **not an executable router or a sparse
inference implementation**.

## Controls and provenance

All synthetic ridge/factor, objective-monotonicity, rank-zero, mass-tie and
reconstruction controls passed. `x` was captured at the MLP pre-hook and `z`
at the down-projection pre-hook. The calibration and held-out slice hashes,
model revision, E19/D0c label hashes, E67 runner/result hashes and finite
factor checks all passed; the result records `heldout_used_for_fit: false`.

At rank zero, every layer reproduces E67's mass-only held-out SSE to relative
difference at most **2.92e-10** (the frozen tolerance is `1e-5`). Each sparse
path selected three distinct equal-size groups. Dense group contributions
reconstructed the down projection inside the `1e-5` relative-RMS control.

The full result JSON SHA-256 is
`02e5ae986ad3f59241d78602378bb7e17575c3bbf38f78db13244442da91be47`; the
one-layer smoke JSON SHA-256 is
`4dad1a9c741ad55ded1787fcfedf029cab59b24db851f21c89521cd6a9b03c24`.
The source hashes embedded in the full result include runner
`0fd10804742ce856f626314a2a251a8982f4a9ff6d63da2d48f79ca815df867e`, the
frozen E68 brief, `density/common.py`, E19's label loader, and E67's runner
and full result.

## Held-out rank-64 result

The metric is summed local squared error; aggregate ratios are formed from
summed SSE, not by averaging layer percentages.

| layer | mass-only SSE (`r=0`) | combined residual rank-64 SSE | combined / mass |
|---:|---:|---:|---:|
| 1 | 26,876.563427 | 47,602.173707 | 1.7711406384 |
| 7 | 154,032.840704 | 134,330.494370 | 0.8720899631 |
| 14 | 113,665.947309 | 86,640.934218 | 0.7622417819 |
| 21 | 446,756.691130 | 374,562.896015 | 0.8384046696 |
| 27 | 7,247,181.889910 | 3,085,428.007897 | 0.4257417648 |
| **aggregate** | **7,988,513.932479** | **3,728,564.506207** | **0.46674069016** |

The combined rank-64 arm cuts aggregate SSE by **53.3259%** against mass-only.
The separately fitted rank-64 shared-only arm has SSE **26,958,119.734236**;
combined/shared-only is **0.1383095165** (shared-only/mass is
`3.3746100917`). The selected sparse residual is therefore essential in this
local comparison.

Layer 27 supplies **90.7200257%** of the mass-only aggregate SSE
(`7,247,181.889910 / 7,988,513.932479`). Its large improvement consequently
dominates the aggregate. Layer 1 instead becomes worse: combined/mass is
`1.7711406384`. Layers 7, 14 and 21 improve, but by less than the frozen 30%
per-layer complementarity threshold.

## Frozen verdict and its limit

The primary rank-64 verdict is **`PARTIAL_LOCAL_SIGNAL`**. It clears the
aggregate ≥20% mass-only reduction band, but is not
`COMPLEMENTARY_SHARED_SIGNAL`: only layer 27 improves mass-only by at least
30%, and layer 1 is worse than mass-only by far more than 5%. It is not
`SHARED_ONLY_SIGNAL`, because the rank-64 shared-only comparator is worse than
mass-only in aggregate SSE.

The layer-1 calibration/held-out contrast is an overfit or distribution
mismatch diagnostic, not a causal proof. On calibration, mass-only has
`sumSSE/sumEnergy = 0.0021752327` but median per-token ratio `0.74264146`; the
aggregate is therefore dominated by high-energy observations (an inference
from the discrepancy, not a claimed identified source). The rank-64 combined
calibration ratio is `0.0008168424`, while its held-out layer-1 ratio is
`0.9834137`. This warns against reading the calibration fit as a stable
layer-1 benefit; it does not identify why the held-out distribution differs.

No end-to-end loss, BPB, token ranking, generation, exact byte count,
cache-residency, engine parity, tok/s, pretrained-10B quality or scale claim
follows. A positive local residual fit does not establish that the dense-`z`
oracle can be replaced by a cheap selector. Conversely, this linear,
post-hoc residual fit does not refute a jointly trained nonlinear shared path,
a different residual selector, partition, rank or target width.

The smoke run is only a one-layer/one-held-out-sequence apparatus check and
returns `NOT_APPLICABLE_SMOKE`; it is not a second verdict.

Canonical artifacts:

- [full result JSON](../../../../benchmarks/donor_adaptation/engine/results/e68_shared_residual.json)
- [one-layer smoke JSON](../../../../benchmarks/donor_adaptation/engine/results/e68_shared_residual_smoke.json)
- [frozen brief](../briefs/BRIEF_E68_SHARED_RESIDUAL.md)
