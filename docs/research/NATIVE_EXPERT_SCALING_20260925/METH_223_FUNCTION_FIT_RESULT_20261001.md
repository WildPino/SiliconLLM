# METH-223: common fit limitation and larger-bank generalization gap

The [protocol](METH_223_FUNCTION_FIT_PROTOCOL_20261001.md) was frozen at
`45c82e4`. The first invocation stops on a diagnostic sequence-reduction
ordering mismatch, with its failure preserved. Commit `dc82bea` restores
the original full-sequence reduction before the second invocation.
No hyperparameter, route, coefficient, source or decision gate changes.

The corrected run reproduces **every original fit parent/leaf count**,
**every validation sequence's common/E16/E160 SSE and energy**, and
the replayed common's BF16 weight/FP32 bias exactly. It reads only
saved METH-222 states and parameters; no new source-model inference,
selected replacement model or training is performed.

| Unchanged function arm | Fit normalized SSE | Validation normalized SSE |
| --- | ---: | ---: |
| Common, BF16 weights | 0.435067452 | 0.466232462 |
| Common, replayed unrounded FP32 weights | 0.435065943 | 0.466231691 |
| E16 | 0.395401591 | **0.439157627** |
| E160 | **0.297576050** | **0.469383645** |

Both precommitted primary diagnoses hold:

- The fixed regularized affine common already has **43.51%** normalized
  fit error. Removing coefficient rounding improves validation by only
  **7.71e-7** normalized SSE. This supports choosing a nonlinear common
  as the next mechanism; it does not prove that every affine solver or
  every alternative regularization must fail.
- E160 has **24.74% lower fit SSE** than E16, but **6.88% higher
  validation SSE**. The enlarged bank learns the fitting targets better
  and generalizes worse on these separate raw windows. This is a
  measured fit/validation relationship, not a universal n limit or proof
  of a sole causal explanation.

The rare-cell-only diagnosis fails. Cells with <65 fit states cover
**77/16,384** validation states (0.470%) and contribute only **1.74%**
of the positive E160–E16 SSE gap. Cells with65–255 states contribute
38.83%, and >=256 states59.42%. More data or shrinkage may help, but
deleting a few rare cells does not explain or repair the observed gap.
Counts remain correlated token states, not independent samples.

The [raw result](meth223_function_fit_diagnostic_result.json) binds the
same exact capture/checkpoint and preserves all 640 sequence rows,
support-stratum energies/errors and diagnostic flags. Runtime4.531 s,
ending RSS1.839 GB, GPU peak2.041 GB on local RTX3060/six threads.
Session45362 exited0; first session37385 exited1. No T4 or project
inference job remains active.
Raw result SHA256: `c0eaa8ed5064e0588573275c6b56492b4dcc75fdcc745f19cfbc1e4f96dc9d77`.

## Next mechanism, not yet implemented or validated

Preserve actual pretrained-output targets. Replace the common's affine
function by a compact learned nonlinear SwiGLU common (initial candidate
hidden width768), and anchor child functions to their parent rather than
fit independent full leaf maps. A confidence/shrinkage rule must be
frozen before fitting; fold parent+child correction into one stored leaf
matrix so E16/E160 keep one equal-width active function. No static-bias
calibration or repetition of the failed independent-affine leaf bank.

First price this whole geometry and freeze native BF16-weight/FP32
common feasibility before training. Width768 common weight payload would
be99,090,432 bytes/token across24 layers, versus the affine38,535,168.
Under METH-222's unchanged hypothetical head/attention/control and
conditional ledger this totals329,811,968 selected weight bytes/token.
These are arithmetic proposals, not a physical trained full core, actual
DRAM trace or measured C rate. The native component may use fixed actual
donor row subsets solely as a shape/precision fixture; such a carve is
not a revived quality candidate. Only credible cost and nonlinear
function retention would license a regularized route/function pilot,
then full causal composition and independently frozen quality. Useful
large-n CPU LUT/DRAM, multiple families/approximately10B and >=50 accepted
tok/s on the same artifact remain required.
