# Source-function coordinates: arithmetic PASS, all quality proxies FAIL

12 October2026. COMPLETE, no owned run live. Full objective ACTIVE/INCOMPLETE.
Seven FIT-only raw residual read/write bases were implemented and evaluated;
their local reading errors improve, but no qualified pretrained chatbot conversion.

## New variable, algebra and applicable tools

[Protocol](SOURCE_FUNCTION_COORDINATES_PROTOCOL_20261012.md),
[source v2](../../../benchmarks/native_expert_scaling/source_function_coordinates_v2.py).
Reuse the independently checked normalized RAW source state covariance C at
boundaries0/4/8/12/16/20/24. Internal M balances actual source input-normalized
SSM gate/x/B/C/dt and attention q/k/v projections by equal FIT energy. Final M
uses actual source gamma/head and the previously audited FIT mixture Fisher.
Cached-generation qbar/full-prefill raw states remain distinct source paths.
No DEV calibration, fitted gains, diagonal gamma projection or source forward.

For K=C^(1/2) M C^(1/2), leading256 eigenvectors U give distinct reconstruction
V=C^(1/2) U and read W=C^(+1/2) U. Encoding z=W^T h and reconstruction Vz are
the declared FIT quadratic optimum on the retained covariance support, not a
nonlinear/task/KL optimum. Source gamma folds into readers; norm uses V^T V,
with epsilon8e-5 in256 form. Changing bases needs W_next^T V_previous skips.
All support dimensions2048 here; no covariance eigenvalue was discarded.

Max W^T V-I defect6.4393e-14; maximum FIT observed/spectral-tail delta1.3878e-16.
F32-rounded dual defects2.16e-8..2.85e-7 are algebraic round-trip observations,
not an F32 inference qualification. The chosen balanced metric is not the
nonlinear recurrence/attention Jacobian, and does not measure task information.

## Full source-history reading errors

All48 cases/22,547 positions at each of7 boundaries; FIT24, DEV24, equal-case
summaries. Oracle uses the actual source denominator; own norm uses decoded h.
Angular control is the previous FIT angular256 basis under the SAME new M.

| Boundary | FIT metric energy retained | DEV oracle error | DEV own-norm error | Angular oracle control |
|---|---:|---:|---:|---:|
| 0 | 94.9254% | 27.2605% | 41.3670% | 33.2771% |
| 4 | 90.7907% | 34.9896% | 43.3402% | 49.3675% |
| 8 | 88.1496% | 40.0269% | 48.5817% | 57.5510% |
| 12 | 87.9542% | 40.7550% | 47.5946% | 54.9118% |
| 16 | 86.0143% | 45.7711% | 52.0293% | 57.0259% |
| 20 | 87.3877% | 41.5770% | 49.0757% | 53.7569% |
| 24 | 97.4798% | 18.8120% | 23.2259% | 29.7734% |

All mean<=10%/worst<=20% proxies FAIL, even with oracle norm. The internal group
most distorted at the selected first/middle/last144 HISTORY states is SSM x:
DEV oracle pooled32.88/45.14/55.20/55.56/57.46/51.20% at0/4/8/12/16/20.
These are pre-convolution/pre-nonlinearity input projections, not state update
errors. Groups512 B/C or48 dt being easier does not preserve3072 x/gate functions.

## Actual output probabilities on the SAME DEV72 states

Actual source BF16 coefficients decoded to F64; full65537-vocabulary source
readout as previously audited reference, not cached BF16 logit equality.

| Representation | Mean KL | Worst KL | Wrong source argmax /72 |
|---|---:|---:|---:|
| Previous angular256, oracle norm | 2.7064073 | 13.3451836 | 31 |
| New source-function256, oracle norm | 2.4290814 | 9.4617459 | 26 |
| New source-function256, own norm | 2.7278198 | 11.4366233 | 26 |

All frozen probability gates FAIL. Positive norm scalar cannot change the
bias-free head argmax; it changes confidence/KL. Exact decoding is not causal
state acquisition. No generated conversation or end-to-end/native speed measured.

### Why the quadratic optimum can still lose a confident answer

[Frozen diagnostic](FUNCTION_COORDINATE_READOUT_DIAGNOSTIC_PROTOCOL_20261012.md),
[tool](../../../benchmarks/native_expert_scaling/inspect_function_coordinate_readout.py),
[all144 rows and full spectra](function_coordinate_readout_diagnostic_result_20261012.json).
For source logits l, q=softmax(l), prediction l+delta, the exact identity is

`KL(q||softmax(l+delta)) = log E_q exp(delta - E_q delta)`.

The local Fisher half-variance is only its second-order approximation. Identity
verified on all144 rows within1.8414e-11 of independently audited KL.
Concrete oracle-norm counterexample: `broad_dev_smol_summarize_034`, position226:
source top2 margin6.2772, local half-Fisher variance.03514, chosen mixture
half-variance.39300, actual KL9.46175. Source winner drops to rank2668;
centered logit perturbation reaches31.34. A previously unlikely competitor can
become important under a finite perturbation, while source-local curvature is tiny.
This witness disproves using a small local quadratic as a finite-loss certificate.

Among26 wrong oracle rows, median FIT qbar weight of source winner1.666e-6,
versus.009227 among46 correct; descriptive association, not a causal claim.
Wrong rows' mean actual KL5.9057 versus mixture half-variance.4981; even actual
per-state half-Fisher mean3.0487 fails to track the finite loss consistently.
Hence merely replacing qbar by initial per-state Fisher is insufficient.

Exact alternative algebra: for p_t=softmax(l+t delta),
`KL = integral_0^1 (1-t) delta^T F(p_t) delta dt`.
It includes competitors emerging along the perturbation path. An average of
these metrics would still be a surrogate for fitting a single rank-limited map;
sample-dependent input and curvature remain coupled. No such fit was run here.

## Active cost, capacity and selected next action

Stored spectra need FIT quadratic99% ranks711/1030/1221/1231/1329/1210/477
at0/4/8/12/16/20/24. These are precise results for THIS global linear oracle
reading criterion, not a universal residual capacity lower bound or DEV guarantee.
Blind active-width expansion is costly and would not establish>=50 useful tokens/s.

Current256 head is67,109,888B F32 and16,777,472 dense MAC/token. Full norm metric
and each skip are262,144B/65,536 dense MAC. Largest coarse skip singular value
10.7207; singular values describe these coordinate maps, not physical-norm
amplification or full dynamical stability. Source state/generator widths3072/256
are still large. W^T h needs fullsource h and524,288 MAC; this injected encoder
is not available as a cheap causal deployed operation. No DRAM/LUT timing claim.

**Next: FIT-only conditional affine source residual charts**, first bounded
geometric feasibility on representative boundaries0/12/24, K4/16, active256.
The new variable is a discrete chart index carrying directions/centers which one
global basis omits. Fit raw-state centers/covariances with source-RMS weighting,
not an angular center silently treated as a raw residual center. Retain exact
affine norm cross terms, reader biases and chart-to-chart skip offsets.
Evaluate actual held-out reader errors and full-V probabilities; include an
oracle source chart choice only as an explicit ceiling, never a deployment claim.
Detailed prospective [next record](CONDITIONAL_RAW_RESIDUAL_CHARTS_NEXT_20261012.md).

If oracle geometry fails, do not invest in a router or long dose on that geometry.
Complete audits regardless. If it passes, next must establish chart selection
from available causal compact state and useful conditional function execution;
fullsource nearest-center selection is not affordable routing. Full exact KL or
finite-step margin control is required for later adaptation. No T4/long job chosen.

## Custody, reproducibility and execution closure

Preparation v1 freeze7874154 exit1/.5394s before binding: physical extent compared
against metadata including shape/dtype. [Fault](source_function_coordinates_preparation_fault_20261012.json)
retained; separate v2 repairs only comparison, source/protocol freeze1861704.
Prepare4.610s/exit0. [Binding](source_function_coordinates_binding_v2_20261012.json)
375 inputs/SHA15c5b6df8ff58b6221ca6f02055a7029e7a1eda56f5bc7b8799cde0f5349e161.
[Exact invocation](source_function_coordinates_launch_v2_20261012.json) commit23d8a4e.

Producer session10659 CLOSED exit0, PID27020 creation1791804040.3368397;
149.047s/result,149.094s terminal, OS660,426,752B,20 outputs1,693,826,180B.
[Result](source_function_coordinates_result_v2_20261012.json) SHA
b2f46e3655d2bfd9dfaafdd75ad39b93f1d12bd60f34a08f4baddf5b04d9786a;
terminalc5d1f0c1240652e67889659dba72a1af947af79c7d938a66c1baaa3617e3504d.
All1200s/6GiB/3GiB limits PASS, input before/after hashes exact.

[Independent audit](../../../benchmarks/native_expert_scaling/source_function_coordinates_stored_audit.py),
[protocol](SOURCE_FUNCTION_COORDINATES_STORED_AUDIT_PROTOCOL_20261012.md),
freezee49b60d/[invocation](source_function_coordinates_stored_audit_launch_20261012.json)
294d84c. Session5070 CLOSED exit0,80.578s/OS779,882,496B.
[Adjudication](source_function_coordinates_stored_adjudication_20261012.json)
SHA66bdad4d06584cdbd03832b117c267bffe81d4f3dbfe8afeef0aff7bc72120e4:
all7 actual source metrics/complete spectra/dual maps/336 histories; all48 full
projection packets/192 metrics; all72 source states/fullhead/full-V outputs/144
probabilities/6 transports. Source-width partial sums and independent quadratic
expansion, no producer fit/import. Max numeric relative1.1358e-14,
scalar1.3776e-11, scoreabsolute1.7408e-13. Full before/after hashes PASS.
Reuse prior verified C/qbar, no refit/model/GPU/native replay.

Diagnostic freeze8b3c1d4/[invocation](function_coordinate_readout_diagnostic_launch_20261012.json)
4f8a561, CLOSED exit0/3.609s/OS157,585,408B; all fixed limits pass.
Original engine, source/old results and foreign work/processes preserved.
No new E copy necessary, no T4/model training, no RESERVED/longalign access.
