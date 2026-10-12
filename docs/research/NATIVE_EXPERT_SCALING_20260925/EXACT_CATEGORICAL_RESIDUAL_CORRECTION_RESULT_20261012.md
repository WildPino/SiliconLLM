# Exact categorical residual correction: gradient PASS, quality still FAIL

12 October2026. Original experiment, separate refinement and both independent
audits CLOSED exit0. Goal ACTIVE/INCOMPLETE; no owned run live.

## Implemented change and verified mathematics

[Tool](../../../benchmarks/native_expert_scaling/exact_categorical_residual_correction.py),
[frozen protocol](EXACT_CATEGORICAL_RESIDUAL_CORRECTION_PROTOCOL_20261012.md).
Actual raw source h24, first/middle/last LABEL states on FIT24 (72 total), equal
case/label weight. Fixed trained source head with gamma/head multiplier folded;
actual full65537 F64 probabilities, not cached BF16 teacher logits. Initial
source-function V/W rank256 plus raw zero center. Optimize exact categorical KL,
not initial Fisher. No source model forward/backward framework or GPU.

For y=mu+V W^T(h-mu),r=sqrt(y^T y/2048+1e-5),p=softmax(Ay/r), gradient
g=A^T(p-q) passes the complete RMS Jacobian
`gy=g/r-y*(y^T g)/(2048*r^3)`. Both center paths included. Full V/W/mu gradients
were independently reconstructed with explicit2048x2048 Jacobian matrices and72
per-state outer products, not just the producer vector formula.

Six separate harmonic directions plus the actual joint normalized-gradient
direction, two original scales,28 full finite-point outputs; all4 fixed trust
candidate parameter updates, original selection and DEV audited.

## Original nonacceptance preserved; numerical convergence resolved separately

Original tolerance2e-6, central steps1e-5 and2e-6. All separate-group controls
PASS; joint analytic derivative -578.9316751364058. At1e-5 finite derivative
-578.9299075716236/relative error3.0531492e-6 FAIL. At2e-6 error1.2212209e-7 PASS.
Original13/14 flags and `accepted=false` remain immutable. No correction was
accepted in that first record despite the candidate FIT improvements below.

Error falls~25-fold when step falls5-fold, consistent with central-difference
truncation proportional to step^2. [Separate refinement protocol](EXACT_CATEGORICAL_RESIDUAL_REFINEMENT_PROTOCOL_20261012.md),
[tool](../../../benchmarks/native_expert_scaling/exact_categorical_residual_refinement.py)
tests ALL7 original directions at2e-7, same tolerance. All pass; joint error
1.2215023e-9. Original two-scale Richardson estimate is stored as a diagnostic.
New rule requires all7 original2e-6 plus all7 new2e-7 flags. It does not overwrite
the original1e-5 failure. No new gradient, steps, charts, samples or epoch added.

The refined selection qualifies the original FIT-minimum candidate alpha.001,
sealed with its parameter hashes BEFORE calculating its new DEV metrics.
Independent audits verify probabilities, exact bit updates, new directions and
selection. Source control flow establishes that order; artifact hashes alone do
not timestamp semantic data reads. The refinement seal's parent acceptance field
contains parent's unchanged VALUE(false), not a flag indicating mutation.

## Measured probabilities

| Point | FIT meanKL | FIT worstKL | FIT wrong /72 |
|---|---:|---:|---:|
| Initial global factors | 1.1139558 | 9.4237332 | 18 |
| alpha.0001 | 1.0568170 | 8.3282082 | 17 |
| alpha.001 (selected) | .6451858 | 5.8989855 | 13 |
| alpha.01 | 24.5826238 | 48.9406343 | 69 |
| alpha.1 | 35.2969068 | 54.0149366 | 70 |

Selected FIT mean improves42.0816%. Large normalized-group steps destroy quality;
correct gradient alone does not justify an unbounded radius or long training.

| DEV72 point | MeanKL | WorstKL | Wrong source argmax /72 |
|---|---:|---:|---:|
| Initial global factors | 2.7278198 | 11.4366233 | 26 |
| Refined qualified alpha.001 | 2.5543312 | 11.6484663 | 26 |

Mean improves~6.36%;worst worsens slightly;number of wrong decisions unchanged.
All frozen meanKL<=.01/worst<=.1/disagreement<=.01 gates FAIL. This is a verified
descent/representation primitive, not useful chatbot quality or generative evidence.

## Geometry, remaining defects and next action

Updated max W^T V-I defect.3280169823: generic corrected factors are NOT the old
idempotent projector or old quadratic optimum. Rank<=256 remains algebraic;
affine norm G/cross/constant remains valid. Source-injected W^T h is not cheap
causal state acquisition. Skip/core transport must be explicitly rederived and
tested;do not insert these factors into the old identity-skip assumptions.
Source internal SSM/attention reading errors~40%-50% remain uncorrected.
The last actual native chatbot still0/16;no new native/generative run here.

**Next:** broaden the exact categorical residual test to ALL existing FIT4422
labels/DEV4386 labels, with equal-case weights and FIT-only step selection,
before any repeated epoch. Current calibration saw only72/4422 FIT labels.
Implement compact reproducible score factors/source-width independent probability
replay rather than requiring huge dense score archives;validate arithmetic first.
Recompute one broad gradient/controlled ladder, retain every candidate and audit.
This resolves coverage/generalization and cost before dose;it still does not
qualify causal dynamics, ternary/native export, useful n or same-artifact50.
[Prospective specification](BROAD_EXACT_RESIDUAL_CATEGORICAL_NEXT_20261012.md).

## Reproducibility and resource closure

Original source/protocol2eaa714; [binding](exact_categorical_residual_correction_binding_20261012.json)
SHAbbb6cb1f2bfd6fa69e006a82c57ef9b1b0e1e4b0e21453ce9cf0e6ff319ad301,
prepare3.375s;[invocation](exact_categorical_residual_correction_launch_20261012.json)
8fea948. Producer92013 CLOSED0,57.391s/OS1,661,186,048B;50 files1,526,578,310B.
First FIT72 source-head multiply.25s, not an end-to-end/native timing claim.
[Result](exact_categorical_residual_correction_result_20261012.json)
SHAd35583c9cf0eed09f6479b75511b8a390cf6bda1592c6666ba93825dd5795a40;
terminalf7a2379708ef3ceac26df33d1a64f6da6970e2f98a42c17165042a9a011b997d.

[Original audit](../../../benchmarks/native_expert_scaling/exact_categorical_residual_correction_stored_audit.py),
[protocol](EXACT_CATEGORICAL_RESIDUAL_CORRECTION_STORED_AUDIT_PROTOCOL_20261012.md)
eb5e4cf/[invocation](exact_categorical_residual_correction_stored_audit_launch_20261012.json)
d9cba4a. Session61480 CLOSED0,61.109s/OS1,743,695,872B.
[Adjudication](exact_categorical_residual_correction_stored_adjudication_20261012.json):
all34 full-V points/references/72 full Jacobians/gradient fields/14 difference
controls/4 bit-identical updates/selection/DEV PASS;original nonacceptance intact.
Max relative4.2883e-12/scalar1.7273e-11. Full input/output before/after hashes PASS.

Refinement source/protocol3c1fe43; [binding](exact_categorical_residual_refinement_binding_20261012.json)
SHAb6319aa6b24fb5b2b2eb84fd8f30e74e97f45b2bdfecf2a8d4265549ab3be4fd,
prepare4.828s;[invocation](exact_categorical_residual_refinement_launch_20261012.json)
427f1ec. Producer97501 CLOSED0,30.703s/OS1,493,495,808B;16 files587,362,309B.
[Result](exact_categorical_residual_refinement_result_20261012.json)
SHAc508d0fdf69a9f87c530580e799cd337f73db5f86f77e4fd07135a100e4610dc;
terminala649043795fa908be406bf6707450d3a64032909f739623545f88d41805d6882.

[Refinement audit](../../../benchmarks/native_expert_scaling/exact_categorical_residual_refinement_stored_audit.py),
[protocol](EXACT_CATEGORICAL_RESIDUAL_REFINEMENT_STORED_AUDIT_PROTOCOL_20261012.md)
02e71b2/[invocation](exact_categorical_residual_refinement_stored_audit_launch_20261012.json)
d9cc37b. Session83245 CLOSED0,32.141s/OS1,540,100,096B.
[Adjudication](exact_categorical_residual_refinement_stored_adjudication_20261012.json):
all15 new full-V packets/7 refined differences/unchanged original artifacts/
FIT-only selection/new DEV PASS;max relative4.6676e-13/scalar1.3628e-11.

All declared resource limits pass;no mechanical first fault in these families.
Original scientific gate failure retained. Exact PIDs/hashes/closure in
[session receipt](exact_categorical_residual_session_closure_20261012.json).
Ordered outputs on D;space sufficient,no E copy. Original engine/foreign files/
publishers unchanged;no T4/GPU/source model/native/RESERVED/longalign call.
