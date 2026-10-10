# Next: first-response linear controllability before a new core campaign

10 October2026. Prospective algebra/specification, UNIMPLEMENTED/UNEXECUTED.
[Native full audit](CATEGORICAL_NATIVE_ASSESSMENT_RESULT_20261010.md),
[onset diagnosis](CATEGORICAL_RESPONSE_ONSET_RESULT_20261010.md):14/14 donor-correct
chat tasks wrong at first native ID; FIT onset loss gets only1.17535% mass.
No unchanged32-update dose, larger n, T4 or new native history launched.

## Cheap remaining uncertainty

Do the existing first-response native features admit a small shared map to the
ALREADY obtained good per-label codes? If yes, a temporal objective/unfinished
optimizer remains a concrete possibility. If no, or the map is too sensitive
to qualified native feature uncertainty, another causal feature construction or
an exact native normalized-state observer is justified before more head fitting.
This does not establish a general information/rank ceiling or chatbot quality.

Use ONLY24 FIT label_index0/positions from paired_head_image_result labels,
upper_x72x255 and qualified stored audit. Reuse original same A=aK/gain exactly,
not a new per-label oracle or scale. These are feasible reference codes, not
known globally optimal points. Their source-relative per-label upper values
are adopted unchanged, not recomputed as a new head-image experiment.

Reuse categorical shared FIT whitened_features4422x256,whitener256x256 and
best_theta255x256, full prior geometry audit. T0 is24x256 with each case's FIRST
label row, matching exact source/native positions and original records. Desired
Y0 is24x255 of corresponding previous upper codes. Shared parameter domain
remains Frobenius16; distinct from per-label ball16. No new whitening/geometry,
source or held-out labels choose this diagnostic.

## Exact linear algebra, new24-row SVD only

One thin SVD T0=U diag(s) VT, with full-row-rank criterion s_min>1e-12 s_max.
No replay of prior weighted4422-row SVD or any optimizer. If full row rank,
let Q=U diag(1/s) VT (24x256), so (T0^T)^+=Q. Then

    Theta_min = Y0^T Q,
    Delta_min = (Y0^T-Theta_base T0^T) Q,
    Theta_preserve = Theta_base + Delta_min.

Theta_min is the unique minimum-Frobenius-norm exact code interpolant.
Delta_min is the unique minimum-norm exact correction to the current map.
The latter retains Theta_base on the orthogonal complement of the first-feature
row span. With P=VT^T VT,

    Theta_preserve = Theta_min + Theta_base(I-P),
    ||Theta_preserve||_F^2 = ||Theta_min||_F^2 + ||Theta_base(I-P)||_F^2.

Save full SVD factors,all target/feature matrices,minimum/correction/preserved
maps,all24 code residuals/radius margins. Check SVD reconstruction/orthogonality,
normal equations,projection identity,orthogonal norm decomposition and complete
scalar arithmetic witnesses in independent stored audit, without another SVD.

Within radius16 implies a feasible reference-code interpolation on approximate
FIT features, not generalization. Approximate min norm>16 only excludes exact
matching of these CHOSEN codes on the approximate features. It does not exclude
other codes with acceptable KL or a changed target state/decoder.

## Propagate native coordinate uncertainty before an impossibility claim

For old first-feature errors E_i and fixed W, let

    eta = ||W||_2 * sqrt(sum_i E_i^2).

Then ||T0_true-T0_hat||_2<=eta; Weyl gives s_min_true>=s_min_hat-eta.
Report this lower separately; a nonpositive lower requires caution/possibly an
exact observer. Do not infer actual rank from a tiny numerical singular value.

When approximate T0 has full row rank, form dual witness

    Lambda=Y0^T U diag(1/s^2) U^T,
    numerator=<Lambda,Y0^T>=||Theta_min||_F^2.

Any exact interpolant on a true T0 consistent with that error model obeys

    ||Theta_true||_F >= numerator /
        (||Lambda T0_hat||_F + ||Lambda||_F eta).

This follows from Cauchy-Schwarz after pairing the constraint with Lambda and
the perturbation inequality. A lower>16 excludes exact reference-code matching
even under the stated feature uncertainty. Still not a positive categorical-KL
floor unless additional vocabulary-image injectivity/error arguments are proved.
These are F64 evaluated conditional bounds, not interval certificates.

## Proposed gates/resources and decision (freeze with actual code)

One24x256 SVD. No source/native/history/head FG/GPU/optimizer/DEV/RESERVED/T4 call.
Producer60s/OS256MiB/output4MiB; full stored audit120s/OS256MiB/output1MiB.
Inputs about10-20MiB plus small parent receipts; no repeated87GB/6GB scan.
Numeric factor relative1e-10,orthogonality1e-10,code/projection/norm residual1e-8,
radius16+roundoff1e-10. Exact values/refined caps must be sealed before execution.

Branches: feasible preserved correction; only feasible rebuilt interpolant;
robustly outside16 for chosen reference; or uncertain rank/conditioning.
No probability or performance admission from this geometry alone. If feasible,
consider ONE new temporal objective w_first=.5/24 and
w_continuation=.5/[24(m-1)], with original A/W/features/domain fixed, initialization
transport-free from current Theta and all quality gates unchanged. Only the
objective weights change. If geometry is unstable, prioritize the compact
causal-code/core/ternary-function construction or qualified state observation.

The ultimate pipeline still requires a useful source CHATBOT executed by the
original SSM/SWA/LUT/ternary engine, useful RAM-driven experts/structured CPU mass,
physical DRAM and>=50 accepted tokens/s on SAME artifact, plus family/scale
variants. First-response control is a necessary practical part of that transfer,
not a replacement goal or a claim that repeated head work completes it.
