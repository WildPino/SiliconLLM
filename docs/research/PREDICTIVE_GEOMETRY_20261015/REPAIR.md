# Qualified numerical frame repair

15 October 2026. The initial geometry kernel terminated with an explicit
numerical-control fault before any held-out score evaluation. Its two actual
source directional derivative checks both passed. FIT score collection completed,
but the FP32 SVD's Fisher vectors had maximum Gram error
`0.00010800361633300781`, exceeding the frozen `1e-4` orthogonality tolerance.
The original fault, terminal and manifest remain preserved. It is not a negative
scientific result, nor a successful geometry study.

## Correction

After SVD, apply an unpivoted reduced QR to each ordered basis, then perform the
same orthogonality check at the same threshold. In exact arithmetic this preserves
each leading column span, so the nested-rank experiment, FIT examples, categorical
samples, PCA comparison and held-out decision rules are unchanged. Record Gram
errors before and after QR. Do not increase tolerance or remove a failed control.

This is one explicitly qualified replacement, not a hyperparameter search. It
uses a new private kernel identity `wildpino/predictive-geometry-qr-repair-20261015`
and distinct frozen source/attempt records. The original three bundles are not
overwritten. The temporal/reuse kernels completed and are not restarted.

## Budget custody

Before replacement, authenticate that the original kernel is terminal ERROR,
verify its manifest against the bootstrap binding, verify the failed numerical
report and passed derivative checks against that manifest, and confirm no held-out
gradient records exist. Bound the original used allocation conservatively by all
wall time since its recorded launch attempt, including queue and post-failure time.

The replacement platform cap is the smaller of 6,600 seconds and the original
7,200-second geometry allowance minus that conservative elapsed bound and a
180-second reserve. The new bootstrap/numerical deadlines shrink accordingly.
This releases only verified unused time from a terminal job; the geometry lineage
remains within four conservative T4-hours. The whole wave remains within twelve.
The launcher allows exactly this one replacement reason and refuses duplicates.

The source and receipt changes were checked structurally locally. Actual numerical
QR checks and the original experiment run on Kaggle. No local tensor calculations
or model inference are authorized by this repair record.
