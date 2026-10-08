# Next actual converter problem: joint values and complete source derivatives

8 October2026. [Completed directional result](CHATBOT_DIRECTIONAL_RESULT_20261008.md)
supersedes the previous collector NEXT. Full CHATBOT-to-engine goal unchanged.
This document is a PROSPECTIVE implementation plan: learner/binding/actual
training protocol are not implemented or executed here. Bind all actual code/
used bytes/runtime and final criteria before the first new fit. Any change must
be made explicitly before those values; no consumed-result parameter ladder.

## Why this is the next conversion step

The whole chain has an available canonical input adapter, original calibration,
necessary compact cost and reusable archive/native loader. Source functions
are the missing connection. Supported E16 value fit has21.68% FIT/61.39% novel
response RMS and now82.74% full-J RMS on BOTH representative FIT and novel states.
Encoding contributed negligible response change. UniformE160 support failure
remains closed. There is no eligible new compact function bank to export.

Choose a NEW learning problem at shared512/leaf128/top4/query32/E16, keeping
geometry and per-token work fixed. Original row initializer/geometry are reused,
not repeated. Start from the actual retained failed final F32 coefficients;
reset Adam moments since none were saved. Keep original source-row initializer
checkpoint as coefficient reference. Reuse original x/y and saved E16 parent
IDs/mass for response batches; use ONLY16 FIT Jacobians as derivative targets.
Sixteen DEVELOPMENT Jacobians and responses are evaluation, never fitted.

## Algebraic objective and fixed proposed recipe

Let g_theta be the actual joint full-input SwiGLU target with fixed selector.
Every shared/leaf G/U/B is trainable. No fitting of only a readout or fixed G/U:
old231 is a different closed class. For each response batch:

    L_value = sum_r omega_r*||g_theta(x_r)-y_r||² /
              sum_r omega_r*||y_r||²,
    omega_r = 1/(exact FIT multiplicity of x_r).

For a FIT anchor a selected round-robin0..15:

    L_J(a) = ||J_g_theta(a)-J_source(a)||F² / ||J_source(a)||F²,
    L = L_value + L_J(a) + .001*original_normalized_coefficient_anchor.

Include selected-mass derivatives. Use full896x896 J, no cropping or replacing
the nullspace with only the32 selector coordinates. Coefficients/loss arithmetic
F32; saved source F64 J targets are rounded once to F32, without recomputing the
source. Final source/student derivative diagnostic uses exact stored coefficients
promoted to F64 as in the completed collector. The inference block/active cost
is unchanged; derivatives are conversion-time work only.

Hold batch128/final batch5/24 epochs/744 steps, Adam.0003/.9/.999/eps1e-8/
no weight decay/foreachFalse, clip1, seed2601007/TF32off/deterministic order from
the previous finite fit. One complete FIT-J loss each update: eight anchors47
times and eight46 times. Save every update loss/anchor index/clip norm/actual
elapsed. Select final epoch only, no development-based checkpoint/early stopping.
The changed variable is source-structural supervision, not more training of the
closed objective. Normalized derivative coefficient1 is fixed, not tuned.

This uses known source information in a real converter optimization problem.
It does not promise convergence, prove sufficient representational capacity or
turn full-space Jacobian preservation into a semantic-quality theorem. Full-space
directions may differ from the actual activation covariance; measure whether
the new objective helps previously reserved responses rather than assume it.

## One finite trial, explicit decision and cost

Proposed local3060/no T4: worker360s/family540s, fit arm240s, summed OS4GiB/
GPU allocated4GiB/reserved6GiB/output512MiB/log2MiB. Previous value-only arm66.187s
is observed context, not a measured derivative-fit cost. Stop first nonfinite/
shape/dtype/resource/input/gradient failure; preserve update journal and any
completed checkpoints before a separately scoped repair. No completed fit replay.

Before promotion require final F32 novel response RMS<=1%, ALL16 category<=3%,
every function finite/distinct/used and frozen resource gates. These original
absolute fidelity criteria do not change. Necessary matrix/logical-cost ledger
is reused; no rate/DRAM extrapolation. Only a successful F32 candidate enters
separately qualified BF16/export/native stages.

For mechanism evaluation, proposed gates fixed before values:

- novel response RMS improves by>=10% relative to saved baseline61.3920583%;
- FIT response RMS does not worsen by>5% relative to21.6779384%;
- no development category RMS worsens by>5% relative to its saved category;
- novel AND FIT full-J RMS each improve by>=10% relative to82.7371307%/82.7386834%;
- final Jacobians/directional checks/provenance/actual resources pass.

Mechanism improvements are diagnostic only if absolute fidelity fails. Close
this ONE finite recipe in that event, retain its measured benefit/failure, no
automatic lambda/epoch/width ladder. If structural supervision does not improve
novel responses, re-examine representation/source-feature coverage or the relevant
activation-direction metric; do not infer that all derivative learning is impossible.

Final observations should include two new response matrices (FIT/development),
final F32 coefficients and32 NEW student full-J matrices/perturbation checks.
Reuse existing source J/probes, never recapture source output or repeat geometry/
initializer/a-squared features. Independently audit saved new outputs/decisions;
do not rerun any completed model/control/audit for cosmetic reasons.

## Return to the full pipeline

Eligible functions -> bounded ALL24 converter -> complete bank/export/new C
profile/canonical adapter -> FRESH donor-relative own-history dialogue/tasks ->
quality AND>=50 accepted IDs/s on SAME artifact, including prefill/full head/
attention/cache/router/first vs warm and actual selected DRAM.

Large n needs support-dependent allocation or NEW qualified source information,
with useful capacity at matched active work. Uniform10 children is not currently
eligible. RAM/redundant copies can provide overlapping source coverage; their
usefulness must be measured. LUT winner AND normalized mass, Giga~10B operator/
canonical variants, other families/~100B remain required, not declared supported.

Exact resumption: verify branch/foreign SHA/processes, read this NEXT/result,
implement one differentiable full-J joint learner and its used-input binding,
freeze actual loss/layout/checks/resource/decision, then run ONCE. All original
source capture and the first directional plan/acquisition/audit are complete;
reuse their bytes. No current live scientific process or missing approval.
