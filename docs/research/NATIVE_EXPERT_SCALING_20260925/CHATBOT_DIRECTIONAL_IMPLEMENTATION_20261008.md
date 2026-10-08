# Phase2 implementation freeze and phase1 provenance correction

8 October2026. No derivative values observed. Phase1 executed ONCE with all32
eligible distinct original anchors, exact real rank32/null864. Actual bound
source commit ec197f9efdf1b38dfdc6f8a958e2203a354ee6c8. Its raw revision label
was incorrectly passed as `9bbeef-placeholder`: that FIELD is INVALID, not
silently corrected. Raw/terminal bytes are preserved. Separate administrative
`chatbot_directional_plan_provenance_20261008.json` reconstructs every bound
committed input byte from the actual commit with only LF/CRLF conversion and
binds raw/terminal SHA. No anchor or eligibility replay. Launcher now requires
an explicit40-hex revision. This amendment is bound before phase2 values.

The implemented phase2 follows the original prospective diagnostic protocol:
full896x896 source/student analytic F64 Jacobians, exact stored coefficients,
CPU F64 QR, fixed original parent IDs, selected F64 mass, mass derivatives.
Extra derivative operands include individual selected leaf values at anchors;
no full old student/source anchor response is repeated. No optimizer/geometry/
source-a-squared features/original BF16 full model forward.

Retain Q[896,32], five unit directions[5,896], all source/student J[32,896,896],
all derivative vectors/probes[32,5,2,4,896]. Probe axes explicitly denote
source/student then Jv/finite-difference/f(x+h*v)/f(x-h*v). These point values
are NEW smooth-surrogate perturbations, not original inference or chatbot
quality. Per-anchor journal records exact output frame offset/bytes/SHA and
full scalar metrics; exclusive NPY files flush/fsync before journal completion.
Partial frames survive a first fault; no completed scientific replay.

Additional prospective observable: selected-mass Jacobian total/parallel/null
energies and closure, using the same full matrices already needed. This tests
the algebraic fact that mass derivatives lie in the selector rowspace. It does
not change the four frozen diagnostic decision thresholds or prove global
continuity at hard-selection boundaries. Equal-anchor pooled metrics are not
source-row/conversation prevalence or whole fidelity.

Source is original Qwen2.5-0.5B-Instruct layer12 SwiGLU; student is the retained
failed E16 final F32 checkpoint. Full source SHA fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe;
student SHA f7ec59fcf608748af4232b7afc4eca83012eca852b5fcb7e94ff682a8a761b5d.
The numerical formulas are F64 representations of BF16/F32 coefficients, not
the derivative of a rounded program or an inference-precision repair.

Original phase1 worker2.172s/family4.640s/OS49,393,664B through exit. Rank minor
columns0..31, nonzero determinant163870988 modulo1000000007 after exact2^-149
scaling. All32 score margins>.000315754; maximum savedF32/shadow mass gap
1.606e-7. These are retained plan observables, NOT Jacobian results.

Phase2 uses Torch2.6.0+cu124/NumPy2.4.6, local CUDA0/CPU6 threads/TF32off/
deterministic algorithms, restricted72-root runtime. Keep worker180s/family300s/
summed OS4GiB/GPU allocated4GiB/reserved6GiB/output512MiB/log2MiB. Independent
finite-difference tolerance and prospective four-threshold decision unchanged.
Future retained audit reads saved derivatives/probes only; no source evaluation.
