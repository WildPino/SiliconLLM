# METH-230: source nonlinear readouts fit well, but one child remains a weight copy

The [protocol](METH_230_SOURCE_ANCHORED_NONLINEAR_PROTOCOL_20261001.md)
and runner were frozen at `f16722f`; session32860 completes,exit0.
Source/routes/selection/gradient checks, all176 solve/intercept checks
and every snapshot tensor readback pass. Fit normalized SSE:initial
source prior.1128587811,trained E16.0305380161,E160.0072337443. These
are fit measurements, not independent quality or even validation.

**Fixed stop:** only175 of176 fitted affine/down pairs have distinct
BF16 bits. E160 child156 equals E16 parent15 in both weight matrices.
Its bias differs. The distinct-weight gate fails; validation never runs
(result summary/rows empty). Do not describe the fit error below .01 as
accepted knowledge transfer, useful n or full quality.

An independent read-only fit-data replay (session80423,exit0) diagnoses
this exact pair: child156 has18 states but **one unique full input**,
maximum input and selected nonlinear-feature variances both0. Its dual
normal-equation residual is0 and intercept shrink18/(18+1024)=.01727447.
Bias change L2=2.804083e-5. This source of degeneracy is not a solver bug:
centered fit features supply no slope information, so the parent prior's
weights remain unchanged. No source/validation targets are opened by that
read-only diagnosis beyond already consumed fit observations.

**Decision:** stop this fixed parent-only child-weight anchoring recipe.
The changed [METH-231 protocol](METH_231_SOURCE_DERIVATIVE_CHILD_PROTOCOL_20261001.md)
would transfer original-source derivatives into **every** child prior,
while inheriting the existing trained parent nonlinear response. This
supplies real source slope information absent from repeated observations;
it does not perturb weights to satisfy a hash gate. Routes, features,
parent controls, strength and width stay fixed; no gate is relaxed.

[Raw result](meth230_source_anchored_nonlinear_result.json),SHA256
`724837b3f5093e0d388477755d02f9dbae72ffdda6c10051c64aad14ed436dd7`.
Local ignored snapshot `results/native_expert_scaling/meth230_layer12_source_nonlinear_functions.safetensors`
is1,131,947,520bytes,SHA256
`84d5bfc071dd598173bae1bfb9b799cb2fd52e115459f225d5418b4549acfffe`.
Runtime61.766s after imports,ending RSS3.238GB,GPU peak1.755GB,local3060,
no T4. The initialization warning/partial progress are preserved. No job active.
