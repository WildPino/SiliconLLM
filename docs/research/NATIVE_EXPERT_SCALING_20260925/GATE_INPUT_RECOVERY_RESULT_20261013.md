# Approximate source in_proj input recovery: complete field consistency

13 October 2026. **COMPLETE / INDEPENDENT_STORED_PASS / ALL FIELD CONSISTENCY PASS.**
Goal ACTIVE/INCOMPLETE. This experiment creates an independently reproduced
approximate calibration proxy from existing captured gates. It has not recovered
true hidden bits or validated temporal, causal, chatbot or throughput quality.

## Question, equations and fixed scope

Reuse all48 existing FIT/DEV histories at sites0/12/23, full-rank3072x2048
trained source gate matrices and actual source BF16 weights. With G the raw
source gate matrix and m the effective BF16 gate MuP scalar:

    P = solve(audited G.T @ G, G.T)
    H64 = (captured_gate / m) @ P.T
    candidate = BF16_RNE(H64)

The unknown module input is AFTER source ssm_in_multiplier .625 and BEFORE
in_proj; .625 is not applied again. H64 is a diagnostic, not an alternative
selected on DEV. No ridge, rank search, refit, gradient update or DEV selection.
The independent audit uses rectangular QR(G)=QR, P_QR=solve(R,Q.T).
Quantization and original CUDA reduction errors mean neither injectivity of G
nor the normal-equation residual proves recovery of the true source input.

The source nonpersistent MuP buffer is BF16 under the actual loading context.
Its five section values are [0.353515625, 0.25, 0.1767578125, 0.5, 0.353515625].
Pinned dtype/construction/capture witnesses and CPU primitive controls establish
conv4 orientation (tap0=t-3/tap3=current), bias, SiLU, delta addition/softplus/
clamp semantics. Actual full projection has6704 rows; section widths are
3072/3072/256/256/48. Replay uses promoted F64 contractions on BF16 candidate
and source weights with explicit BF16 projection/MuP/conv/activation/delta
boundaries. It does not establish original CUDA reduction bit equivalence.
Resolved NumPy ABI/BLAS, psutil, TorchCPU/C10 and HF source files are directly
bound; this preparation avoids the earlier parent-prefix junction omission.

## Fixed field consistency, every case and every field

The frozen approximation-data gate is whole-history relative RMS<=.01 for
EVERY one of144 case/site pairs and EVERY gate/x/B/C/delta field:720 comparisons,
zero failures. The four independently captured x/B/C/delta fields all pass;
gate replay alone would be circular. This new1% calibration budget is distinct
from unchanged native arithmetic tolerances and donor-relative output budgets.

| Site | Field | DEV mean | DEV worst case | DEV worst domain mean | Failed DEV cases |
|---|---|---:|---:|---:|---:|
| 0 | gate | 0.156105% | 0.160931% | 0.160579% | 0 |
| 0 | x | 0.378573% | 0.401408% | 0.394165% | 0 |
| 0 | B | 0.544471% | 0.654586% | 0.604527% | 0 |
| 0 | C | 0.421347% | 0.459717% | 0.454110% | 0 |
| 0 | delta | 0.154927% | 0.176689% | 0.168579% | 0 |
| 12 | gate | 0.164614% | 0.168820% | 0.168638% | 0 |
| 12 | x | 0.468082% | 0.524120% | 0.515046% | 0 |
| 12 | B | 0.529401% | 0.571262% | 0.557127% | 0 |
| 12 | C | 0.458964% | 0.512064% | 0.493136% | 0 |
| 12 | delta | 0.305021% | 0.346686% | 0.326385% | 0 |
| 23 | gate | 0.147141% | 0.153512% | 0.152460% | 0 |
| 23 | x | 0.342639% | 0.358441% | 0.358135% | 0 |
| 23 | B | 0.516750% | 0.707236% | 0.666680% | 0 |
| 23 | C | 0.376531% | 0.449556% | 0.421014% | 0 |
| 23 | delta | 0.212846% | 0.283775% | 0.257919% | 0 |

All24 FIT and24 DEV cases/site and twelve domains are retained in the
[machine result](gate_input_recovery_result_20261013.json), including per-case
centered and label metrics. No subset or alternative recovery was admitted.

### Variation and label diagnostics (worst across all48, not new selection gates)

| Site | Field | Worst centered relative RMS | Worst label relative RMS |
|---|---|---:|---:|
| 0 | gate | 0.205453% | 0.165806% |
| 0 | x | 0.522595% | 0.418023% |
| 0 | B | 0.729680% | 0.749629% |
| 0 | C | 0.542122% | 0.499874% |
| 0 | delta | 0.890429% | 0.190403% |
| 12 | gate | 0.223361% | 0.172366% |
| 12 | x | 0.817606% | 0.573828% |
| 12 | B | 0.705765% | 0.623837% |
| 12 | C | 0.619727% | 0.546865% |
| 12 | delta | 0.772028% | 0.356712% |
| 23 | gate | 0.258903% | 0.153492% |
| 23 | x | 1.070170% | 0.364452% |
| 23 | B | 1.381059% | 0.861885% |
| 23 | C | 0.744827% | 0.700063% |
| 23 | delta | 0.928048% | 0.299119% |

Site23 centered x1.070170% and B1.381059% exceed1%, despite whole-history
consistency passing. Mean subtraction changes the reference-energy denominator;
retain this limit before fitting. Whole-history PASS is not a blanket1% bound
on every temporal variation or every vector coordinate. Maximum candidate
rounding relative to H64 is 0.00161565696134; maximum normalized normal-equation
residual is 3.62355951393e-17. Neither is measured error against the unknown true H.

## Independent audit and custody

[Audit result](gate_input_recovery_adjudication_20261013.json): all fixed controls
PASS, all144 cases, three inverse matrices, every H64/candidate/predicted array,
all metrics/flags/aggregates/domains and complete before-after hashes verified.
Separate shifted-window4tap convolution, sign-split sigmoid and stable softplus
formulas are used; the producer replay helper is not imported.

- Inverse coefficients: 18,874,368.
- H64 words: 138,528,768; BF16 candidate words: 138,528,768.
- Predicted BF16 field words: 453,465,264.
- Full inverse versus rectangular QR relative error: 7.26302073547e-15 (gate1e-10).
- Full H64 versus QR relative error: 1.54530098633e-14 (gate1e-9).
- Predicted gate/x/B/C/delta replay relative errors: gate=0, x=0, B=0, C=0, delta=0 (gate1e-6).
- Scalar metric reproduction maximum: 0 (gate1e-8).
- Candidate bits exactly match integer BF16 RNE of stored H64.

Producer freeze f51f63c; binding SHA256
`44f61d400d980cc1550eae8d7440a913c60b811bc65c865cf538c035cd5e8536`.
Producer launch2aec0c3; auditor freeze7074212; audit launch788071c.
[Producer launch](gate_input_recovery_launch_20261013.json) and
[audit launch](gate_input_recovery_audit_launch_20261013.json) contain exact
argv/environment; protocols fix all equations, gates and caps before observation.
Result SHA256 `0ae19f5fcb114c42c37b0d7ba018136f4b638409f43b85516a9649eea79d7f6f`;
adjudication SHA256 `bd8b663feb9d28c635fafe87f308a03ad1f21035af76a74374ae5268d9ef7e2e`.

## Resources and terminal closure

| Stage | PID | Elapsed including closure | Peak OS working set | Exit |
|---|---:|---:|---:|---:|
| Preparation | 7328 | 25.141205 s | 520,273,920 B | 0 |
| Producer | 13352 | 177.337299 s | 748,306,432 B | 0 |
| Independent audit | 4944 | 139.846878 s | 1,090,789,376 B | 0 |

All owned PID/creation pairs absent at closure. Preparation/producer sessions
80263/46665 were already complete when this turn resumed; neither was repeated.
Audit session7256 finished with terminal0. CPU0..5/six BLAS threads;
limits producer900s/audit1200s/OS6GiB/output3GiB. No timing admission.
New E directory `E:/SiliconLLM/results/native_expert_scaling/gate_input_recovery_20261013` contains147 NPZ files,
2,443,476,576 B (2.275665 GiB), within3GiB. It retains three P maps and
144 histories, each with H64/H_BF16 and five predicted BF16 fields.
No GPU/T4, donor model forward, native call, optimizer, new teacher reply or
RESERVED access. Engine and three foreign tracked file hashes unchanged;
foreign daemon snapshot and one read-only diagnostic parse error retained in
[session closure](gate_input_recovery_session_closure_20261013.json).

## Decision toward the pipeline

The recovery removes one missing offline calibration-data boundary without
another donor forward. It preserves independent local source-function outputs
under the fixed field-consistency budget. It provides approximate inputs for
initialization/analysis; it is not exact hidden supervision or a cheap runtime
encoder (the supplied gate already comes from the donor).

Next: [temporal propagation gate](RECOVERED_INPUT_TEMPORAL_NEXT_20261013.md),
PROPOSED/UNIMPLEMENTED/UNBOUND. Regenerated operands must pass through the SAME
fixed trained full3072/state96 native recurrence/output and be compared with
qualified original-operand states/outputs, before calibration admission.
Keep per-case final-state and own-norm physical-output1% gates separate;
retain original donor-relative local gates and full audits even failure.
Then fitting/compiling cheaper generator/readout/conditional ternary functions
must connect to a shared core producing its OWN causal history. Full dense
source generators/readout remain expensive references. Useful n, CPU IDs AND
masses, actual expert DRAM, O(n) conversion masters/gradients, multiple donor
families/scales, donor-relative chatbot quality and>=50 useful batch1 token/s
on the SAME complete artifact are still unproved.
