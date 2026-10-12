# Actual D256 trained operator export: norm folding passes, diagonal control fails

12 October2026. Goal ACTIVE/INCOMPLETE. Previous and current goal turns are
progress: full-D transport first, actual restricted D256 compiler/export/audit now.
No owned job live, no useful chatbot/speed/generality admission.

## Experiment, provenance and scope

[Protocol](FALCON_PROJECTED_S6_TRANSPORT_PROTOCOL_20261012.md),
[compiler](../../../benchmarks/native_expert_scaling/falcon_projected_s6_transport.py),
[binding](falcon_projected_s6_transport_binding_20261012.json),
[exact invocation](falcon_projected_s6_transport_launch_20261012.json).
Source/protocolbb9383a, binding/invocationbe656e5. Binding SHA
c3490181a801c4b9fb115dec40d756e74e7ba55c75570d6a54ee0ed8b1a11612.
Actual inherited P256 SHA3dd96e0fd782833f51c005109e7ba1f95a98996bac1f5babb6a43ab3833e465d,
no basis reselection or training; measured P^T P defect<=7.7e-7.

Actual source sites0/12/23,13 FIT embedding-derived vectors each. Let
z=embedding@P; source reference raw input is Pz. These are INJECTED restricted
inputs, not actual source intermediates or full-source fidelity measurements.
Each actual source input norm gamma folds into the input matrix; use G=P^T P,
target input epsilon8e-5 and sqrt(8) coefficient factor. Keep source postgate
epsilon1e-5 and3072 primary channels. Source postgate gamma is folded into output
columns, then output projected by P^T. Keep full source feature/state widths.

## Measured outcomes

[Raw result](falcon_projected_s6_transport_result_20261012.json),
[terminal](falcon_projected_s6_transport_result_20261012.terminal.json),
[independent full stored audit](falcon_projected_s6_stored_adjudication_20261012.json).

| Site | Max8 exact operand errors | Max coefficient-roundtrip errors | Diagonal gamma output error | Identity G output error | Wrong input epsilon error |
|---|---:|---:|---:|---:|---:|
| 0 | 1.31487e-15 | 2.86637e-8 | 28.3677% | 5.15825e-8 | 0.0474288% |
| 12 | 1.40936e-15 | 4.55385e-8 | 92.4334% | 6.35393e-8 | 0.106785% |
| 23 | 1.19695e-15 | 3.69225e-8 | 19.0556% | 3.47548e-8 | 0.0363322% |

Relative RMS, not task accuracy. All exact<=1e-10, coefficient-roundtrip<=1e-5
and scalar recurrence<=1e-10 gates PASS. All controls evaluated, no tuning/selection.
F32 coefficient/G/epsilon exports are promoted to F64 to isolate coefficient
rounding; **F32 runtime accumulation/cache/transcendentals and C parity untested**.

Diagonal control uses projected gamma=(P^2 gamma).sum and trained source matrices
without exact gamma folding, otherwise identical metric/epsilon/output map.
It demonstrably does not preserve the restricted function on these inputs.
This is NOT an attribution of all learned chatbot failure: its sequence core
started fresh, later trained, and source/target histories/routes differ.
Small identity-G errors here are not a bound on all inputs/composed behavior.
No universal D256 capacity ceiling follows. Residual projection between source
sublayers still discards information even when each restricted operator is exact.

## Export and actual costs

Three durable F32 original-S6-field packages:24,388,228 payload bytes EACH,
including input metric/epsilon and exact-zero auxiliary output columns.
D256/DN3680/N256/DT48;6,031,520 dense core coefficients before metric/epsilon,
5,063,680 dense linear MAC/operator/token. Naive state942,080 F32 values/operator,
source-primary-only786,432. No sparse/native optimization measured.
Across24 source layers this shape implies585,317,472B coefficient/metric payload
and121,528,320 dense linear MAC/token, excluding attention/FFN/head/nonlinear/state
work. Shape accounting, NOT a full-model latency or selected cheap endpoint.

Preparation .032s. Capture89624 CLOSED exit0,13.047s/OS530,104,320B. Six new NPZ
files82,985,472B:three coefficient packages,three complete reference/compact/
roundtrip/control frame packets.300s/OS4GiB/output128MiB gates PASS.
Result SHA79d2d50f478853500f15d646704f37d358f3ec9c0722cd5394e9a2486938bb82;
terminal SHA698d41c13392394ac58351e24d1919d3e3b4aeb9c6ebccfdd5b053f9db1644d9.

Independent [audit source](../../../benchmarks/native_expert_scaling/falcon_projected_s6_stored_audit.py)
frozenf4a83c7:exit0/8.094s/OS279,605,248B,120s/1GiB caps PASS. Every field
checked against an independent DIRECT source map, no compile_lift replay.
Weighted matrix coefficient sup-relative bound predeclared1e-6 for distinct
contraction/scaling order; actual maximum3.9422954269381584e-8. Copied/selector/
zero/control fields exact F32. Python fsum verifies all8 saved reference/compact/
roundtrip fields and all controls:maximum report delta3.3306690738754696e-16.
Three scalar recurrences use actual raw A/D and saved x/B/C/delta. Full input/
output hashes before/after PASS including complete source weights and P.
Audit SHA3266c847061ea903c28872a51c4185e508cb40818588f60a20c332914053c985.
No generator/model/native/optimizer replay in audit. Principal runtime binding,
not full DLL tree. Foreign publisher/uploader preserved, timing isolation not
admitted. engine.c unchanged; no GPU/T4 allocation in this experiment.

## Decision and exact next action

Restricted pretrained compiler IMPLEMENTED/LOCALLY QUALIFIED, F32 packages
available. This is not current packed ABI support: dimensions/input norm metric/
epsilon/primary-only postgate norm require an explicit consumer. Residual width
is compact; source feature/state widths remain large and provisional.

Next implement/review/bind one coherent24-layer pretrained projected shadow:
actual SSM+attention parallel composition, source SiLU FFNs/depth/tokenizer,
correct norm folding at EACH boundary/final head. Price actual arrays, freeze
precision/operator/readout controls and budget before execution, then existing
16 canonical tasks+own-answer followup and24 DEV labels/quality criteria.
No long optimizer epoch chosen. Native LUT/ternary selective capacity, useful
n/CPU IDs AND masses/physical DRAM, useful same-artifact50 and family/scale
extension remain required; the shadow itself cannot conclude the final goal.
