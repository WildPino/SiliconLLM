# FIT recurrent-output bases with retained full skip

12 October2026. **COMPLETE / INDEPENDENT_STORED_PASS / GLOBAL LOCAL GATES FAIL.**
Goal ACTIVE/INCOMPLETE. This is source-injected operator evidence,not a chatbot.

## New variable and method

Preceding [full-skip comparison](RECURRENT_ONLY_CHANNEL_SKIP_RESULT_20261012.md)
used bases fitted on full y. Now choose FIT-only equal-case per-head covariance
GY_rec from y-D*x;retain the old GM exactly. Balanced SVD of sqrt(GM)*sqrt(GY_rec)
produces biorthogonal master V/W. Fixed8 or adaptive min1/head plus336 largest
remaining squared singular values,total384. No DEV fitting/rank selection.

Actual NEW384-channel F32 SSD scans with the new W,source state256/B/C/delta/A,
all48 histories/sites0/12/23. Stored projected inputs/scan values are not old
latent reuse. Decode recurrence,add full source D*x,then jointly gate/normalize.
Compare projected-skip arm separately. Outputs use controlled F64 contractions
on promoted actualF32 maps/scans/inputs,then explicit BF16 input/output rounding.
Old-F32 precision equivalence is still not admitted.

## DEV results

Mean equal-case output relative RMS with reconstructed denominator/full skip:

| Site | Old all-y bases fixed8 | New recurrent fixed8 | Old all-y adaptive384 | New recurrent adaptive384 |
|---|---:|---:|---:|---:|
| 0 | .290536 | .290525 | .286465 | .273076 |
| 12 | .799232 | .797691 | .791990 | .792727 |
| 23 | .304151 | .212568 | .303801 | .063571 |

Site23 adaptive passes all four original local gates in BOTH denominator modes:
ownnorm mean.063571,worstcase.087007;source-denominator mean.064564,worst.087047.
The original thresholds remain mean<=.1,worstcase<=.2,worstdomainmean<=.2,
centeredmean<=.1. Sites0/12 still fail;no layout passes every site. Global local
quality gate FAIL,not a preserved chatbot or whole-model admission.

Adaptive allocation matters:site23 assigns58/63 channels to heads20/31,with the
other heads sharing263. Fixed8 cannot reproduce this result. Those two retained
widths nearly equal their original64;count384 does not imply uniform compression.
Total source SSD state3072x256 versus projected384x256,an8x recurrent-coordinate
reduction. Full source feature generation/gate/conv4/3072 skip remain retained.
No8x whole-model cost claim,engine-native rate or stateless raw-token claim.

## Numerical and temporal qualification

All288 packed scan/layout/history records are newly computed (grouped scan
function calls can be more numerous). Source projected full/recurrent agreement
max3.91454e-7/5.62910e-7. Prior old-basis projected/fullskip controls reproduce
every previously audited scalar exactly,maxdelta0.

Independent CPU audit covers all FIT recurrent covariances,unchanged GM,288
head/layout bases and rank allocations;every51,948,288 projected input words and
51,948,288 scan words via direct384-channel,state256,stepwise recurrence.
All1,108,230,144 BF16 output words,energies/cross terms,labels/aggregates/domains/
gates and before/after hashes are checked. Numerical pass is separate from
quality,and does not retrospectively pass the prior original-F32 bridge.

Full independent audit PASS:covariance relative1.08e-15,weighted kernel8.18e-13,
projected input2.04e-7,full sequential scan2.93e-7,stored output relative0,
metric delta1.65e-14,energy2.41e-15. Every bound hash before/after exact.
Separate closure check confirms all144 unique expected case/site pairs and
exact sites0/12/23;no omitted/duplicated case. Audit379.76615s,OSpeak781,139,968B.
All owned sessions closed;no owned run live.

## Resources and custody

Producer297.87985s/terminal298.00955s,OSpeak1,337,954,304B,CUDA allocated
1,310,158,336/reserved1,470,103,552B.3 basis NPZ+144 output NPZ total2,663,633,148B.
Preparation23.710s,tool52288closed0. Capture30075closed0,PID30212,
creation1791833747.0317235. Source/protocol freezee409079;binding SHA256
`b9a4f56cca82a6970582ff7d069bc878b95b60bb8b00a62f04083aaf8f295c18`.
Audit source/protocol freeze1f06599,launchda67157. No clean speed admission.

The initial Git path-array attachment command failed before capture;binding/
launch were already written and consumed unchanged,and attached immediately
after process creation. [Sequencing fault retained](RECURRENT_OUTPUT_BASES_LAUNCH_CUSTODY_20261012.md).
Do not claim the Git launch attachment preceded creation. Full pinned source/
input/result hashes are the numerical custody checks. No source/model/optimizer/
native/T4/RESERVED query in this experiment.

- [Producer](../../../benchmarks/native_expert_scaling/recurrent_output_bases.py), [protocol](RECURRENT_OUTPUT_BASES_PROTOCOL_20261012.md), [binding](recurrent_output_bases_binding_20261012.json), [launch](recurrent_output_bases_launch_20261012.json)
- [Result](recurrent_output_bases_result_20261012.json), [terminal](recurrent_output_bases_result_20261012.terminal.json)
- [Audit](../../../benchmarks/native_expert_scaling/recurrent_output_bases_audit.py), [audit protocol](RECURRENT_OUTPUT_BASES_AUDIT_PROTOCOL_20261012.md), [audit launch](recurrent_output_bases_audit_launch_20261012.json)
- [Adjudication](recurrent_output_bases_adjudication_20261012.json), [audit terminal](recurrent_output_bases_adjudication_20261012.terminal.json), [closure](recurrent_output_bases_session_closure_20261012.json)

## Decision

One layer's trained recurrence now has a locally accepted compact channel
representation under source injection;uniform allocation and full-y covariance
were inadequate there. Site12 remains the major causal representation defect.
Do not repeat unchanged averaged fitting:next [coupled sensitivity diagnostic](COUPLED_RECURRENCE_SENSITIVITY_NEXT_20261012.md)
separates RMS radial effects,same-time gate dependence and cross-head coupling.
This can select an objective correction before long adaptation. Site23 is also
a candidate for explicit source-derived native layout/cost qualification,with
all retained source work counted. State96 composition/native/own-history/model/
n/families/same useful artifact>=50 remain open.
