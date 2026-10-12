# Preserve trained functions where cheap;reduce actual active cost

13 October2026. **SELECTED / PROPOSED / UNIMPLEMENTED / UNBOUND.**
[Actual native cost result](COMPOSED_OPERATOR_MIXED_RESULT_20261013.md) changes
the next action:state384x96~10us,normalize/output~1157us,retained input generator
separately~2535us on one core. No joined-layer rate or whole chatbot admission.

## 1. Do not enforce a channel bottleneck solely to shrink a cheap state

The source-state-only dual96 maps retain ALL3072 channels and have independently
measured DEV local output errors~7.63%/8.72%/2.49% at0/12/23,versus failed
fixed/adaptive384 channel maps at0/12. The current joint local pass is site23
only. These are different valid operator approximations,not global capacity
floors. All generator/gate functions remain supplied in these comparisons.

Selected bounded cost control:actual C state-only3072x96 versus384x96,with
source head scalar decays/both norm modes/all48 histories. Reuse exact FIT
dual96 maps,no refit/DEV choice. Source3072x96 F32 state1,179,648B versus147,456B;
8x state-coordinate ratio is arithmetic,NOT presumed8x whole-runtime cost.
Record projection/scan/gate/norm/out stages and full numerical/source-quality
audits before any cost admission. Original engine.c remains immutable.
Producer<=900s/audit<=1200s,OS6GiB,newE artifacts<=2GiB per frozen family proposed;
actual protocol/runtime/compiler binding required before launch. No T4/hours job.

If state-only cost is a small part of the active budget,preserve those trained
functions rather than spend long fits repairing an arbitrary384 bottleneck.
If cost is material,revise the shared state geometry using causal/output
weighting. Width does not have to scale with total expert n;neither does this
single-source example establish a general constant-cost mapping from100B.

## 2. Cheap input/output geometry still requires information and causal closure

For illustration only,changing input hidden2048->256 would change the copied
6704-row generator coefficient count13,729,792->1,716,224;output2048->256
would change3072-channel out count6,291,456->786,432. These are matrix dimensions,
not measured target kernels,quality-preserving encoders or a selected model.
Prior fixed P256 shadow is0/16;preserving source states alone did not solve its
hidden-information/whole-model failure. Conditional functions/changed coordinates
must be trained or derived and tested with their OWN causal histories.
Do not adopt a generic donor runtime or assume depth24->6 composes harmlessly.

Next data uncertainty:the captured postconv x/B/C/delta/gate/y/output does not
include the actual input of source in_proj. The full FFN capture has a different
module input and only label positions;it is not this missing generator input.
Existing recurrent in_observer receives args but saves only gate/delta.

First investigate an algebraic reuse route before an expensive source replay:
the3072 pre-SiLU gate features are linear measurements of the2048 in_proj input.
Source code applies ssm_in_multiplier before in_proj and MuP sections after it;
captured gate is BF16. Define the exact scaled gate matrix G from actual weights
and buffer/rounding semantics. If full column rank,an unrounded model has
`h_hat=G^+ gate` and `||h_hat-h||<=||G^+||*||quantization/reduction_error||`.
Rank/condition and explicit BF16 intervals are unknown and MUST be measured.
This is a possible OFFLINE calibration-input reconstruction,not a cheap
runtime algorithm,not exact recovered hidden bits and not a causal encoder.

Preflight choice:full source G singular spectrum/condition,scaling/dtypes,
FIT-only recovery policy;then all48 generator/conv/B/C/delta/gate replay against
OTHER captured fields,with rounding errors/uncertainties explicitly retained.
Recovering gates from gates alone is circular and cannot qualify generation.
If the measurement is rank-deficient or uncertainty is too large,capture actual
in_proj input with the pinned donor/runtime and verify stored-field continuity.
Freeze selected scope/caps BEFORE that source execution;do not query RESERVED.
Do not substitute inferred source inputs for true hidden-state supervision
without the stated multi-operand qualification and uncertainty bounds.

## 3. Conversion work that must follow

Learn/derive compact causal hidden->generator/readout functions,shared recurrence
and selectively consulted ternary bank under full donor categorical/feature
supervision. Dynamic gated readout cannot be folded into one constant cheap
matrix;mixed maps cannot simply commute through conv/SiLU. Keep trained-state
initializers and the qualified differentiated-precision boundary available.
Validate full composition and own emitted histories,not only injected features.
Routing IDs+masses,actual expert DRAM and dense master/gradientO(n) remain open.
First useful same-artifact quality+>=50 case,then other families/scales.
