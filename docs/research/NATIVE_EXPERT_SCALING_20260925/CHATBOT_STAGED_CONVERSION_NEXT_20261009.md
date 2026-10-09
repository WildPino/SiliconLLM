# Next conversion stage: preserve a useful donor before composing losses

9 October2026. Source FFN conversion/preflight IMPLEMENTED/COMPLETE;
[preflight result](CHATBOT_SOURCE_TERNARY_FFN_PREFLIGHT_RESULT_20261009.md).
The [full control](CHATBOT_SOURCE_FFN_CONTROL_RESULT_20261009.md),frozen
d4f7f0f/98-input3d86ba62,is COMPLETE/qualityFAIL,session25136closed. The
[F32 positive control](CHATBOT_SOURCE_FFN_CAST_RESULT_20261009.md) is COMPLETE/
PASS,d94f854/00251063,ALL16 original source generations exactly preserved.
[Next local recovery](CHATBOT_SOURCE_FFN_RECOVERY_NEXT_20261009.md) is selected,
UNIMPLEMENTED/UNEXECUTED. No training or final native
architecture admission. Goal remains useful
chatbot->original LUT/ternary/compact SSM/SWA with same-artifact quality+50,
then useful large n/addressing/mass/DRAM/families. Month-plus T4 remains allowed
after an actual feasible conversion stage and communicated budget/stops.

## Why this changes the next action

[Broad24](CHATBOT_BROAD_PILOT_RESULT_20261009.md) reduces newDEV KL9.823->6.991,
but absolute/domain gates fail andoldDEV greedy agreement worsens. The pipeline
currently changes width2048->512,depth24->12,parallel attention/recurrent
composition,SSM channels,FFN sum->top8 normalized mixture,andternary/AQ63
together. Weight-informed initialization is available;usefulness is not preserved.
One small pass cannot distinguish inadequate coverage/convergence from a
representation/selection/precision problem. Avoid selecting another larger
active target solely from that result.

## First concrete control: implemented and priced

Use the pinned Falcon1.5B as an OFFLINE intermediate. Retain original source
width/depth/SSM/attention/norms/tokenizer/readout andcomplete FFN row coverage.
Change ONLY the FFN arithmetic package to the actual deployable ternary/AQ63
integer-dot/F32-row-scale/signed-SiLU formulas already used in engine.c.
Exact source MUP multipliers/operator order/cast boundaries are mapped in
the frozen [preflight protocol](CHATBOT_SOURCE_TERNARY_FFN_PREFLIGHT_PROTOCOL_20261009.md):
actual intermediate width4608,multipliers.4419417382415922 and.13020833333333331.
Source FFN sum is not a normalized selected mixture.
Do not reuse compressed-target trained master weights as an allegedly trained
F32 positive control;they were optimized under a different forward formula.

This retains a source-shaped intermediate offline;it is not an affordable final
runtime or goal completion. Purpose:measure the first operator conversion loss
while source knowledge/trajectory organs andcoverage are still present. If it
loses usefulness,price recovery there before adding compression/selection.
If it retains useful behavior,then change one of selection/representation/core
at a time and recover,carrying whole-output/own-history evidence between stages.
Redundant conditional functions remain a way to trade stored capacity for
fixed selected work;coverage/coefficient requirements still must be learned.

Reuse the original8808 source rows/IDs/forced histories andpinned3.11GB source;
NO regenerated source replies. Keep original cached execution/precision for
unmodified organs andonlyqualified source SSD storage helper. A full-prefill
versus cached schedule change is a separate variable,not free bit-equivalence.
Metadata-only shortest/longest FIT pricing is complete:7.390s conversion,
340,819,968B actual packed FFN payload,GPU4.902/7.155GB,heldOS3.724GB/family77.781s.
All six actual integer-dot witnesses match every CPU F64 output exactly.
Two FIT KL1.74557/2.14675 and5/18,141/256 differing IDs are retained,not broad
quality admission. Int8 cached codes/one temporary F32 projection are actual
implementation. The full control reuses both trajectories andactual payload
without recalibration;46 new forced trajectories/16 screen+4 actual own-history
generations have criteria frozen before observations. No training/native/T4.

Original proposed1800s envelope is superseded BEFORE full observations by
the [priced control protocol](CHATBOT_SOURCE_FFN_CONTROL_PROTOCOL_20261009.md):
2100s family/60s reserve,OS8GiB,GPU10/11GiB,outputs2GiB/no overlap. Actual long
FIT .16327s/label includingprefill prices8534 new forced rows at1393.8s;
20 capped generations add atmost1280 IDs plus their independentprefills.
The actual full-control family is1674.125s,all resource/input checksPASS,
quality7of10 gatesFAIL:DEVKL2.09967/42.9549% disagreement,screen1/16 andhistory0/4.
All observations/hashes are retained inits result/terminal receipt. The next
F32-only control costs75.829s andpreserves all16 original generated sequences,
withtwoFITKL<.0005. Select [local FFN recovery](CHATBOT_SOURCE_FFN_RECOVERY_NEXT_20261009.md)
before adding selection/core loss. Existing controls must be reused;new local
capture/calibration requires itsown immutable numerical/cost binding.

## Preconditions for a longer pipeline

Keep a useful measured intermediate at each stage;report loss and recovery.
Separate total distinct capacity from selected work and head/core/router cost.
Final packed target must pass NEW whole evolving-state C/own-history/tasks and
>=50 accepted IDs/s on that SAME artifact;no inherited8-update rate/admission.
The completed broad24 candidate remains available as the coverage/retention
control. A mixed old/new curriculum is an alternative controlled intervention;
it cannot silently become another repetition of the old286 recipe.
Useful n andgeneralization need additional distinct functions/data andactual
structured CPU winner/mass/DRAM tests. Neither this intermediate nor duplication
establishes useful100B capability.
