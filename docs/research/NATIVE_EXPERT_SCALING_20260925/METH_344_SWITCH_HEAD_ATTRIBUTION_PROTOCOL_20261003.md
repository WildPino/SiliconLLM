# METH-344: fixed-state head attribution after343 top1 failure

Prospective diagnostic AFTER committed343 failure, BEFORE counterfactual
precision observations.343 raw SHA
`a4838903958be5f6bb567c1dedd802a17e6b27e2d97c4f490210139f296cbadf`;
all24 books/96 cases complete39min.7/8 gates PASS, original top1 agreement
1001/1056=94.7917% FAIL95%.55 differences,52 on eight-mask-token positions.
No unchanged W8A8 quality/generation/rate promotion. Actual masked-token
accuracy336/768 original versus337 native; both whole8-token exact0/96.

Uncertainty: which head precision variable can address that failure, and how
much discrepancy remains from upstream states? Reuse ALL96 saved original
and actual native final-normalized states/logits, exact343 NPZ/bin hashes.
These sources are now consumed diagnostics; no source/candidate selection
or inferential quality gate may be relabeled untouched from this experiment.

## Identities and oracles

Fresh entire338 payload hash; verify actual original F32 shared lookup bytes
match327, actual I8 head codes/scales match338. Original tied head is exactly
that F32 lookup, no58GB source archive/full teacher load needed. Fixed original
and native final-normalized vectors scaled by SAME F32 D^-0.5. CPU1 original
Torch Float32 linear per vector must reconstruct saved original logits with
relative L2<=1e-6 and exact choices BEFORE counterfactual interpretation.
Actual serialized I8/I64/F64-scaling reference on native vectors must reproduce
native343 head logits BYTE EXACT. Fail oracle before any experiment conclusion.

A16 reference uses F32 absmax/32767 scale (zero1), F32 quotient/nearest-even,
clip[-32767,32767], I64 dots with serialized I8 weights, F64 scales product
then F32. Scalar controls cols8/768/4096 with ties/cancellation/extremes;
global4096 maximum17,045,131,264. No I32-overflow/saturating intermediate
assumption. This is only a mathematical reference, not an implemented native
A16 kernel or whole-model activation-precision intervention.

## All fixed counterfactuals, chosen before observation

On ORIGINAL states: same I8 head with A8, A16, or unquantized F32 activation;
source F32 weights/F64 dot/F32 return as numerical floor. On ACTUAL NATIVE
states: I8+A16, I8+unquantized activation, source F32/F64 dot. Weight-only
uses F64 dot of actual I8 codes with scaled F32 input, then F64 row scale/F32
return. All counterfactual logits saved with hashes.

Record per-case/aggregate original-top1 agreement, native disagreements
recovered AND new disagreements introduced, all-token/masked-token NLL delta
and logit relative error versus actual original; native/original normalized
state error and native vector maxabs. Descriptive fixed-cohort means only.
No bootstrap/hypothesis decision gate on precision repair from these now
consumed states. In particular HEAD-only A16 does not measure full upstream
A16, and source F32 at native states does not restore original upstream state.

All96 original-head/native-head oracle/primitive gates mandatory; CPU1,
MAIN20min after imports/RSS8GiB, no full-model/GPU/download/training/timing
overlap. Freeze code/protocol before observations; preserve failures/arrays.
PASS provides attribution only for separately frozen smallest new variable,
independent native arithmetic/actual cost and NEW whole quality cohort.
343 original95% criterion remains failed; no postcount threshold change.
Free generation C draft exists UNFROZEN/UNEXECUTED;343 failure prevents its
promotion work. No accepted-rate/useful larger-n/LUT/DRAM/family conclusion.

Command:
```
results\native_expert_scaling\meth324_switch_reference\venv\Scripts\python.exe benchmarks/native_expert_scaling/meth344_switch_head_attribution.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth344_switch_head_attribution_result.json
```
