# METH-276: diagnostic complete I16/private128 composition

## New uncertainty and scope decision

259's complete source32 model fails semantic267;269 implicates source
features and268 shows induced routing.271's source128 function improves
mean/reserve error22.4%/21.9%;274 qualifies all6144 actual CPU source/reserve,
quantizer/integer/numerical controls.274/275 native cost gates fail. No
complete model with the changed function exists,so composition/new complete
quality and actual whole-model cost are unknown. More isolated scheduling
tweaks have not resolved that uncertainty.

Build one explicitly **diagnostic-only**,unpromoted complete archive/reference
with274 I16 arithmetic and271 source128 rows. This is a new composition
experiment and an explicit change to the former assembly-order assumption.
The10ms component allocation remains failed; it is not the user-provided
whole-model50tok/s gate and cannot prove total native feasibility either way.
Do not accept a sum of historical component timings as whole performance.
No fixed stopped kernel is reclassified. Subsequent full native promotion
requires prospective fresh complete quality plus actual same-artifact
accepted>=50/lowerCI95 on declared hardware/context; all useful n/RAM/
route-LUT-real DRAM/family10B/100B requirements remain unchanged.

## Bindings and exact changed fields

Use original259 complete archive SHA
`3c0949fb2b7f99dc887c6aec1041aa162b35801939349966d18e52b8807a6f71`,
its export/helper hashes;271 source fixture SHA
`ece28ea3344d9ecef674d30b4c43b067ee4290379cf9130c97c2eafee1e344d8`,
274 raw SHA `aa2acf6452031d621568cb4712c1a976587755c398af2dc03274ac844cd451cc`,
275 raw SHA `4d2b9bcb69c8b1d39d1d6a7c1090b212b129373b11322791edd376c3eea9a0c9`,
125vectors and actual6144 CPU qualification output/quantizer hashes.
Preserve all725 logical fields/names. Only72 private IDs/gate/up tensors
change from32 to128; all653 other fields must retain original259 raw tensor
hashes,including all banks/router/keys/leaf maps/non-FFN/head proposal/table.
Additional payload8,262,144bytes,total1,329,221,892bytes (metadata/header
extra measured separately). Do not save the rejected275 physical pack in
this logical archive; native I16 arithmetic remains274's reference.

Version format distinctly,record input arithmetic/private128/frozen helper
hashes and `diagnostic_only=true`,`native_promotion_qualified=false` in
metadata. Count30,556 original unique functions/164 aliases unchanged; no
new learned capacity,n utility or donor-size compression claim. Snapshot
old config/source provenance; source/checkpoints are export controls only.

## Apparatus and prospective archive gates

New source `meth276_diagnostic_complete_core.py`,artifact
`results/native_expert_scaling/meth276_qwen05b_i16_private128_diagnostic.safetensors`.
Implement own versioned standalone loader; do not mutate frozen259 helpers
or their format/global values. From archived config,construct model/remove
all72 random FFN parameters before execution,copy218 archived non-FFN
parameters,tie embedding/head,consume all725 fields,validate all bank map
counts/ranges/surjectivity. No donor/conditional checkpoint fallback.

Base FFN flattens BF16 x,uses274's exact documented quantizer/FP32 GPU
reference integer-code dots/private128/LUT/readout,returns BF16. Existing
MappedConditional algorithm/parameters remain unchanged. Reference model
uses full-prefix/no-cache arithmetic; failed264 CUDA cache recipe stays
closed. It is a semantic reference,not production throughput.

Require metadata/field-count/name/shape/dtype/finite/hash/readback exact,
all653 unchanged fields byte exact,all72 new private fields exact271 fixture
and old32 retained. Standalone loaded base must reproduce the same274 GPU
equation on all6144 consumed125 states; all6144 C numeric limits retain
median<=1e-4/max<=5e-4 versus the archived equation. Same-input parent IDs,
scores,child IDs,selected A/B/aliases and conditional contribution exact259
on all6144 controls. Full prediction/generation/task/semantic preservation
cannot be inferred from these component checks.

Local3060/sixthreads10min export/readback/loader/component controls,20GiB
RSS/10.5GiB GPU,3GiB output,no T4/download. Refuse existing artifact/result,
preserve all partial/failure/code/helper/source/shape/hash/resource evidence.
Freeze code before any276 observation. At this protocol freeze no276 code,
artifact or score exists. Pass licenses ONLY separately frozen consumed
whole-model regression; freeze new excluded-source prediction/generation/
tasks/anonymous-semantic checks before independent scoring. Do not invoke
complete native rate/promotion from this archive-consistency experiment.

## Apparatus freeze

The new loader declares format
`M276_DIAGNOSTIC_I16_INPUT_PRIVATE128_UNIQUE_BF16_V1`, checks its own source
and imported project helper hashes, and validates the complete tensor
inventory before model construction. M274's FP32 equation is exposed
separately from the BF16 return: native numeric gates compare FP32 against
FP32, then require the returned BF16 value to equal its exact BF16 rounding.
Conditional parity isolates the original BF16 branch before adding the
changed dense output, avoiding a subtraction contaminated by BF16 rounding.
All 725 fields and per-layer 256-state errors are retained in the result.
The old259 model is loaded solely as the fixed same-input conditional
control, not as a fallback for the diagnostic loader. No model inference or
276 archive construction has occurred before this apparatus freeze.

Command (repository root; existing outputs are refused):

```powershell
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth276_diagnostic_complete_core.py --artifact results/native_expert_scaling/meth276_qwen05b_i16_private128_diagnostic.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth276_diagnostic_complete_core_result.json
```
