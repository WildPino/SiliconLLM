# Actual broader-data learner cost: long forward/backward, resource FAIL

9 October2026. Original collector fault and repair1 are terminal/FAILED.
Repair1 has TWO durable numerical/cost observations;GPU caps fail. No weight
update,full cohort before/after,equality audit,new export/native/T4 admission.

## Frozen identity and distinct faults

Original collector [fault](CHATBOT_BROAD_COST_REPAIR1_PROTOCOL_20261009.md)
occurs after a short forward/backward but before recording its costs. Lost
timings/gradients remain unavailable. Explicit module grouping repairs its
misnamed norm selector. Preparation tuple syntax fault is caught before repair
worker launch and corrected;no model run occurred with invalid source.

Repair freeze3b0317be4d1358df2d20747d84ddefed4280e1c4;
[28-input binding](chatbot_broad_cost_binding_repair1_20261009.json) SHA256
d6ceee0f96e2d6d911c8e100fbe9da8c28fc4a296b076deb4fa5dc340f0299e6;
[launcher failure](chatbot_broad_cost_result_repair1_20261009.launcher_failure.json)
and [worker log](chatbot_broad_cost_result_repair1_20261009.worker.log).
Actual records/firstfault/workspace extents are in
`results/native_expert_scaling/chatbot_broad_cost_repair1_20261009`.
Audited checkpoint286/Adam294/CPU+CUDA RNG restored,original target/SSD geometry,
training-only AQ dither.025;shortest/longest FIT selection uses metadata only.

## Measured operations and algebraic storage

| Case | TF IDs | Labels | Dithered training KL | Forward s | Backward s | With gradient inspection s |
|---|---:|---:|---:|---:|---:|---:|
| explore-instruct-rewriting_008 |58|18|10.0721674|4.140|4.844|9.922|
| smol-magpie-ultra_022 |1507|256 partial|9.9604397|9.281|28.578|39.469|

ALL211 parameter gradients finite/present;all12 core/bank/norm groups positive
finite on BOTH cases. No optimizer.step is called. These are two FIT training
loss/cost observations,not original-inference held-out metrics or chatbot quality.
The original unrecorded short forward/backward was re-executed for collector
repair;it is not an extra independent sample. Total attempted training calls
across both families include that original partial work,with0 optimizer steps.

Actual source SSD code for target C16,H48,d16,N256 constructs
`new_states=(decay_chunk[...,None,None]*states[:,:,None,...]).sum(dim=1)`.
With R=ceil(T/16)+1,the multiplication has R^2*H*d*N F32 elements. T1507/R96
gives7,247,757,312B for this single temporary;the intra-chunk C*B product is
1,195,376,640B. These are exact shape deductions,not measured allocation
attribution or an automatically valid sum of simultaneously live tensors.
Native engine.c computes recurrence with bounded state;this quadratic array
is an offline Torch implementation property,not a necessary model-state size.

## Resource failure and scope

Firstfault `AssertionError()` in GPU allocated guard at62.937s,AFTER both case
records were saved. CUDA allocator reports19,312,518,656B allocated versus
fixed11GiB=11,811,160,064B;reserved22,003,318,784B also exceeds fixed12GiB.
The first failing allocated assertion stops the family;caps are not raised.
On a12GiB Windows device,these allocator counters do NOT certify physical
VRAM residency. Physical memory traffic/residency/possible paging are unmeasured;
do not derive a physical bandwidth result from these figures.

Family69.609s/held worker OS15,543,529,472B<16GiB;launcher22948/worker8216
exit1/session90924 closed.6 outputs/21,308B. Time/output/OS are within limits,
but GPU gate FAIL. Final model/Adam tensor equality comparison is NOT reached;
it is not certified retroactively. No successful aggregate cost result/terminal
exists. Immutable checkpoint/input hashes are verified separately at session
closure. No source generation,optimizer update,native or T4 work.

## Next decision, without changing engine geometry

Stop this unmodified long Torch storage schedule. Do not replay either completed
cost observation merely to confirm the failure. Qualify a storage-only variant:
tile destination chunk index of the interchunk product,retain the full original
source-chunk reduction,concatenate outputs. Consider the other three independent
contractions already implemented for the donor. F32 target forward AND backward
require their own paired numerical/cost verification;BF16 donor bit equality
is not an inherited gradient certificate. Source code/parameter geometry/
ternary/LUT inference and archived failures remain unchanged.

Bind saved actual operands or a declared comparison with this changed storage
variable;freeze error/gradient/resource gates before observations. Reuse actual
58/1507 costs and8808 adopted teacher labels. After bounded storage is qualified,
freeze ONE finite broad-data pilot from Adam294 with all48 before/after and old
DEV retention/native/own-history gates;no silent full512 eligibility change.
Accepted50/useful n/routing mass/physical DRAM/other families/long context remain.
