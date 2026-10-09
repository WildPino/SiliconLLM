# Actual chatbot-target conversion at fixed eight-function work

9 October 2026. FIXED_WORK_FEASIBILITY_PASS. Goal ACTIVE/INCOMPLETE.
This makes parent-state conversion and actual whole-model updates available;
it does not admit a useful chatbot, native artifact or accepted50.

## Relation to the original engine

The destination remains the original engine's LUT/ternary/SSM machinery.
This experiment keeps the current compact target's eight active H128 functions
per site, replacing two selected functions with two always-consulted functions.
It preserves 18,874,368 expert products per token. It does not restore the
original smaller D256/L6 core, head or expert work: the current D512/L12 target
still has 8.891 times the original counted matrix products. Those are static
operation counts, not measured speed ratios. See the
[engine reconciliation](CHATBOT_ENGINE_ANCHOR_AUDIT_RESULT_20261009.md).

This result answers whether the proposed target-shaped conversion can actually
learn within local resources. The next comparison asks whether the allocation
helps transfer; useful chatbot quality and same-artifact native cost govern the
eventual pipeline. Stored parameter count alone does not answer either question.

## Provenance and actual sequence

[Worker](../../../benchmarks/native_expert_scaling/chatbot_fixed_work_feasibility.py),
[2+6 target](../../../benchmarks/native_expert_scaling/chatbot_hybrid_fixed_work_target.py),
[original protocol](CHATBOT_FIXED_WORK_FEASIBILITY_PROTOCOL_20261009.md) and
[repair protocol](CHATBOT_FIXED_WORK_FEASIBILITY_REPAIR1_PROTOCOL_20261009.md).
Initial code freeze 89fc91d3/binding 7fbc0e3e is retained UNEXECUTED: individual
Windows package junctions required resolved-package-root runtime binding.
Corrected first execution freeze 31816ace563cc16d53db5fe5eb52c9d8e4d4a4aa,
[34-input binding](chatbot_fixed_work_feasibility_binding_junctionfix1_20261009.json)
SHA 1651f639e6dc4613148c0732fb9ebcc087c1932a160bda919e7cf40f966d99b9.

First family adopts parent broad24/Adam318 exactly and computes both NEW initial
responses. First backward fails: checkpoint saved 984 tensors versus 983 on
recomputation. No Adam update completes. The common-output observation hook
ran with autograd enabled only in the original forward. Retain
[first fault](chatbot_fixed_work_feasibility_first_failure_20261009.json),
[launcher failure](chatbot_fixed_work_feasibility_result_20261009.launcher_failure.json),
[log](chatbot_fixed_work_feasibility_result_20261009.worker.log) and
[frozen original worker](chatbot_fixed_work_feasibility_worker_frozen_20261009.py.txt).
89.422 s family; 4,947,341,312 B held worker OS; 3,881,272,832 B CUDA allocated;
4,114,612,224 B reserved. Terminal exit 1/session 9819 closed.

Actual durable fault state0: 3,078,711,622 B,
SHA 38460a03093656e46ef5a86728db2e7c7fcc7ad3c1ce816909cd58e37074b6dd.
All 283 tensors, old 211 moment pairs at 318, new common slots absent, zero common
down/RNG/config/provenance. First parent adoption and state serialization are
retained; the interrupted gradient is discarded and its cost charged.

Repair freeze 1dc2590482ea53cfb34223783175cee357f3b608,
[50-input binding](chatbot_fixed_work_feasibility_binding_repair1_20261009.json)
SHA cc402a03f20880e9b3340f6daec3f581c6943bcad051730a6ac304ccc84a8b9d.
Only the observation hook changes functionally: torch.no_grad. Checkpoint
consistency verification stays enabled. Resume exact state0 weights/moments/RNG,
reuse both initial/baseline responses, complete two NEW longest FIT updates.
Both actual backwards now pass, supporting the observation-hook diagnosis.
Historical detailed initial routing exposure was lost in RAM; it stays missing.
New update/final exposure is retained, without replay to fill that gap.

[Actual result](chatbot_fixed_work_feasibility_result_repair1_20261009.json)
SHA a15e2ee77e09c8295cbeaad9bbb8ba3d24967ad72e92fab9f7ff3513f855063e,
[terminal receipt](chatbot_fixed_work_feasibility_result_repair1_20261009.terminal.json),
[log](chatbot_fixed_work_feasibility_result_repair1_20261009.worker.log).
Repair family 129.734 s; worker JSON 116.359 s/complete event 116.406 s.
All pre/post input and resource gates PASS, exit 0/session 11388 closed.

## What is now real

Actual source-informed compact state is converted to 2 always-consulted common
functions plus 6 selected private functions per site. D512/L12/10 SSM/2 SWA,
H128/private n72/V65537 and F32 controls remain. All expert forward weights
are ternary with AQ63; same whole-source KL/Adam/dither policy as parent.
This is the Python integer-dot deployment-form forward, not a new C LUT run.

259,669,760 F32 parameters/283 tensors. First adoption verifies all 211 original
weights and both moment arrays bitwise against broad24 state; repair verifies
all 283 restored weights and 211 moments against actual fault state. Old slots
advance 318->320; all 72 new common tensor slots 0->2. Every actual parameter
gradient and all model/moment entries are finite; core/private/common-down
gradient paths are positive in all 12 sites. New common gate/up/scale gradients
may initially be zero. No source model/reply, baseline forward, native or T4 call.

All NEW update/final positions select exactly 6 private functions and execute 2
common functions in each site; IDs/range/uniqueness/mass checked in the actual
forward. Largest final private-mass sum defect 2.3842e-7. Each long update exposes
1507*6=9042 private and 1507*2=3014 common function-positions per site; across
both updates 18,084/6028. These are exposures, not unique useful expertise.

Common down initial masters 0/scale 5e-5. After update 1: 1,570,365 nonzero ternary
coefficients; after update 2: 1,507,151 of 1,572,864 possible. Every site's combined
common output is nonzero on both final cases. Actual common-output mean-energy
fraction varies by site/case from .2340 to .9902. Always-consulted common functions
are not necessarily constant responses; neither this fraction nor code change
measures recovered knowledge or independently useful capacity.

## Whole-output FIT measurements and attribution limit

These two cases are already consumed FIT, 18/256 labels. Only the long case is
trained twice. Initial 2+6 has zero common output and a changed private mixture.
Baseline eight-private outputs are reused from actual Adam318, not rerun.
KL here is independent stable F64 computation from the complete V65537 logits.

| Case | Parent 0+8 KL | Initial 2+6 KL | Final 2+6 KL | Initial -> final ratio | Differing donor IDs before -> after |
|---|---:|---:|---:|---:|---:|
| Short rewrite, 58 input IDs/18 labels | 2.537745 | 2.525586 | 2.582894 | 1.022691 | 9->9 |
| Long magpie, 1507 input IDs/256 labels | 5.690210 | 5.685551 | 3.948761 | .694526 | 232->204 |

Before any update, k8->6 changes 1 short/6 long parent greedy IDs. Two whole-model
updates reduce trained long KL 30.55%, while short KL worsens 2.27%. Final long
204/256 disagreement remains high. All core/head/private/common parameters
were updated: the gain cannot be attributed to common functions without a
matched eight-private control. No DEV, fresh generation, history/task or C
measurement follows this table. It is not chatbot quality preservation.

## Actual cost, resources and durable resumption

New complete update intervals including gradient/state checks, before snapshot:
26.391 s/25.265 s, total 51.656 s. Final snapshot 42.282 s; repeated full snapshots
would materially increase training cost. Intervals include scientific guards;
they are not a stripped optimizer throughput or T4 forecast.
Repair CUDA allocated 5,069,031,936 B/reserved 5,836,374,016 B;
held worker OS 4,963,360,768 B/launcher 27,504,640 B.
Failed+successful family 219.156 s <= 300 s original allowance. Repair cap 210 s/
45 s reserve/OS 8 GiB/CUDA 10/11 GiB/namespace 4 GiB respected, no timing overlap.

Old namespace 11 files/3,150,639,124 B; new 12 files/3,188,673,535 B.
Combined retained off-repo bytes 6,339,312,659. Final actual state:
`results/native_expert_scaling/chatbot_fixed_work_feasibility_repair1_20261009/candidate.pt`,
3,116,677,466 B,
SHA d6bac2d054cc16d9f736ddf94264abc2051e70f13ee9b0b3e080289a931938b3.
Model 283/Adam 283/config/parent metadata/new history/start+final RNG are retained.
It is an offline master/optimizer checkpoint, not a packed engine artifact.

## Decision

Feasible target-shaped learning is established. Construction advantage is open.
Next [matched whole-model comparison](CHATBOT_FIXED_WORK_PAIRED_NEXT_20261009.md)
reuses this actual two-update state and prices a 0+8 control with the same new training dose,
then a fixed broad/domain-balanced continuation. Avoid selecting the construction
from a single trained-prefix gain or repeating initial observations. Variable-n
packed export/loading, useful n, structured CPU routing/mass/DRAM, useful fresh
chatbot/native >=50 and family/scale variants remain required.
