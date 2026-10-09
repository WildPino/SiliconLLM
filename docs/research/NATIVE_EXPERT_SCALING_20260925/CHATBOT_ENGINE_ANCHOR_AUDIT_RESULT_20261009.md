# Re-anchor the converter to the original engine

9 October 2026. Goal ACTIVE/INCOMPLETE. Human direction correction and static
audit, not a new model or performance result. This decision supersedes the
additional source-site0 dose as the operational priority.

## What the audit actually establishes

[Tool](../../../benchmarks/native_expert_scaling/chatbot_engine_anchor_audit.py)
and [protocol](CHATBOT_ENGINE_ANCHOR_AUDIT_PROTOCOL_20261009.md) frozen at
`c8d3473ee094d7222cdf21fd478164d9404d751c` before execution. Command exit0,
0.016s in-script before serialization. [Exact result](chatbot_engine_anchor_audit_20261009.json)
11,390B, SHA `42f1b3af38f151893adcf03501705ed1b06a82ce553760ff320fae358d65df09`.
All ten input-file identities unchanged. No tensor library, source model,
training, compiler, native timing or T4 call. Memory/through-exit process-family
peaks were not measured for this standard-library static calculation.

The current target shares original matrix/LUT/AQ primitives, but differs in
residual width, depth, recurrence, activation and vocabulary. A compile branch
inside engine.c and unchanged kernel bodies do not establish equal operators,
active cost or quality. The current scan shares parameters per head; the
original scan has channel/state-specific A and different input projections.
Their respective matrix counts were derived from their own code.

| Quantity | Original default E32 | Current compact Falcon E72 | Ratio |
|---|---:|---:|---:|
| Residual width / sites | 256 / 6 | 512 / 12 | 2 / 2 |
| Core matrix products per token | 2,801,664 | 16,760,832 | 5.982 |
| Selected expert products per token | 4,718,592 | 18,874,368 | 4 |
| Full head products per token | 262,144 | 33,554,944 | 128.002 |
| Flat router products per token | 49,152 | 442,368 | 9 |
| Sum of counted matrix products | 7,831,552 | 69,632,512 | 8.891 |
| Useful persistent state payload, bytes | 1,286,144 | 9,117,696 | 7.089 |

These are integer shape deductions, not latency multipliers. State payload
counts the active SSM/SWA state; the original allocator also reserves unused
SSM slots for its SWA site. Neither column is total workspace. Nonlinearities,
state-update arithmetic, normalization, selection, interface and prefill cost
are excluded from matrix-product counts.

The original measured701.7/s remains a real small-model result. The current
[actual engine probe](CHATBOT_HYBRID_ENGINE_PROBE_RESULT_20261009.md) measured
46-59 raw one-core decode IDs/s with degenerate replies. Neither proves useful
large-donor conversion with n-independent complete latency.

## Stored capacity was compressed as well as active computation

Original Falcon source FFNs contain679,477,248 coefficients. The current
conditional bank contains169,869,312:one quarter as many. Width/depth/state
compression and sparse consultation were imposed together. This is not a
measurement of retained information or a proof that more coefficients solve it.

For fixed D512/L12/h128/k8, capacity accounting gives:

| Prospective bank | n per private layer | Bank coefficients | Packed model bytes with current flat router | Flat router products/token |
|---|---:|---:|---:|---:|
| Current | 72 | 169,869,312 | 425,210,736 | 442,368 |
| Source FFN coefficient count | 288 | 679,477,248 | 693,296,112 | 1,769,472 |
| Twice source FFN coefficient count | 576 | 1,358,954,496 | 1,050,743,280 | 3,538,944 |
| Approximately10B bank | 4,239 | 10,001,055,744 | 5,597,024,448 | 26,044,416 |
| Approximately100B bank | 42,386 | 100,001,120,256 | 52,942,639,440 | 260,419,584 |

Selected expert products remain18,874,368 in every row. Model-byte projections
reuse the observed packed payload's fixed part, tensor-table overhead and
current F32 flat router/scales; they are not implemented new models or measured
DRAM traffic. Equal coefficient counts do not establish equal useful capacity.
Additional branches must become different useful functions; duplicates alone
do not populate the capacity dial.

Two concrete implementation gaps precede RAM-driven scaling:

1. Current compact C has fixed E72 and a512MiB blob cap. Every enlarged profile
   is rejected by its current header contract; all listed n>=288 also exceed
   its allocation cap. Original default C reads E dynamically but retains F32
   and unpacked expert copies:5.5 bytes/coefficient before scales/control data,
   versus0.5 for byte-pair codes alone. Packed-only capacity is required.
2. A flat router costs LDn=P_bank/(3h) products. At about100B it costs13.8 times
   the selected experts' matrix products. A structured learned selector needs
   its own CPU cost, declared exact reference, selected-ID/mass and quality
   evidence. Selected-only normalization can avoid the global denominator;
   it does not remove the exhaustive score scan.

Training memory is another separate constraint: F32 bank parameters, gradients
and two Adam moments alone require160GB at10B and1.60TB at100B. Long T4 time
cannot solve that allocation. A large-bank recipe must bound resident parameter/
optimizer blocks, retain dormant state on host/storage and price actual transfers,
activation memory and teacher supervision. These are missing pipeline stages.

## Correction to the scientific dependency

The source-site0/site23 arithmetic and recovery failures remain valid, useful
diagnoses of those recipes. Their local<=.10/cosine>=.99/centered<=.10 criteria
were preregistered for faithful local functions. They are not necessary
conditions for a jointly adapted chatbot with changed internal representations.
There is no general implication from large local error to unacceptable final
quality, or from small local error to useful generation after composition.

Requiring every large donor FFN to pass those gates before learning conditional
functions would select a narrower problem than the goal. Stop that operational
dependency. Reuse captured source operands/labels and retained states when a
specific diagnostic can change a target-shaped conversion decision.

## Selected direction and exact next action

[Fixed-work conditional conversion](CHATBOT_ENGINE_FIXED_WORK_NEXT_20261009.md)
is the next construction: test shared/conditional functions inside the current
eight-function active allowance, train against whole source chatbot outputs,
then qualify the actual exported candidate. Its whole quality and native cost
determine continuation. Capacity growth must later use a versioned variable-n
packed contract and bounded optimizer storage. Do not reduce the goal to a
small student or add stored copies as a substitute for useful functions.

The site0 dose worker is retained as a syntax-checked, unexecuted draft; its
binding/launcher integration and optimizer/native-state adoption remain
unvalidated. No dose result or new checkpoint exists. Month-plus T4 adaptation
remains allowed after a concrete recipe, measured feasibility, budget and stops.

## Exact command

```powershell
& 'C:/Users/giosa/AppData/Local/Programs/Python/Python312/python.exe' -I -S -B -X utf8 benchmarks/native_expert_scaling/chatbot_engine_anchor_audit.py --freeze c8d3473ee094d7222cdf21fd478164d9404d751c --out docs/research/NATIVE_EXPERT_SCALING_20260925/chatbot_engine_anchor_audit_20261009.json
```

The output path is exclusive. The result already exists; reuse it instead of
repeating the command. Original engine.c SHA remains
`5f948fc0dcd28b1647a2a2c73067bdc0a76a7d41ae7f34395ede8120aeaa83ce`.
