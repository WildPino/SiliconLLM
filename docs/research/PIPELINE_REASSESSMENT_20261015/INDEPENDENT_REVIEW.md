# Independent strategy review: native MoE transfer and the large-model gate

15 October 2026. Read-only advisory review requested by the owner, followed by
the owner's [scope clarification](../NATIVE_EXPERT_SCALING_20260925/SCOPE_CLARIFICATION_20261015.md).
This records the substantive recommendation and the resulting research decision;
it is not a model result or a new frozen numerical experiment.

## Diagnosis

The current source-preserving Falcon path has too much unconditional work:
[encoded generators plus readouts](../NATIVE_EXPERT_SCALING_20260925/ENCODED_ORGANS_WAIT_RESULT_20261014.md)
already take22.44–24.13ms before remaining organs. The narrow target has the
opposite defect: [balanced replay](../NATIVE_EXPERT_SCALING_20260925/CHATBOT_BALANCED_REPLAY_RESULT_20261015.md)
learns repeated FIT but fails new DEV. Its changes include width, depth,
precision and routing; insufficient examples are not the only possible cause.
Neither another unchanged fit nor a minor kernel improvement resolves both.

Native MoE family-first is a valid path. Dense70B conversion and systematic
attention-to-SSM replacement are not prerequisites under the corrected scope.
However, a small native MoE is only an enabling pilot; it does not establish
retained capacity and speed at70–100B.

## Proposed transfer mechanism

Partition each pretrained SwiGLU expert along its intermediate dimension into
disjoint child functions. With every child active, their sum reconstructs the
parent in real arithmetic; floating-point ordering still needs qualification.
Retain the source parent router and its mass. Each all-on child initially
inherits the parent's contribution mass: a sum-to-one normalization among
children would otherwise attenuate the parent unless compensated explicitly.

Then learn input-driven child selection and contribution scales under a fixed
selected-byte allowance, initially freezing the pretrained bank weights. Child
selection must use the candidate's current state. Computing every source
gate/up value before selecting is an oracle and cannot be priced as a cheap
route. All original coefficients remain in the pool; causal useful capacity
must still be demonstrated after omissions and quantization.

[Granular upcycling](https://arxiv.org/html/2410.07524) supports the importance of
initialization and scaling, but its substantial continuation-token budgets are
not evidence of an economical local conversion. The previous local predictor
and residual-bank failures remain counterevidence within their tested scopes.

## Suggested small pilot, not selected as the final family solution

The review proposes `ibm-granite/granite-3.1-1b-a400m-instruct` as an engineering
and conditional-transfer pilot. It reports official revision
`0da7a48b0276d500ce5922fd2b33944091fc6c09`,1,334,628,352 BF16 parameters,
24 layers/width1024/FFN512/32 parents/top8 and vocabulary49155. These are
reviewed official metadata claims; this repository has not yet acquired and
audited its complete instruct tensor headers or values. The earlier
[M411](../NATIVE_EXPERT_SCALING_20260925/METH_411_GRANITE_MOE_SOURCE_HEADERS_RESULT_20261004.md)
header result is for BASE, whose vocabulary49152 differs.

Reuse the complete-shape synthetic
[M412 apparatus](../NATIVE_EXPERT_SCALING_20260925/METH_412_GRANITE_I8_COST_RESULT_20261004.md)
and the existing
[Q4_K/Q8_K helper](../../../benchmarks/phase60/strat01_q4k_q8k.h), rather than
claiming that the failed row-I8 route rules out Q4 experts. A proposed single
policy uses Q4_K experts, I8/A16 attention/head and source-preserving router
semantics. The initial four128-coordinate proposal conflicts with the existing
Q4_K/Q8_K kernel's256-coordinate block. The first implementation therefore splits
each parent512 into TWO256-coordinate children, selecting one per source-selected
parent and learning contribution scales. Both children form the initialization
control. Quantized all-child and sparse-child controls must be compared
separately with the donor.

For width1024, a parent's gate/up/down Q4_K payload is884,736 bytes; a contiguous
child256 payload is442,368 bytes. Selecting two128 children with down inputs
padded to256 instead costs589,824 bytes, and enlarges the stored parent pool.
Child-major packing must copy original quantized blocks without requantization.
The default helper dispatch is scalar; active AVX2 requires the explicit
`STRAT01_Q4K_Q8K_DIAGNOSTIC_ACTIVE_AVX2` build definition. Payload counts are
addressed bytes, not measured DRAM traffic. Floating-point association and Q8
activation segment equality require independent controls before timing.

The review estimates299,541,516 addressed bytes/token for the all-parent
mixed policy before cache work; this is config arithmetic, not a complete
header audit, executable export or physical DRAM measurement. Full F16 KV
reads at context2048 would add100,663,296 logical bytes/token. Corpus texts and
split groups can be reused, but Falcon logits/output IDs are not Granite labels.

The review proposes a maximum4h pilot/2h GPU after preparation, RAM16GiB and
GPU9.5GiB, starting with a bounded resource probe. These are proposed budgets,
not a launched job or a measured implementation ETA. The implementation and
complete donor/operator/numerical contracts remain to be built and frozen.
Quality, own generation and >=50 accepted batch-one CPU tokens/s must pass on
the same export. No checkpoint or policy may be selected using DEV.

## Why the large-model gate comes first

[M397](../NATIVE_EXPERT_SCALING_20260925/METH_397_QWEN_NEXT_SOURCE_HEADERS_RESULT_20261004.md)
counts3,563,766,528 main active parameter positions for Qwen3-Next80B. Routed
top10 expert matrices account for48*10*3*2048*512=1,509,949,440 positions.
Removing **all** routed expert work still leaves2,053,817,088 other positions.
Four ideal bits per remaining position imply1,026,908,544 bytes, or25.673ms
at an assumed40GB/s before scales/cache/compute. This is a deliberately
favorable analytical screen, not actual memory-controller traffic or an
impossibility theorem. It nevertheless rejects an FFN-only remedy for that
unchanged backbone/head/router representation.

Therefore a small Granite success would not justify a70–100B claim. The method
must either select a large family with a sufficiently compact always-on path,
or state and validate another transformation of that path. Expert count,
metadata forecasts, shared names and synthetic timing alone do not prove the
required scaling. Full dense Adam state at16 bytes/parameter would cost1.12TB
for70B, before activations. Streamed conversion and mainly frozen source
banks are essential resource hypotheses, not a guarantee that fitting is cheap.

## Decision after review

Accept native MoE transfer and exact parent-preserving child initialization as
plausible method ingredients. Accept the small Granite case only as a possible
bounded engineering/scientific pilot. Do not automatically acquire its weights,
train selectors or treat its kernel result as the large-family pipeline.

Before a substantial new pilot, identify the intended larger family member,
separate its protected always-on cost from reducible expert cost, and name the
transformation that addresses each exceeded allowance. If no large-model path
is supported yet, say so. The current capture closes as reusable Falcon
supervision and does not compel continuation of the failed narrow target.

Evidence that would reverse the child-selection proposal includes a need for
almost every child at the quality gate, masses that destabilize own generation,
or routing/packing costs that erase the selected-byte savings. Evidence that
would strengthen it is a held-out causal benefit from the retained bank at a
fixed active budget, stable whole-artifact CPU performance and bounded
conversion state, followed by an actual larger family member passing the same
admission process.
