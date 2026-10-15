# Conditional source attention contributions: one bounded falsifier

15 October 2026. Decision after the [protected-core timing rejection](PROTECTED_CORE_120B_CPU_RESULT_20261015.md)
and [actual first-block rank-256 rejection](FIRST_ATTENTION_RANK_RESULT_20261015.md).
No deployable representation, selector, full-model quality or rate is established.

## Why this changes the representation

Do not extend the uniform Q/O rank fit or assume that a nominally hybrid donor
removes the shared-core cost. One remaining cheap question uses the source
weights already acquired: can a token's branch be reconstructed from one small
source attention contribution, with the chosen contribution varying by token?
Unlike a fixed low-rank map, this preserves all source Q/O banks as potential
conditional capacity. K/V must still be updated for every source KV head so
that a future selection sees correct history.

Partition 64 query heads into 16 contiguous groups of four. Each group spans
256 O-input columns, matching the existing Q4_K block width. For token t let
`h[t,g]` be that group's source contribution after O, without O bias, and let
`target[t] = source_branch[t] - O_bias`. Select one group and scalar mass.

The primary falsifier deliberately relaxes the proposed bounded positive mass:
allow **any real scalar**, using `a = dot(h,target)/dot(h,h)` and a=0 for a
zero vector. Evaluate the exact residual for all 16 groups in F64 and choose
the minimum, breaking ties by the lowest group ID. This optimistic class
contains every nonnegative or [0,2]-clipped one-group rule. Failure therefore
cannot be repaired merely by improving a selector within that class. Negative
or large optimal masses remain diagnostics, not evidence of a stable deployable
router. No number-of-groups grid is selected after viewing DEV.

Use the same 36 cases and variable-query mask as the previous experiment.
Reconstructed all-group output must first match the sealed full-F32 source
trace and pass the branch precision gates against actual BF16 source output.
Then require one-group pooled branch L2<=1% and sequence L2 p95<=5% separately
on FIT and DEV. The oracle sees the target on every token, including DEV: it
does not establish learned holdout quality. A TV gate over all 64 query heads
has no equivalent semantics when only four heads are present; this is a new
branch-function representation, not a revision of the completed rank gates.

This diagnostic computes every source group before selection. It has no cheap
ID/mass predictor and no CPU speed claim. A pass permits only a subsequent
predictability question; a fail closes this one-group source-bank variant.
It does not rule out arbitrary transformed banks or every native MoE family.

## Complete cost remains the controlling constraint

For 36 source layers at width 2880, keeping all K/V and one four-head Q/O group
addresses the following projection payload. These are exact representation
counts for the proposed geometry, not measured DRAM or latency.

| Item | Coefficients | Physical bytes under the screened Q4_K/F32 formats |
| --- | ---: | ---: |
| All K/V | 106,168,320 | 63,700,992 |
| Selected Q | 26,542,080 | 15,925,248 |
| Selected O | 26,542,080 | 14,929,920 |
| Existing F32 expert routers | 13,271,040 | 53,084,160 |
| Total projection/router payload | 172,523,520 | 147,640,320 |

Q/K/V rows include 2880->3072 padding; O group width is exactly256. Norms,
biases, Q8 packing/scales, head-group selector and masses, attention/KV work,
head retrieval and expert work still need pricing. All eight KV heads remain
stored and updated. If only the chosen query group reads history, it needs
one of eight KV heads: at context2048 the optimistic F16 K+V history reads are
10,027,008 bytes across 18 full and 18 window128 layers, versus80,216,064 for
all eight. A selector that scans the other heads loses that saving. No actual
cache/DRAM result is implied by either logical count.

Even this aggressive conditional core does not solve the expert path. Original
top4 experts address3,583,180,800 active coefficients, or1,791,590,400 bytes at
ideal4bit before format overhead. A credible20ms plan needs a very large further
reduction there, with useful IDs and masses, while reserving time for the full
core/head/state/frontend. No proportional scaling of previous timings is admitted.

## Literature and alternative-family limits

[MoH](https://arxiv.org/html/2410.11842v3) supports the general idea of selecting
attention heads, but its LLaMA3-8B adaptation uses75% of heads after400B continued
training tokens. That does not validate retaining6.25% cheaply. The present
oracle is deliberately a fast falsifier of a much more aggressive hypothesis.

The existing [Qwen3-Next source-header ledger](../NATIVE_EXPERT_SCALING_20260925/METH_397_QWEN_NEXT_SOURCE_HEADERS_RESULT_20261004.md)
already records3,563,766,528 active parameters, including1,509,949,440 routed
expert coefficients. The remaining2,053,817,088 include nonrouted projections,
shared FFN, vocabulary and controls. Linear attention reduces history growth;
it does not remove these coefficients. Skipping a recurrent head's updates
also changes its future state. No unchanged Qwen3-Next acquisition follows.

The next numerical action is this single CPU-only oracle using retained
weights/traces, with code and arithmetic frozen before outcomes. No optimizer,
new labels, GPU pass or source download is needed. If it fails, park the two
tested cheap GPT-OSS core transformations rather than repeatedly fitting them.
A subsequent pipeline proposal must introduce a distinct mechanism or a
pretrained family whose small shared core and much finer conditional activity
survive a quantitative full-cost screen. The overall goal remains incomplete.
