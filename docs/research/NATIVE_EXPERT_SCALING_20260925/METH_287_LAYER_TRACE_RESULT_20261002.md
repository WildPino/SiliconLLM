# METH-287: small local differences accumulate through the residual stream

Code/protocol frozen `8c3cd42`; session92947 terminal exit0,59.781s.
Same276 archive/725 fields; consumed source0/all147 tokens/all24 layers,
compact-core-only arm (not donor), original285 equation plus diagnostic copies.
Removing copies restores original source exactly. Instrumented CPU final
hidden/full-head logits are BYTE-exact original285. GPU tail8 hidden is
BYTE-exact original285 reference. No apparatus repair or output selection.

## Measured same-input controls

| Operator | Median relativeL2 | Maximum relativeL2 | Unequal coordinates |
| --- | --- | --- | --- |
| o projection | 0 | .0010384355 | 1868 |
| Post-attention RMSNorm | 0 | .0020082286 | 355 |
| Source FFN FP32 | .000000469553 | .000000850805 | 2941231 |
| Source FFN BF16 return | 0 | .0005778873 | 1781 |
| Attention residual addition | 0 | 0 | 0 |
| FFN residual addition | 0 | 0 | 0 |

Source FFN FP32 remains close on actual prefix states, consistent with274's
different consumed6144 controls. BF16 return differs on1781 coordinates.
Both residual additions are exact given the SAME GPU operands. These local
results do not imply entire residual trajectories are exact.

## Measured propagated states

Whole CPU stream diverges after the first layer: layer1 input (zero-based)
median .00130220/max .00312611. Final residual median over all states/layers
.0115890/max .0673500; propagated source FFN FP32 median .0184962/max
.0858270. Layer20 input reaches maximum .0673500. Per-layer/raw details
are retained, not reduced to the last8 smoke positions. Small differences
in preceding operands are amplified through subsequent operations. No one
operator or corrected equation is proven responsible for the whole error.

Same-input norm reduction is a remaining precision hypothesis: CPU uses a
sequential FP32 sum of896 squares; GPU uses a reduction. A prospective
changed reduction can test whether more accurate accumulation reduces whole
drift, preserving BF16 boundaries, weights/routes and unchanged285 limits.
It must not be presented as a repair before the complete test passes.

GPU52.891s, subsequent CPU6.875s, end referenceRSS3,259,920,384bytes,
peakCUDA1,468,556,288bytes. All prospective resource stops pass, no rate.
[Raw result](meth287_layer_trace_result.json) SHA
`32adcf57d582270510bd689726b1e34c217310a213f19635a5f15056e99f1650`
contains all file/code/executable hashes,controls and per-layer arrays.
[Protocol](METH_287_LAYER_TRACE_PROTOCOL_20261002.md) supplies reproduction.
Original285 failure remains stopped; no native quality/rate qualification,
new useful capacity, large-n DRAM/LUT or family/10B/100B claim.
