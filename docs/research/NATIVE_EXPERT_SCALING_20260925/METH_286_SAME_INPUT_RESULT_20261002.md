# METH-286: identical-input operator localization

Frozen `57fb6eb` before observations; session82145 terminal exit0.
Same original276 archive/725 fields, unchanged285 model header. One
consumed source0/147 tokens/all24 layers, compact-core-only ablation.
All3528 isolated states retained; no weights/routes/fitting changes.

**Measured backend:** all24 actual calls use
`aten::_scaled_dot_product_attention_math`. Forced GPU MATH control is
byte exact to actual default. SDPA output is byte exact to the captured
o_proj input; grouped KV values match actual v projections.

| Isolated original CPU operator vs same-input GPU | Median relativeL2 | Maximum relativeL2 |
| --- | --- | --- |
| Input RMSNorm | 0 | .001457996 |
| q projection | 0 | .000900622 |
| k projection | 0 | .001423694 |
| v projection | 0 | .001673436 |
| RoPE q and k | 0 | 0 |
| Current F32 probability attention | 0 | .001640416 |
| Diagnostic BF16 probability attention | .002287753 | .004539893 |

Current attention differs on2136 coordinates; BF16 probability variant
differs on939422. Its prospective median/max halving indicator is FALSE;
do not adopt this variant. No isolated operator above establishes the
cause of accumulated285 whole-model5.4224% error. Flash-specific explanation
does not apply to the observed backend. RoPE is exact on these inputs only.

56.750s total; GPU52.750s, CPU3.984s after GPU completion. End reference
RSS2,577,928,192bytes/peakCUDA1,466,170,368bytes. All prospective resource
limits pass; no accepted-rate inference. [Raw result](meth286_same_input_diagnosis_result.json)
SHA `1c73b9fa032318f820de796b3e7743708846fcd4888bb414d4e75fd1750202bc`
contains every source/executable/fixture/reference/native hash and per-layer
metrics. [Protocol](METH_286_SAME_INPUT_PROTOCOL_20261002.md) gives command,
bindings, controls and limits.

Next: locate propagated layer-residual drift and qualify the remaining
o projection/post norm/source FFN on actual GPU inputs. Preserve285 failure;
no complete changed execution, quality/rate or useful large-n claim yet.
