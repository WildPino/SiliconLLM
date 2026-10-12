# Source-head feature moments: complete numerical qualification

12 October 2026. **COMPLETE / INDEPENDENT_STORED_PASS. Goal ACTIVE/INCOMPLETE.**

## Reproducible component

Fixed actual Falcon-H1-1.5B source head `A` includes BF16 weights, final gamma
and LM multiplier. Teacher `q` comes from the captured raw BF16 hidden states
decoded into the F64 source operator, with source epsilon 1e-5. It is distinct
from a new source generation and from cached BF16-runtime probabilities.

Compiled sufficient statistics: `m=A^T q` and `H(q)`, for all 4422 FIT and
4386 DEV labels. For any changing source-feature decoder output `u`:

```
L = logsumexp(A u) - m^T u - H(q)
g_u = A^T softmax(A u) - m
y = mu + B z; r = sqrt(||y||^2/2048 + epsilon_source); u = y/r
g_y = g_u/r - y*(y^T g_u)/(2048*r^3)
```

This supports moving/conditional source-derived decoders without recompiling
teacher moments. Arbitrary readers outside this fixed head column space, changed
source heads/gamma and vocabulary bias variants require separate treatment.
Native F32 factor-product rounding is a separate export/probability gate.

## Full qualification

Three existing controls: original broad decoder, corrected alpha .0001 decoder,
and oracle source reconstruction. Every loss and all feature/RMS gradients are
stored; source-self loss/gradient are an additional control. No optimization.

| Mean case KL, F64 source operator | FIT | DEV |
|---|---:|---:|
| Original decoder | 1.02905035 | 2.96308892 |
| Corrected decoder | .99714032 | 2.93424563 |
| Oracle reconstruction | .01725408 | .04834649 |

These are control values, not chatbot quality admission. The oracle is not the
quantized native artifact.

Independent CPU audit checked all 18,038,784 moment words, 54,116,352 feature
gradient words, 54,116,352 decoder gradient words, 26,424 control rows and all
8808 source rows. Independent feature/vocabulary reductions and direct `(p-q)A`
gradients were used. RMS Jacobian actions were verified through independent
radial/tangent eigenspaces, not allocation of dense Jacobian matrices.
Maximum relative difference 5.42821e-12; scalar difference 0; all before/after
hashes exact. GPU internal comparison maxima remain bound observations rather
than exact CUDA trajectory replay.

## Cost and custody

Moment payload 144,310,272B versus 4,617,999,168B full F64 logits (~32x).
All eight outputs total 1,011,872,132B because full gradient controls are retained.
Producer terminal 211.45265s, OS peak 1,796,055,040B, CUDA allocated/reserved
1,384,143,360/1,407,188,992B. Audit terminal 332.15522s, OS peak 2,371,555,328B.
These are component resource observations, not clean/native throughput measures.
Both tool sessions closed exit0; no owned process remains live.

Source/protocol freeze `01a4ec1`; binding SHA256
`51cf7f03d5deab37a2b9a7e1ed8724953f8380e37b0ec9104847113cad154747`.
Audit source/protocol freeze `15438cd`; launch frozen in `61a6f08`.

- [Producer and controls](../../../benchmarks/native_expert_scaling/source_head_feature_moments.py)
- [Protocol](SOURCE_HEAD_FEATURE_MOMENTS_PROTOCOL_20261012.md), [binding](source_head_feature_moments_binding_20261012.json), [launch](source_head_feature_moments_launch_20261012.json)
- [Result](source_head_feature_moments_result_20261012.json), [terminal](source_head_feature_moments_result_20261012.terminal.json)
- [Independent audit](../../../benchmarks/native_expert_scaling/source_head_feature_moments_audit.py), [audit protocol](SOURCE_HEAD_FEATURE_MOMENTS_AUDIT_PROTOCOL_20261012.md), [audit launch](source_head_feature_moments_audit_launch_20261012.json)
- [Adjudication](source_head_feature_moments_adjudication_20261012.json), [audit terminal](source_head_feature_moments_adjudication_20261012.terminal.json), [closure](source_head_feature_moments_session_closure_20261012.json)

## Decision

Supervision for a changing source-derived decoder is available and verified.
No further unchanged emission-oracle optimization selected. Next isolate the
recurrent channel projection from the instantaneous skip contribution:
[selected diagnostic](RECURRENT_ONLY_CHANNEL_SKIP_NEXT_20261012.md).
Causal closure, native useful conditional functions, conversation preservation,
multiple families/scales and same useful artifact >=50 tokens/s remain open.
