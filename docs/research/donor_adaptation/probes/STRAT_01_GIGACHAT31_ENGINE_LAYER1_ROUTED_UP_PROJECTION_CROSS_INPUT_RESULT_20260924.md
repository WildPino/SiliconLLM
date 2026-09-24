# STRAT-01 layer-1 routed-up projection cross-input result

**Status:** `LAYER1_ROUTED_UPSTREAM_NORM_RESIDUAL_SUFFICIENT`

## Scope and provenance

This result executes the frozen [protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_UP_PROJECTION_CROSS_INPUT_PROTOCOL_20260924.md)
with the model-free-qualified [apparatus](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_UP_PROJECTION_CROSS_INPUT_APPARATUS_RESULT_20260924.md).
The accepted artifact was opened and validated: 6,474,702,976 bytes,
SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
The invocation ran one diagnostic, computed three up arms, propagated five
Q6 arms, and executed zero donor/reference graphs.

| identity | value |
|---|---|
| repository HEAD | `a7e1f9d12d90c04f3cbfb3ff6e635ac0c357c126` |
| adjudication SHA-256 | `2de61cdc02deeb9639044cb75cc614705a41eb3e133b00f3ad2746b80bc7c71d` |
| executable SHA-256 | `905591dafa6659c28b1cffacf420a7f825b609f2925ec2b5f42bc8b98920587a` |
| raw result directory (unversioned) | `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_routed_up_projection_cross_input_20260924/` |

## Findings

| arm | downstream NRMSE | normalized maximum | gate |
|---|---:|---:|---|
| captured reference | `0` | `0` | PASS |
| captured production C | `0.002642086799405445` | `0.004687597394884091` | FAIL |
| current Q4_K up on reference `ffn_norm-1` | `0` | `0` | PASS |
| current Q4_K up on C `ffn_norm-1` | `0.002642086799405445` | `0.004687597394884091` | FAIL; exact C replay |
| token-7-negated reference-input control | `0.20962964201860176` | `0.3668577386863162` | FAIL as required |

The current up projection on the exact reference normalized input is
byte-identical to the reference at every emitted stage. On the C normalized
input, it is byte-identical to captured production C at every emitted stage.
Thus the selected current Q4_K up matrices pass when given the reference
`ffn_norm-1`, while their C-input path reproduces the complete downstream
failure exactly.

For the direct current-C chain, the stage NRMSE / normalized maximum values
are: up `1.6474086443438292e-7 / 1.1459829065728941e-7`; SwiGLU
`1.2415877785574164e-7 / 4.3461239180759115e-8`; down
`7.951168981266576e-5 / 9.274169798632739e-5`; routed output
`1.354682443199535e-5 / 1.1138183142366114e-5`; and downstream
`0.002642086799405445 / 0.004687597394884091`.

All pinned-input mutations were refused, the label swap was rejected, and
schedule twins were byte-exact. Artifact and binary identities matched the
frozen run. The planted control fails by design. No donor or reference graph
was run.

## Adjudication and next boundary

Close the selected Q4_K up projection and every downstream coordinate covered
by this chain. The active boundary is upstream of the up input, at production
of `ffn_norm-1`. This result does not determine whether the cause is the
`ffn_inp-1` payload or the production RMSNorm that maps it to `ffn_norm-1`.

The frozen
[layer-1 FFN RMSNorm protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_FFN_RMSNORM_CROSS_INPUT_PROTOCOL_20260924.md)
tests that specific split using immutable inputs and the same downstream
amplifier. Do not infer either component as causal from this result alone. No
quality, speed, or full-model claim follows.
