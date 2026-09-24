# STRAT-01 layer-1 FFN RMSNorm cross-input result

**Status:** `LAYER1_FFN_INPUT_RESIDUAL_SUFFICIENT`

## Scope and provenance

The sole scientific invocation authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_FFN_RMSNORM_CROSS_INPUT_PROTOCOL_20260924.md)
and canonical repair4
[apparatus](STRAT_01_GIGACHAT31_ENGINE_LAYER1_FFN_RMSNORM_CROSS_INPUT_APPARATUS_RESULT_20260924.md)
completed without error. It opened and validated the accepted 6,474,702,976-byte
GGUF, SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.

| identity | value |
|---|---|
| repository HEAD | `318d93e1c075d9b3c2916ca3cf56e9c8da67d989` |
| adjudication SHA-256 | `831fe805bdffeee26693543bed60e96f9f0fe42945e315b16108677736589943` |
| executable SHA-256 | `0af7c80015204223b9d4b4b3d35ba0c3d832fc9b158583304c8cce8f02bed1a3` |
| orchestration time | `40.06300360000023` seconds; not a throughput result |
| raw result directory (unversioned) | `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_ffn_rmsnorm_cross_input_20260924/` |

Exactly one diagnostic ran. Four RMSNorm arms, four selected-up arms, and six
downstream propagation arms completed. Donor and reference graph execution
counts were both zero.

## Findings

| arm | downstream NRMSE | normalized maximum | gate |
|---|---:|---:|---|
| captured reference norm | `0` | `0` | PASS |
| captured production C norm | `0.002642086799405445` | `0.004687597394884091` | FAIL |
| production RMSNorm on reference `ffn_inp-1` | `0` | `0` | PASS; exact reference replay |
| production RMSNorm on C `ffn_inp-1` | `0.002642086799405445` | `0.004687597394884091` | FAIL; exact C replay |
| reference-input token-7-negated control | `0.20962964201860176` | `0.3668577386863162` | FAIL as required |
| negated norm-weight control | `1.578934067640191` | `1.0547903679865394` | FAIL as required |

The computed-reference arm is byte-identical to every captured reference
stage. The computed-C arm is byte-identical to captured production C at the
RMSNorm output and every later emitted stage. Its aggregate failure is also a
per-token failure: tokens 5 and 6 have downstream NRMSE
`0.002280978866229438` and `0.005616336995401208`, respectively. Thus an
aggregate-only acceptance rule would not change the decision.

The C-input chain remains tiny locally and is amplified downstream. Its stage
NRMSE / normalized maximum values are RMSNorm
`8.355378529300766e-8 / 1.2984894931424172e-7`, selected up
`1.6474086443438292e-7 / 1.1459829065728941e-7`, SwiGLU
`1.2415877785574164e-7 / 4.3461239180759115e-8`, Q6 down
`7.951168981266576e-5 / 9.274169798632739e-5`, routed output
`1.354682443199535e-5 / 1.1138183142366114e-5`, and complete downstream
`0.002642086799405445 / 0.004687597394884091`.

All captured anchors and schedule twins were byte-exact. Every one-byte input
mutation was refused, the arm-label swap was rejected, and both planted
causal controls failed. The computed-reference replay, computed-C replay,
artifact identity, source hashes, binary identity, and execution accounting
all passed.

## Adjudication and next boundary

Close production layer-1 FFN RMSNorm, the selected up projection, and every
later operator covered by the frozen amplifier. The accepted RMSNorm operator
passes on exact reference `ffn_inp-1`; the admitted C `ffn_inp-1` residual is
already sufficient to reproduce the downstream failure.

The immediate predecessor is
`ffn_inp-1 = output_projection(kqv_out-1) + l_out-0`. The production and
reference `kqv_out-1` payloads are byte-identical, while current `l_out-0`
differs from reference at NRMSE `5.4808396483682273e-8` and current
`ffn_inp-1` differs at `5.400853707562925e-8`. These descriptive facts do not
replace a controlled replay of the Q4_K output projection and residual add.
That sole successor is frozen in the
[layer-1 attention-output residual protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ATTENTION_OUTPUT_RESIDUAL_CROSS_INPUT_PROTOCOL_20260924.md).
It must use immutable captures and zero graphs. No quality, generation, RAM,
or throughput claim follows.
