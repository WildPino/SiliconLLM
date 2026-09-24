# STRAT-01 layer-1 attention-output residual cross-input result

**Status:** `LAYER1_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT`

## Scope and provenance

The sole scientific invocation authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ATTENTION_OUTPUT_RESIDUAL_CROSS_INPUT_PROTOCOL_20260924.md)
and qualified
[apparatus](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ATTENTION_OUTPUT_RESIDUAL_CROSS_INPUT_APPARATUS_RESULT_20260924.md)
completed without error. It opened and validated the accepted 6,474,702,976-byte
GGUF, SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.

| identity | value |
|---|---|
| repository HEAD | `0828a61b43d706ab9cd990b3672c7156cd633281` |
| adjudication SHA-256 | `5b1fcdce903fccea697ff5235a566e0f04ef04ebffe9c2ef46af8bc249bfe0cf` |
| executable SHA-256 | `1dcbd6764c802f6a30ae9644b993796aa6642b885b59476c09476ca285633699` |
| orchestration time | `39.6501` seconds; not a throughput result |
| raw result directory (unversioned) | `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_attention_output_residual_cross_input_20260924/` |

Exactly one diagnostic ran. Two output projections, four constructed FFN
inputs, six RMSNorm arms, six selected-up arms, and six downstream arms
completed. Donor and reference graph execution counts were both zero.

## Findings

| arm | downstream NRMSE | normalized maximum | gate |
|---|---:|---:|---|
| captured reference `ffn_inp-1` | `0` | `0` | PASS |
| captured production C `ffn_inp-1` | `0.002642086799405445` | `0.004687597394884091` | FAIL |
| current projection + reference `l_out-0` | `0` | `0` | PASS; exact reference replay |
| current projection + production C `l_out-0` | `0.002642086799405445` | `0.004687597394884091` | FAIL; exact C replay |
| reference-residual token-6-negated control | `0.19465512820895675` | `0.4157779679557519` | FAIL as required |
| exact-`kqv_out-1` token-7-negated control | `0.07619580311759673` | `0.14497258715833466` | FAIL as required |

The current output projection on exact `kqv_out-1` is byte-identical to the
captured reference projection. Adding reference `l_out-0` reproduces reference
`ffn_inp-1` and every emitted downstream stage byte-for-byte. Replacing only
that residual with production C `l_out-0` reproduces production C at every
stage. Tokens 5 and 6 fail the downstream per-token NRMSE gate at
`0.002280978866229438` and `0.005616336995401208`.

The C-residual chain remains very small locally and is amplified downstream.
Its NRMSE / normalized-maximum sequence is `ffn_inp-1`
`5.400853707562924e-8 / 5.066021404143073e-8`, FFN RMSNorm
`8.355378529300766e-8 / 1.2984894931424172e-7`, selected up
`1.6474086443438292e-7 / 1.1459829065728941e-7`, SwiGLU
`1.2415877785574164e-7 / 4.3461239180759115e-8`, Q6 down
`7.951168981266576e-5 / 9.274169798632739e-5`, routed output
`1.354682443199535e-5 / 1.1138183142366114e-5`, and complete downstream
`0.002642086799405445 / 0.004687597394884091`.

All captured anchors, source identities, schedule twins, normal projection,
reference/C stage replays, one-byte mutation refusals, arm-label refusal, and
both planted controls passed.

## Existing-evidence composition bridge

The protocol proposed a new split of
`l_out-0 = ffn_inp-0 + ffn_out-0` if this result fired. That run would now be
duplicate. Existing immutable evidence already binds both addends after the
SSE2 repair:

| payload | SHA-256 | relation |
|---|---|---|
| reference/current `ffn_inp-0` | `baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1` | byte-identical |
| reference `ffn_out-0` | `f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4` | pinned llama.cpp capture |
| current Q6 on byte-exact reference SwiGLU | `f52e0e5bdb22a17ae4a9e0ee9e0c6ffc6bb485b3e61e3f821039eae38697d6ce` | post-SSE2 candidate |
| reference `l_out-0` | `385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa` | exact reference sum |
| current `l_out-0` | `a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11` | production capture |

A read-only float32 reconstruction over the frozen payloads established both
homogeneous identities byte-for-byte:

`reference ffn_inp-0 + reference ffn_out-0 = reference l_out-0`

`reference ffn_inp-0 + current-Q6 ffn_out-0 = production C l_out-0`.

The second sum has SHA-256 `a71c814d...f05f11` and is byte-identical both to
the post-SSE2 candidate and to the later production-integration capture. The
current Q6 output differs from reference at NRMSE
`8.348319997561633e-8`, normalized maximum
`7.777310102196965e-8`, across 8,072 of 12,288 float32 elements. The resulting
`l_out-0` differs at NRMSE `5.4808396483682273e-8` across 6,135 elements.
No graph, model invocation, or new scientific arm was used for this identity
bridge.

## Adjudication and next boundary

Close the layer-1 Q4_K attention-output projection and residual addition. The
post-SSE2 block-0 terminal residual alone is sufficient to reproduce the
failure under the stronger layer-2 amplifier.

The byte-exact `ffn_inp-0`, byte-exact repaired SwiGLU input, current-Q6 output
hash, and exact terminal sum jointly localize that sufficient residual to the
block-0 Q6_K×Q8_K down-projection result. This does not invalidate the older
Q6 cross-input PASS: that result used a weaker layer-1 gate, under which the
same `8.35e-8` direct residual passed. The estimand changed because the later
layer-2 chain exposed amplification that was not then available.

Do not repeat the terminal-component split, the SwiGLU repair, or the old Q6
cross-input cell. The sole new coordinate is exact arithmetic parity between
the current scalar/generic Q6 reduction and pinned llama.cpp's active AVX2/FMA
Q6 reduction on the same Q8_K bytes. It is frozen in the
[block-0 Q6_K×Q8_K AVX2 parity protocol](STRAT_01_GIGACHAT31_ENGINE_BLOCK0_Q6K_Q8K_AVX2_PARITY_PROTOCOL_20260924.md).
No quality, generation, RAM, or throughput claim follows.
