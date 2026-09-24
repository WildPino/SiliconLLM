# STRAT-01 GigaChat 3.1 layer-1 terminal-component cross-input result

**Canonical verdict:** `LAYER1_FFN_OUTPUT_RESIDUAL_SUFFICIENT`

The accepted C `ffn_out-1` residual is sufficient to reproduce the layer-2
failure with exact reference `ffn_inp-1`. Conversely, C `ffn_inp-1` with exact
reference `ffn_out-1` passes by a wide margin. The terminal addition and all
layer-2 operators are closed; causality moves inside `ffn_out-1`.

## Deciding measurement

| arm | terminal-sum NRMSE | downstream NRMSE | verdict |
|---|---:|---:|---|
| captured reference `l_out-1` | `0` | `0` | PASS |
| reference input + reference FFN output | `0` | `0` | PASS; byte-exact replay |
| captured C `l_out-1` | `9.31051008726708e-06` | `0.002642086799405445` | FAIL |
| C input + C FFN output | `9.31051008726708e-06` | `0.002642086799405445` | FAIL; byte-exact replay |
| reference input + C FFN output | `9.310916242406284e-06` | `0.002642086799405445` | **FAIL; sufficient** |
| C input + reference FFN output | `3.595353521346349e-08` | `4.89406005505059e-05` | PASS |

The decisive reference-input/C-output arm produces the same downstream
SHA-256 as the complete C path,
`81635464fe7ec02d659bbc70d4d961e07743a670720bb3c4c1842dfcbc3ac254`,
despite its distinct terminal-sum and intermediate hashes. The registered
gates remain NRMSE `<=0.002` and normalized maximum `<=0.01`.

Direct component metrics agree with, but do not substitute for, the causal
result: C versus reference `ffn_inp-1` is NRMSE
`5.400853707562925e-08`; C versus reference `ffn_out-1` is
`1.247051153528305e-05`.

## Controls and provenance

- captured reference/C anchors are exact;
- reference/reference and C/C recompositions replay all five stages
  byte-exactly;
- prefill and cached-composition twins are exact;
- mutations of all nine frozen inputs reject;
- the cross-arm label swap rejects;
- token-7 FFN-output negation rejects at downstream NRMSE
  `0.276314735587165`;
- one diagnostic invocation; zero donor and zero reference graphs;
- accepted artifact SHA-256:
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- committed source HEAD:
  `99e7bee107cf986e4a136fa38932e8a7f61778ed`.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_terminal_component_cross_input_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `9bdda63cabb7bfe00d7e9dbf76cbead7216aa03064e816cdfa62748fe314ff98` |
| scientific binary | `3ac7b82da743ec318d2815cc51905777b2bae3f79480ecdc7a7668177e7ad4ab` |

## Consequence and next boundary

Do not repeat the terminal-component cell, either producer, terminal addition,
or any layer-2 operator. The immediate captured decomposition is
`ffn_out-1 = ffn_moe_out-1 + ffn_shexp-1`.

Frozen-payload arithmetic predicts the routed branch: C versus reference
NRMSE is `1.3547864451924633e-05` for `ffn_moe_out-1` and
`1.358712399393316e-07` for `ffn_shexp-1`. The C-routed/reference-shared sum
retains NRMSE `1.2470959514881644e-05`; the inverse cross is only
`4.121498651045465e-08`. This is hypothesis evidence, not a downstream
verdict.

The separately frozen [FFN-output component protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_FFN_OUTPUT_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md)
must propagate both immutable cross-sums through exact reference
`ffn_inp-1` and the closed layer-2 path. No graph, tokenizer, later-layer,
logits, generation, quality, RAM, or rate claim follows.
