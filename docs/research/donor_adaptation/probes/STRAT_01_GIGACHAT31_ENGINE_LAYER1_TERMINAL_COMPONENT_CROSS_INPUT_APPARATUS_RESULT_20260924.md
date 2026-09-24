# STRAT-01 GigaChat 3.1 layer-1 terminal-component apparatus result

**Canonical verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The model-free apparatus for the immutable
`l_out-1 = ffn_out-1 + ffn_inp-1` split qualified on its first attempt. The
accepted artifact and frozen scientific payloads were not opened, and no graph
or diagnostic arm executed.

## Qualification

- Clang C11 `-O3 -mavx2 -mfma` build passed;
- new terminal-component self-test: 6/6 checks;
- inherited attention RMSNorm self-test: 5/5;
- inherited KV-A projection self-test: 4/4;
- inherited KV RMSNorm self-test: 7/7;
- inherited KV-partition self-test: 10/10;
- legacy engine: 73,024 kernel checks plus exponential check;
- Python tests: 8/8;
- diagnostic invocations: 0; donor/reference graphs: 0;
- accepted artifact opened: false; errors: none.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_terminal_component_cross_input_apparatus_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `0bc70655164ad5d0cba9c6ac93ff62ea0e4d2cdf8797bc28042584a66fdcb2fc` |
| C binary | `8de555cca0234032ef77afc052661b200fc35847060220abd8e40b87d24a5314` |

The apparatus observed HEAD
`a54c109f6cbb0dd05d34072ef1329b210eafa748` and records the exact
uncommitted source hashes. Commit those exact sources before the sole permitted
zero-graph scientific invocation.

No layer-1 or layer-2 repair, causal verdict, tokenizer, later-layer, quality,
RAM, or rate claim is made.
