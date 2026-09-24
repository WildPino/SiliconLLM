# STRAT-01 GigaChat 3.1 layer-1 FFN-output component apparatus result

**Canonical verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The model-free apparatus for the immutable
`ffn_out-1 = ffn_moe_out-1 + ffn_shexp-1` split qualified on its first
attempt. The accepted artifact and frozen scientific payloads were not opened,
and no graph or diagnostic arm executed.

## Qualification

- Clang C11 `-O3 -mavx2 -mfma` build passed;
- new FFN-output component self-test: 6/6 checks;
- inherited terminal-component self-test: 6/6;
- inherited attention RMSNorm self-test: 5/5;
- inherited KV-A projection self-test: 4/4;
- inherited KV RMSNorm self-test: 7/7;
- inherited KV-partition self-test: 10/10;
- legacy engine: 73,024 kernel checks plus exponential check;
- Python tests: 9/9;
- diagnostic invocations: 0; donor/reference graphs: 0;
- accepted artifact opened: false; errors: none.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_ffn_output_component_cross_input_apparatus_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `8c6569cc1f6a8a701ae2f9a631e1dd5e9dd9f48df45c9e4a9fd2ac2da2098c58` |
| C binary | `f7a703969ee4277119894126c9d847a2496b2d0703d9f1a85eb48091d590d24d` |

The apparatus observed HEAD
`2be483e09998a46ea86e0ce5d1405905eaac85b0` and records the exact
uncommitted source hashes. Commit those exact sources before the sole permitted
zero-graph scientific invocation.

No layer-1 or layer-2 repair, causal verdict, tokenizer, later-layer, quality,
RAM, or rate claim is made.
