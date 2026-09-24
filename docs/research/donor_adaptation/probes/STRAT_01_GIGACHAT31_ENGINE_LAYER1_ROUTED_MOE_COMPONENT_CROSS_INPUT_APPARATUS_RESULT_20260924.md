# STRAT-01 GigaChat 3.1 layer-1 routed-MoE component apparatus result

**Canonical verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The model-free apparatus for the immutable expert-down versus normalized
router-weight split qualified on its first attempt. The accepted artifact and
frozen scientific payloads were not opened, and no graph or diagnostic arm
executed.

## Qualification

- Clang C11 `-O3 -mavx2 -mfma` build passed;
- new routed-MoE component self-test: 7/7 checks;
- inherited FFN-output component self-test: 6/6;
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
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_routed_moe_component_cross_input_apparatus_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `d3919553fe98c70b1a470f1be22a5c75156811bd4ae9ba90d8c67c80fb4e8607` |
| C binary | `f0e6dc155ab5f7e75cabd2642bae8f79d3d0ee3b74af3f3b17f30a6298b7c61c` |

The apparatus observed HEAD
`44bedc919d3149e80256822cdc0178a9b1eecc24` and records the exact
uncommitted source hashes. Commit those exact sources before the sole permitted
zero-graph scientific invocation.

No routing or expert repair, causal verdict, tokenizer, later-layer, quality,
RAM, or rate claim is made.
