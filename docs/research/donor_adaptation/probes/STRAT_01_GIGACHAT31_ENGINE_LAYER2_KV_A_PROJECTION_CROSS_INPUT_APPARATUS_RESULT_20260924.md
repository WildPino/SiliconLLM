# STRAT-01 GigaChat 3.1 layer-2 KV-A projection apparatus result

**Verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The frozen layer-2 KV-A projection cross-input diagnostic is qualified without
opening the accepted GGUF or any evidence payload.

## Qualification

- Clang C11 `-O3 -mavx2 -mfma` build passed;
- new KV-A projection self-test: 4/4 checks;
- inherited KV RMSNorm self-test: 7/7;
- inherited KV-partition self-test: 10/10;
- legacy engine: 73,024 kernel checks plus exponential check;
- Python tests: 6/6, including the projection `(8, 576)` metric-shape guard;
- diagnostic invocations: 0; donor/reference graphs: 0;
- accepted artifact opened: false; errors: none.

The helper uses the production Q4_K KV-A projection and the already closed
RMSNorm, cache/attention, and V-B operators. It contains no replacement
attention kernel and makes no claim from the accepted model.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_kv_a_projection_cross_input_apparatus_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `2863723bab80e928aaecff889c71fe6e4be00c6d94a72b555ea811bd48c5ecff` |
| C binary | `c17b1a8a6fcd47b77212a626dc11d7f17308ce9ebd3ca264f6b714c66e2733e1` |

The apparatus observed frozen protocol commit
`4d9586c7683c8d225414c528dfcd0748dbe3634e`. Commit the exact passing
implementation before the one permitted zero-graph scientific invocation.

No causal, repair, quality, RAM, or rate claim is made.
