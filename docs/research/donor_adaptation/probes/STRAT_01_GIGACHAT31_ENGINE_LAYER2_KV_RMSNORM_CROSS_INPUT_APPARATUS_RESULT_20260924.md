# STRAT-01 GigaChat 3.1 layer-2 KV RMSNorm apparatus result

**Verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The frozen KV RMSNorm diagnostic is qualified without opening the accepted
GGUF or any evidence payload.

## Qualification

- Clang C11 `-O3 -mavx2 -mfma` build passed;
- new RMSNorm self-test: 7/7 checks;
- inherited partition self-test: 10/10;
- inherited attention self-test: 5/5;
- legacy engine: 73,024 kernel checks plus exponential check;
- Python tests: 6/6;
- diagnostic invocations: 0; donor/reference graphs: 0;
- accepted artifact opened: false; errors: none.

The helper calls the production RMSNorm and accepted downstream
cache/attention/V-B functions; it contains no replacement attention kernel.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_kv_rmsnorm_cross_input_apparatus_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `6df7e7659be0265d7af9b5bc7aca9b31ccb644a9d856785ff6efb1f9f75ec3a8` |
| C binary | `29cb605c2804793fd7605d668733fea1792cb938c6d748431cc98e7998fa97c7` |

The apparatus observed protocol commit
`6159d2e53333d850778bc06fe24a65032d173772`. Commit the exact passing
implementation before the one permitted zero-graph scientific invocation.

No causal, repair, quality, RAM, or rate claim is made.
