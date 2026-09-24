# STRAT-01 GigaChat 3.1 layer-2 KV-partition apparatus result

**Verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The separately frozen layer-2 KV-partition diagnostic is qualified without
opening the accepted GGUF or any frozen evidence payload. Its only scientific
coordinate is composition of the compact K row at the exact 512/64 boundary
under a fixed reference query.

## Qualification

- Clang C11 build passed with `-O3 -mavx2 -mfma`;
- new partition self-test: 10/10 checks;
- predecessor whole-KV self-test: 5/5 checks;
- inherited Rung-2C cross-input self-test: 11/11 checks;
- legacy engine self-test: 73,024 kernel checks plus exponential check;
- Python tests: 7/7;
- diagnostic invocations: 0;
- donor graphs: 0;
- reference graphs: 0;
- accepted artifact opened: false;
- errors: none.

The new helper composes `[0,512)` and `[512,576)` before the existing F16
cache-write path, then calls the accepted causal-attention plus
`blk.2.attn_v_b.weight` operator. It does not implement a second attention
kernel.

## Evidence

Raw apparatus directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_kv_partition_cross_input_apparatus_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `d903f770a95afe87319da7705ed43b809f35c25bea964953b8a9acef387699a8` |
| compiled C binary | `3196f496cf26df7ea0031680c1aa8bb492cc3417b8ecbe04c4edae4661029671` |

The apparatus observed protocol commit
`fc732844a891d9745603c4b8d44cd0714005e6cd`. The implementation sources were
necessarily uncommitted during this model-free qualification; the scientific
runner refuses untracked or dirty source. Commit this exact passing source set
before the sole allowed boundary invocation.

## Next action

Run the diagnostic exactly once from the committed sources. It may hash/parse
the accepted GGUF only to apply the layer-2 V-B matrix. It must execute zero
donor/reference graphs and must not add a cached-schedule duplicate.

No claim is made about the causal partition, a repair, later operators,
quality, RAM, or rate.
