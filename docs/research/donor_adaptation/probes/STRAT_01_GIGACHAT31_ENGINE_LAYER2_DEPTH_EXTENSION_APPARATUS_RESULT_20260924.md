# STRAT-01 layer-2 depth-extension apparatus result

**Verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The model-free apparatus required by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_PROTOCOL_20260924.md)
is qualified. It did not open the accepted GGUF, invoke either scientific
producer, or execute a donor/reference graph.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_depth_extension_apparatus_20260924/`

- `adjudication.json` SHA-256:
  `2aec353e297a05916b1db10916feb3fae2851a29794eb1a736ef314fed878c05`;
- C binary SHA-256:
  `41b0901204eda0e40c78d7c9bc967ab0a37ea22854b87261e437a75de0c5cd02`;
- reference binary SHA-256:
  `3bf2f03b744fd7dd0d8d6b2ede7d83adb721b47f0389d3f5d680f2f7369dfe6c`;
- observed pre-implementation Git head:
  `b21d242e5583366b0489b31daae774677cbfe4b7`;
- apparatus duration: `154.873` seconds.

The duration is apparatus cost, not emitted-token throughput.

## Qualified surface

- all 179 STRAT-01 Python tests pass;
- all 22 registered C self-test commands pass, including the new Rung-2D
  self-test after the complete Rung-2A/B/C chain;
- the separately compiled pinned Rung-2D reference self-test passes;
- all twelve source controls pass;
- the accepted production predecessor validates at SHA-256
  `d543c56d7e463cfec23e4fa3a7ec7fd5bfecbdccfdf84351ddf12085fcc3cc94`;
- the sixteen `blk.2` tensor descriptors, 32-checkpoint surface, three-layer
  cache schema, both schedule markers, and separate reference variant are
  present and fail-closed;
- expected scientific helper accounting is fixed at `6912` QK and `786432`
  value invocations;
- scientific C/reference invocations and donor/reference graphs are all
  exactly zero.

The qualified source hashes include:

| source | SHA-256 |
|---|---|
| `engine.c` | `092e64fa953a86b2a2c10d8437f9b46667abc6fb630eeb3b4a511e012f2c897a` |
| `strat01_gguf_rung2d.h` | `05ac554f51298e55868a36bb80631f308490650b35b693819312147814f9e348` |
| reference source | `1772b79db20bfda9c9114794e287e171a7e3fcce949b8c935be78060972e0c0e` |
| runner | `fff59cdacbd572c1b063df38fdf241abb044d0c0fba70a9ec62c2edec3b332dc` |
| tests | `8f7e269c6c5a421677a79c5dc9bec08f1dc40e464a3f6fa417cfd75b64f9851c` |

## Interpretation and next action

This result qualifies evidence plumbing only. It establishes no layer-2
numerical result. Commit these exact sources before accepted-artifact access.
After that commit, the protocol permits exactly one pinned-reference
invocation and one C-engine invocation, each completing the two frozen
schedules. Do not rerun Rung-2C or any predecessor cell.

No later-layer, tokenizer, logits, generation, quality, RAM, or rate claim
follows. No `SPEED_LEDGER.md` update is due.
