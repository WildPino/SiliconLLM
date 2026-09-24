# STRAT-01 layer-1 routed-SwiGLU component apparatus repair-1 result

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

Repair 1 of the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_SWIGLU_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md)
qualified model-free after preserving
[VOID 1](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_SWIGLU_COMPONENT_CROSS_INPUT_VOID1_20260924.md).
The repair removes only the contradictory reference/reference scalar replay
assertion, retains the production C/C replay assertion, and makes completed-Q6
accounting exact on failure.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_routed_swiglu_component_cross_input_apparatus_repair1_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `4c9b0f23bca44db0d48d0d3e0c6d5a4722c932eb9141adeb415a3a20c31a73b1` |
| compiled C binary | `66b5711033d7721e90cfa12ef7c21e92d1db2f41ad524e37ca3697a05d876270` |
| runner | `57c0a3a04c87fdd211973e756c70881f17da444ce28580bab9f20805ee3e9c7c` |
| Python tests | `07770d10e3ec30a4f20fc72b25c861a82c47599b606313b9faf76a69968fe4e7` |
| repaired protocol | `b95909f0636a6db634fd71befd40fd87d184bfa514f69d0f31396d3bd479be6e` |
| `engine.c` | `24a43b132439479036cd3025210e7d05bc3d3d7a40fb8b424c6cdb0dd1858817` |
| diagnostic header | `307b8c01b9706e4f15c8e1a4ea746a6a1ee8ea3abc2e850000e533da68e35f6f` |
| shared SSE2 primitive | `c87a7f9456807d5cca82a644cfe87a1d6b0e8c0dc08dcb7c9fc1c33e06ec42d1` |

The observed repository head was
`4f79078d82b44fc522df019df50cdcb1ccfe6f5e`; repaired sources were intentionally
uncommitted during qualification. The run took `7.907` seconds.

## Qualification facts

- `diagnostic_invocations = 0`;
- `q6_arms_completed = 0`;
- artifact `opened = false`;
- donor/reference graph executions are `0 / 0`;
- all six new/inherited C self-tests pass;
- all nine Python tests pass, including explicit repair-scope checks;
- the nine scientific arms, thresholds, hashes, and decision order are
  unchanged from the frozen protocol.

Graphify refresh was attempted once for repair 1 and interrupted after 30
seconds in the same Windows extraction failure. It was not retried.

## Stop rule

This is apparatus evidence only. Commit these exact sources and verify their
hashes, then execute one repaired scientific invocation. No graph, production
repair, quality, RAM, or throughput claim is licensed here.
