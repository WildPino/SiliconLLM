# STRAT-01 GigaChat 3.1 layer-1 routed-SwiGLU component apparatus result

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The model-free qualification authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_SWIGLU_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md)
passed on its first invocation. It compiled the new diagnostic, passed all six
new/inherited C self-tests and all eight Python tests, and opened neither the
accepted GGUF nor any frozen tensor payload.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_routed_swiglu_component_cross_input_apparatus_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `8e5583e821800383ca07d6df75e566ae06a7c8830ed739c5ef128157eb7a7f7b` |
| compiled C binary | `b0decb9cd7a975019fca9585a4bd70e008587fca1a3902db7054eedb645e3c65` |
| runner | `79d7b6f73f5de78e0862efcb2ea46dfd375f34f4369a08b3053f87d5e783588b` |
| Python tests | `cc5cda876e65bd3d0c841046a9611391ed5834a78cad959c5f64c7cc5105a823` |
| protocol | `9679ec5b0b508c42aaf57d860c20e814703b8ef8579b92971816165d090d72c3` |
| `engine.c` | `24a43b132439479036cd3025210e7d05bc3d3d7a40fb8b424c6cdb0dd1858817` |
| diagnostic header | `e01acbdfba55d5f5c711ed38e11a2ee5cce7036d4f8c68ad5ac31026dd236b20` |
| shared SSE2 primitive | `c87a7f9456807d5cca82a644cfe87a1d6b0e8c0dc08dcb7c9fc1c33e06ec42d1` |

The observed repository head was
`57d968739be00dd5ec8424001a09e2a9f1301111`; the source files were intentionally
uncommitted during qualification. The orchestration took `7.013` seconds.

## Qualification facts

- `diagnostic_invocations = 0`;
- `q6_arms = 0`;
- `donor_graph_executions = 0`;
- `reference_graph_executions = 0`;
- accepted artifact `opened = false`;
- the nine fixed arm labels and four payload shapes are schema-tested;
- arm-label swapping is rejected by the Python manifest validator;
- scalar and SSE2 synthetic semantics are distinct and both execute;
- current Q6, routed-reduction, propagation, SSE2-semantics, and legacy kernel
  self-tests all pass.

The required Graphify refresh was attempted once after code modification. On
Windows it remained in extraction and was interrupted after 30 seconds; no
second attempt is authorized for this code cycle.

## Stop rule

This is apparatus evidence only. It establishes no causal, repair, quality,
RAM, or throughput result. Commit the exact qualified sources, verify their
hashes against this record, then run exactly one scientific invocation. That
invocation may execute the nine fixed Q6 arms but must execute zero donor and
zero reference graphs.
