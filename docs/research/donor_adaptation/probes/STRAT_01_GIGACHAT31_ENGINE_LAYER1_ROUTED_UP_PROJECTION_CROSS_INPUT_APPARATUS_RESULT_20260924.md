# STRAT-01 layer-1 routed-up projection apparatus result

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The model-free qualification authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_UP_PROJECTION_CROSS_INPUT_PROTOCOL_20260924.md)
passed on its first invocation. It opened neither the accepted artifact nor
any frozen tensor payload.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_routed_up_projection_cross_input_apparatus_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `aac0babb4456e604885a7885290d1375e49fe8941f18a481ab70b1d988985682` |
| compiled C binary | `24af0b327aea327249448e97164783ff9365eca0f8bc9ed38733a9be8da17c52` |
| runner | `6d655e7f0b3ce96787b0a3f714f449d43bff84bbbcd412b75fc4fef8c322634c` |
| Python tests | `3e385ad473f8d2deac15c3dea1a40f8c83024fcb7e99e267f81a21f488c89379` |
| protocol | `ebeabe4f7ed4fac9672d1f68f90784ee5232bba97dac4eab1f8f829281b5e614` |
| `engine.c` | `7e4308c8640f3a09f2fdb321743b386daced63a1c438995f675cdaf77325c033` |
| diagnostic header | `bcbcfc4d987298d718267557f48bdebc59a732815179d34918aeebfa38abb073` |

The observed repository head was
`fb0a09c7bc4967e08bcde314bc556583d78674d3`; new sources were intentionally
uncommitted during qualification. The run took `7.306` seconds.

## Qualification facts

- `diagnostic_invocations = 0`;
- `up_arms_completed = 0`;
- `q6_arms_completed = 0`;
- donor/reference graph executions are `0 / 0`;
- accepted artifact `opened = false`;
- the new selected-Q4_K up helper compiles and passes six C checks;
- all five new/inherited C self-tests and all eight Python tests pass;
- fixed arm order, payload shapes, predecessor identity, protocol decisions,
  and label-swap refusal are tested.

The required Graphify refresh was attempted once and interrupted after 30
seconds in the known Windows extraction failure. It was not retried.

## Stop rule

This is apparatus evidence only. Commit these exact sources and verify their
hashes before one scientific invocation. That invocation may execute the
three fixed current-up projections and five Q6 arms, but no donor/reference
graph. No repair, quality, RAM, or throughput claim follows here.
