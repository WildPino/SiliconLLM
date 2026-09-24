# STRAT-01 layer-2 attention cross-input apparatus result

**Verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The model-free apparatus required by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER2_ATTENTION_CROSS_INPUT_PROTOCOL_20260924.md)
is qualified. It did not open the accepted GGUF, invoke the boundary
diagnostic, or execute a donor/reference graph.

Raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_attention_cross_input_apparatus_20260924/`

- `adjudication.json` SHA-256:
  `b454a141effabb1d423c31518a6c84e168d67a8d6db203b43923910324af3edd`;
- C binary SHA-256:
  `cd3f594ef2dbdebfbce0b853ebd903c1ba3dcb34f167bcf8c2fe013576c1ed8b`;
- observed pre-implementation Git head:
  `b971fdb36384177ccf2934026a7335a30164f6eb`;
- apparatus duration: `9.339` seconds.

The new five-check layer-2 self-test, inherited Rung-2C cross-input self-test,
legacy engine self-test, and all six targeted Python tests pass. The apparatus
records zero diagnostic invocations, zero donor/reference graphs, and
`artifact.opened=false`.

The exact qualified source hashes are:

| source | SHA-256 |
|---|---|
| `engine.c` | `79c45052db33dcecb15d0b5d23689e8fe0d86bcf36e08d274849e31dc15b239b` |
| layer-2 header | `88998116d2f8f7c900e0b12a9ba548e962e1a8116b1f07a1b90569ce9b6fc37c` |
| inherited layer-1 header | `3d157571efd67ddd0415343147359bea2ba2fbc1f00a79cab2484f68ac0a99f5` |
| runner | `208d4e00d4a5fd04c4ed75ee3e463113f198f715c66bb04af53a03b5dea3706c` |
| tests | `14500f63a7cabac5ef14b55e56e68ffb0ba941f73c13a4e008334ffb5408b6af` |
| protocol | `fa5906886ca02707673c849e077d3a649d2dfbe31d7e811c57837642181fcbb6` |

This qualifies evidence plumbing only. Commit these exact sources, then run
the one permitted boundary diagnostic. It may hash/parse the accepted model
for `blk.2.attn_v_b.weight`, but must execute zero model graphs. No timing or
rate result follows.
