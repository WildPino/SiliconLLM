# STRAT-01 layer-2 depth-extension apparatus result

**Verdict:** `APPARATUS_READY_NO_DONOR_EXECUTION`

The model-free apparatus required by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_PROTOCOL_20260924.md)
is qualified. It did not open the accepted GGUF, invoke either scientific
producer, or execute a donor/reference graph.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_depth_extension_apparatus_repair1_20260924/`

This `repair1` supersedes the earlier model-free apparatus directory. The
first run qualified build, wiring, and evidence plumbing, but the scientific
adjudicator had not yet been implemented in the runner and therefore was not
part of its source hash. `repair1` qualifies the complete runner without
rerunning a model or graph. The first directory remains provenance only.

- `adjudication.json` SHA-256:
  `f9b7abad51357c47351eec7c5937bb0563b6a6ae8b9c9ff32483e6ce7c232ae0`;
- C binary SHA-256:
  `2526b3189f9b192ae4990870d0ec01c059187078bd77a25cf20de9f7f937ea55`;
- reference binary SHA-256:
  `e520ec7e94948498e69b53a8fc7d27db3f03f04ac0eaffba9c20b3fe5398b4d0`;
- observed pre-implementation Git head:
  `9cfe89146b95ad1147954fdd116e523abe0e87e6`;
- apparatus duration: `164.250` seconds.

The duration is apparatus cost, not emitted-token throughput.

## Qualified surface

- all 184 STRAT-01 Python tests pass, including scientific checkpoint
  translation, immutable frontier hashes, and exact-once graph-marker tests;
- all 22 registered C self-test commands pass, including the new Rung-2D
  self-test after the complete Rung-2A/B/C chain;
- the separately compiled pinned Rung-2D reference self-test passes;
- all twelve source controls pass;
- the accepted production predecessor validates at SHA-256
  `d543c56d7e463cfec23e4fa3a7ec7fd5bfecbdccfdf84351ddf12085fcc3cc94`;
- the sixteen `blk.2` tensor descriptors, 32-checkpoint surface, three-layer
  cache schema, variant-specific Rung-2D schedule markers, and separate
  reference variant are present and fail-closed;
- the runner now contains the complete scientific validators and adjudicator,
  including exact predecessor-reference continuity and layer-2 causal
  negative controls;
- expected scientific helper accounting is fixed at `6912` QK and `786432`
  value invocations;
- scientific C/reference invocations and donor/reference graphs are all
  exactly zero.

The qualified source hashes include:

| source | SHA-256 |
|---|---|
| `engine.c` | `092e64fa953a86b2a2c10d8437f9b46667abc6fb630eeb3b4a511e012f2c897a` |
| `strat01_gguf_rung2d.h` | `05ac554f51298e55868a36bb80631f308490650b35b693819312147814f9e348` |
| reference source | `f94a7773b3e59de518a0b8d171f02fd4abdd98100b80321eb4cc6fec96160b75` |
| runner | `b27d9929148c07d67289c55050945ce0bfbaf1e07c117898d1c235d6a6c8007d` |
| tests | `f446288bcf2cc62467c58be92fc40359349ce0eebf3b1e3334707940865fdab9` |

## Interpretation and next action

This result qualifies evidence plumbing only. It establishes no layer-2
numerical result. Commit these exact sources before accepted-artifact access.
After that commit, the protocol permits exactly one pinned-reference
invocation and one C-engine invocation, each completing the two frozen
schedules. Do not rerun Rung-2C or any predecessor cell.

No later-layer, tokenizer, logits, generation, quality, RAM, or rate claim
follows. No `SPEED_LEDGER.md` update is due.
