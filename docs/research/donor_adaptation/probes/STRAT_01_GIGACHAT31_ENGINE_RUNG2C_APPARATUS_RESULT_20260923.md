# STRAT-01 GigaChat 3.1 engine Rung 2C apparatus result

**State:** `APPARATUS_READY_NO_DONOR_EXECUTION`.

The frozen [Rung 2C protocol](STRAT_01_GIGACHAT31_ENGINE_RUNG2C_PROTOCOL_20260923.md)
now has a compiled and model-free-tested implementation. No reference or C
producer graph was executed. The next action is the protocol's single pinned
reference invocation; do not repeat apparatus unless a pre-scientific source
defect is found.

## Accepted apparatus record

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_apparatus_repair1_20260923/`.

| item | accepted value |
|---|---|
| status | `APPARATUS_READY_NO_DONOR_EXECUTION` |
| errors | `[]` |
| adjudication SHA-256 | `29abb028a08372cd1b4aef30ae65cf47b0ee9a220495512096c860bced26c65e` |
| run-manifest SHA-256 | `856070698733b83fe618436bd79b1a126e0e57304b9f4d8c42f40582a9d4a245` |
| accepted GGUF SHA-256 | `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| block-0 adjudication binding | `58f661d9461f3dc17b2a52f830dcc9f36423621528941d475e23ee2aed8e0bf2` |
| block-0 manifest binding | `7b8de95a3766e4f76bde37f4038ece2986b9067ad9ac7a0e4cc11855a3062497` |
| reference producer invocations / graph schedules | `0 / 0` |
| C producer invocations / graph schedules | `0 / 0` |

The preceding apparatus directory without the `repair1` suffix is preserved.
It passed all then-current tests but used a prospective execution-accounting
schema that did not distinguish one producer invocation from its two graph
schedules. It executed neither producer. `repair1` is canonical because the
corrected schema freezes future accounting as one invocation and two completed
schedules per producer.

## What is implemented

- `engine.c` has a dedicated `--strat01-gguf-rung2c` production dispatch and
  model-free self-test.
- Layer-1 attention reuses the accepted RMSNorm, RoPE, Q4_K×Q8_K,
  Q5_0×Q8_0, cache, and residual primitives.
- The new MoE path implements the F32 router, sigmoid plus selection-only
  bias, ordered top four, normalized unbiased weights, selected rank-3
  Q4_K/Q6_K experts, ordered routed aggregation, shared expert, and residual.
- The pinned reference producer captures the corresponding llama.cpp
  callbacks directly. It emits `ffn_moe_topk-1` as raw I32 and creates cache
  witnesses by F16-rounding the captured `Kcur` rows; it does not recompute the
  router or experts in Python.
- Manifests cover the exact block-0 start witness, 31 layer-1 checkpoints, two
  cache layers, payload types/shapes, source hashes, and invocation/schedule
  counts.

## Accepted checks

- Clang C build: PASS.
- Rung 2A + Rung 2B + Rung 2C C self-tests: PASS; Rung 2C reports 14 checks.
- Combined RMS/Q5Q8 repair chain: PASS.
- Legacy kernel self-test: 73,024 checks, zero worst absolute error; PASS.
- Python apparatus tests: 6/6 PASS, including all ten frozen causal controls
  and invocation-versus-schedule accounting.
- Pinned llama.cpp reference build and reference self-test: PASS at commit
  `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.
- Source controls prove that Rung 2C delegates the accepted Q4, Q5, Q6,
  RMSNorm, and cache primitives rather than defining alternate codecs.

## Non-claims and next action

This apparatus result contains no block-1 donor output and establishes no
numerical parity, later-layer coverage, quality, RAM, or speed claim. It does
not update `SPEED_LEDGER.md`.

After the implementation commit, run exactly one pinned-reference producer
invocation. It must emit both `prefill8` and `cached7p1`, reproduce the frozen
reference `l_out-0` hash in both arms, and validate completely before the one C
producer invocation is allowed.
