# STRAT-01 GigaChat 3.1 engine Rung 2C apparatus result

**Current state:** `APPARATUS_READY_NO_DONOR_EXECUTION` after the narrow
post-VOID repair.

The first scientific invocation exposed a lowercase `i32` versus uppercase
`I32` serialization defect and non-durable graph accounting. This record
remains preserved as the pre-execution apparatus result. The fresh `repair2`
apparatus below validates the narrow repair and authorizes one
changed-coordinate scientific rerun.

The frozen [Rung 2C protocol](STRAT_01_GIGACHAT31_ENGINE_RUNG2C_PROTOCOL_20260923.md)
has a compiled and model-free-tested implementation. No reference or C
producer graph was executed in the repair apparatus. The previously exposed
source defect is recorded in VOID 1 above.

## Accepted apparatus record

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_apparatus_repair2_20260923/`.

| item | accepted value |
|---|---|
| status | `APPARATUS_READY_NO_DONOR_EXECUTION` |
| errors | `[]` |
| source HEAD | `4c5d1af6274d1067076485398d58ce67dfc1ab96` |
| adjudication SHA-256 | `280ab3f69960fdb8abc81697b5143c9851c0c8c9fc261e4e0412e587c3d5371e` |
| run-manifest SHA-256 | `97e5ce98efc61a6b40eacd893d29e7f2d03214517e323f28c7aa1345db881ded` |
| accepted GGUF SHA-256 | `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| block-0 adjudication binding | `58f661d9461f3dc17b2a52f830dcc9f36423621528941d475e23ee2aed8e0bf2` |
| block-0 manifest binding | `7b8de95a3766e4f76bde37f4038ece2986b9067ad9ac7a0e4cc11855a3062497` |
| reference producer invocations / graph schedules | `0 / 0` |
| C producer invocations / graph schedules | `0 / 0` |

The directories without a suffix and with `repair1` are preserved historical
apparatus runs. Neither executed a producer. `repair2` is canonical because it
adds enum-derived canonical callback types and durable per-arm completion
markers whose counts survive a producer failure.

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

Launch at most one changed-coordinate scientific rerun from source HEAD
`4c5d1af6274d1067076485398d58ce67dfc1ab96`. That rerun has now completed and
is closed as [Rung 2C FAIL](STRAT_01_GIGACHAT31_ENGINE_RUNG2C_RESULT_20260923.md).
Do not repeat either producer.
