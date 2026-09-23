# STRAT-01 exact F16 vector-reduction propagation protocol

**Frozen:** 2026-09-23, before implementation or execution

**Cell:** `STRAT-01-ENGINE-F16-VECTOR-REDUCTION-PROPAGATION`

**Purpose:** determine whether the pinned `ggml_vec_dot_f16` reduction
semantics remove the small layer-1 attention residual that is locally
inside its gate but is now proven sufficient to break the routed and shared
FFN down boundaries.

## Pre-donor implementation addendum (2026-09-23)

The first Stage-A transcription attempt falsified the protocol's original
mechanistic label, "AVX/FMA reduction order", before any accepted-model graph
was executed. The historical helper's `compile_commands.json` shows that the
pinned `ggml-cpu/vec.cpp` was compiled with `-DGGML_CPU_GENERIC` and without
AVX flags. In that build, `ggml_vec_dot_f16` follows its non-SIMD branch:
each binary16 product is formed in F32, accumulated sequentially in
`ggml_float` (F64), and converted to F32 at the output. An independent
F16-input/F64-accumulation reconstruction reproduced both frozen pinned
hashes exactly; the attempted AVX/FMA transcription reproduced neither.

This addendum corrects the mechanism while preserving the pre-registered
estimand, immutable inputs, required hashes, controls, thresholds, graph
budget, and verdict rules. In the rest of this protocol, "pinned vector
reduction" means the exact reduction semantics reached through the pinned
CPU type trait in the historical helper build, namely `pinned-generic-f64`.

## Why this is new and not a repetition

The closed [F16 vector-dot diagnostic](STRAT_01_GIGACHAT31_ENGINE_F16_VEC_DOT_DIAGNOSTIC_RESULT_20260921.md)
showed two facts on immutable block-0 attention payloads: scalar F16×F16
accumulation passed the then-local tolerance, while the pinned vector
reduction was byte-exact. Production therefore retained scalar accumulation.

The closed [layer-1 Q6 cross-input](STRAT_01_GIGACHAT31_ENGINE_LAYER1_Q6_CROSS_INPUT_RESULT_20260923.md)
changes the estimand. It proves that the remaining locally passing SwiGLU
residual is amplified into routed/shared Q6 failures, confined to tokens 4
and 5. The predecessor propagation has `kqv_out-1` NRMSE `0.000353432`,
`ffn_norm-1` `0.000353176`, routed/shared SwiGLU `0.00149066`/`0.00134965`,
and routed/shared down `0.00385395`/`0.00288634`. A local attention PASS is no
longer sufficient evidence for downstream fidelity.

This cell changes only F16 dot reduction order. It does not repeat F16
conversion, softmax, Q4, SwiGLU, Q6, routing, cache continuity, or a reference
producer.

## Frozen predecessors and identities

- Accepted GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Pinned llama.cpp commit:
  `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.
- F16 vector-dot adjudication SHA-256:
  `d9691cbcd569681f970e5e4af553cec75a7f86ec47f66acbec2a111288239e0a`.
- Reference-generic propagation adjudication SHA-256:
  `79a19806fcf80df3e4d8a84335eaaecfc7beb9aef543f021dd1da01f6558b9c3`.
- Layer-1 Q6 cross-input adjudication SHA-256:
  `db9b478e208c35460d4bd1ae54eb56cfb2a8106361b19f0b7f1f024699509148`.
- Candidate propagation `prefill8` manifest SHA-256:
  `517470cef63b1dd640d24c3fc0743a327d7dc160fad4ba3917148cb5c554190c`.
- Pinned-reference Rung-2C `prefill8` manifest SHA-256:
  `d7506adfd7cb20a54da2d406446c5c261452ecf0eecaa3663ef6a4a7acca1451`.

The runner must validate every referenced manifest payload by path
containment, size, SHA-256, dtype, shape, and finiteness before accepting any
result.

## Changed coordinate

Add an isolated C11 helper that reproduces the pinned historical
`ggml_vec_dot_f16` operation order:

- binary16 operands converted exactly to F32;
- each F32 product accumulated sequentially into F64;
- one terminal conversion from F64 to F32.

The production attention path must use this helper for both query×key and
softmax-probability×value reductions. The accepted exact F32→F16 converter,
causal masking, scaling, `expf` softmax, cache representation, V-B, Q4/Q6,
router, and all other operations remain unchanged. The old scalar F16×F16
path remains a named diagnostic control.

## Stage A: mandatory model-free identity gate

Before any accepted-model graph, run the new C helper on the immutable inputs
already used by the F16 vector-dot diagnostic. It must reproduce exactly:

| output | bytes | required SHA-256 |
|---|---:|---|
| pinned-vector QK | 8,192 | `121c689214a7bcf2e709b8de896df14aabcb8fb00580bab297231b8887a3b009` |
| pinned-vector value reduction | 524,288 | `541183b4eb5992cfcfc711fe3c9f179ab41685106ba9cfe2aa7099f45f821312` |
| exact project F16 conversion stream | 435,240 | `7cdec0744fb5c9dcc6225dd70a0c952283128d4f8668d9861131b868379defbc` |

The retained scalar controls must reproduce their distinct historical hashes:

- QK scalar:
  `5b4caf1ebab46a5f6340dc695a72f8f08145f5f8817081e7cec10bceedaaae1d`;
- value scalar:
  `b77004cfaeb0c96588df64d79e9df091e7a3b75273475f610105d106f9d56bbb`.

At least one legal operand mutation must change each vector output and reject
the tight `2e-6` NRMSE / `1e-5` normalized-maximum gate. A failure, accidental
equality with a scalar control, missing source binding, or any graph execution
makes the whole cell VOID. Stage B is forbidden unless Stage A passes.

## Stage B: one production propagation

After apparatus qualification and a committed implementation, execute the
accepted C producer once, producing both `prefill8` and `cached7p1`. Execute
zero pinned-reference graphs. Reuse the immutable Rung-2C reference payloads
and all gates from the reference-generic propagation cell.

Mandatory reporting:

- every block-0 terminal and layer-1 checkpoint from the predecessor cell;
- old versus new NRMSE and normalized maximum for `kqv_out-1`, `ffn_inp-1`,
  `ffn_norm-1`, both up/gate/SwiGLU/down branches, `ffn_out-1`, and `l_out-1`;
- descriptive per-token values for those boundaries, especially tokens 4 and
  5;
- exact top-4 expert IDs and route weights;
- prefill/cached continuity and all cache checkpoints;
- vector-helper invocation counts for QK and value reduction;
- the old scalar candidate hashes as immutable controls;
- donor/reference producer and graph counts.

The changed coordinate must reach both schedules: vector-helper invocation
counts must be nonzero and at least one attention/downstream candidate hash
must differ from the predecessor. All existing negative controls remain live.
No tolerance may be relaxed.

## Verdicts and stop rule

- `PASS_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION` if Stage A is exact and all
  frozen Stage-B checkpoint, cache, continuity, route, identity, and causal
  gates pass.
- `FAIL_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION` if Stage A is exact, all
  apparatus and causal controls pass, the intervention reaches both graphs,
  but one or more numerical Stage-B gates fail.
- `VOID_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION` for any identity, exact
  replay, build, source, mutation, accounting, intervention, or control
  failure.

Every non-VOID result closes this exact reduction-order cell. A PASS permits
a separately frozen later-layer expansion. A FAIL must localize the earliest
remaining checkpoint and cannot justify another broad graph run. Neither
outcome establishes tokenizer, logits, generation, quality, RAM, or rate, and
neither updates `SPEED_LEDGER.md`.

Planned raw directories:

- apparatus:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_f16_vector_propagation_apparatus_20260923/`;
- scientific:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_f16_vector_propagation_20260923/`.
