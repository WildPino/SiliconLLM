# STRAT-01 reference-generic Q4 downstream-propagation protocol

**Frozen:** 2026-09-23, before implementation or execution

**Cell:** `STRAT-01-ENGINE-REFERENCE-GENERIC-PROPAGATION`

**Purpose:** determine whether the newly exact reference-generic Q4_K×Q8_K
production helper closes the already measured block-0 terminal and Rung-2C
residuals when propagated through the unchanged production graph.

## No-duplication checklist

The closest predecessor is
[`PASS_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY`](STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_RESULT_20260923.md).
It proves byte-exact block-0 Q and KV projections on immutable
`attn_norm-0`, but explicitly does not prove downstream propagation.

The original [Rung-2C result](STRAT_01_GIGACHAT31_ENGINE_RUNG2C_RESULT_20260923.md)
is a valid FAIL from the historical AVX-translation-unit generic helper. Its
two producer graphs must not be repeated as the same cell. Subsequent
cross-input diagnostics show that exact block-0 `l_out-0` makes layer 1 pass,
that the old C `ffn_out-0` residual is sufficient for failure, and that both
gate and up Q4 projections independently contribute. Repeating those splits
would duplicate evidence.

This cell changes one implementation coordinate:

> historical generic Q4_K×Q8_K reduction lowered in the AVX2/FMA translation
> unit → the accepted, isolated baseline-generic lowering for every production
> Q4_K matrix call.

Weights, packed formats, Q8_K quantization, model, tokens, positions,
schedules, all non-Q4 operators, graph topology, comparison metrics, and gates
remain unchanged. This is therefore a changed-coordinate propagation
confirmation, not a rerun of Q/KV parity, Rung 2C, or any cross-input split.

## Falsifiable hypothesis

The exact baseline-generic Q4 arithmetic is sufficient to bring every frozen
block-0 and layer-1 Rung-2C checkpoint inside its original numerical gate in
both `prefill8` and `cached7p1`, while preserving exact/accepted schedule
continuity, cache parity, router selection, and causal controls.

The hypothesis fails if the apparatus is valid but any frozen checkpoint,
cache, continuity, or exact-I32 gate remains outside its original limit.

## Immutable identities

- Accepted GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Reference-generic parity adjudication SHA-256:
  `f0371d9c98e762e6434907ea5c386a7b622adac30dae9939a101fdc22df995f1`.
- Original canonical Rung-2C adjudication SHA-256:
  `3742bc5982dd47be36a8e422f7b79c9e6716dc628153e44e6f9e90553c9ba62c`.
- Immutable Rung-2C reference root manifest SHA-256:
  `9a00ea08a47482323055a7fa8fcfa6c0e079ea2ed4957665bd491c1abfec21eb`.
- Immutable reference arm manifests:
  `prefill8` =
  `d7506adfd7cb20a54da2d406446c5c261452ecf0eecaa3663ef6a4a7acca1451`;
  `cached7p1` =
  `ad214e8d38ee51a3b0f0629bdadf7cc692f6806499e1347c90d46aba937a39b9`.
- Frozen tokens: `1,72,14,14129,14,2135,1512,2015`; positions `0..7`.
- Original C `l_out-0` SHA-256:
  `258232509011378e8470ce6c03cffd51e927f28bfc1ea4f937af07bda9bf0f44`.
- Reference `l_out-0` SHA-256:
  `385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa`.

The reference payloads are reused from
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_repair1_20260923/pinned_reference/`.
No reference graph may run.

## Apparatus and execution contract

Before the scientific execution, an apparatus-only run must:

1. hash-bind and validate both predecessor adjudications and all three frozen
   reference manifests;
2. validate every reused reference tensor/cache payload through the existing
   Rung-2C schema and digest checks;
3. prove from source inventory that production dispatches to the accepted
   reference-generic helper, while historical generic and active AVX2 remain
   named diagnostic controls;
4. compile the ordinary Clang C11 `-O3 -mavx2 -mfma` engine without
   fast-math;
5. pass the Q4 model-free parity tests, Rung-2C tests, every registered C
   self-test, and the 73,024-check legacy suite;
6. execute zero donor and zero reference graphs.

After the apparatus and implementation are committed, exactly one scientific
C producer invocation is permitted. It must run both frozen schedules through
`--strat01-gguf-rung2c`, emit the normal 32 checkpoints per schedule and cache
witnesses, then compare them against the reused immutable reference payloads.

## Gates and controls

All original Rung-2C limits remain unchanged:

- general F32 checkpoints: NRMSE `<= 0.002`, normalized maximum `<= 0.01`;
- terminal `ffn_inp-1` and `l_out-1`: NRMSE `<= 0.001`, normalized maximum
  `<= 0.005`;
- `ffn_moe_topk-1`: exact I32 equality;
- prefill-versus-cached token-7 continuity: original strict continuity limits;
- all six cache comparisons: original general limits.

Mandatory causal and apparatus controls:

- both candidate schedules have one shared new `l_out-0` hash;
- that hash differs from the original C `l_out-0`, proving the changed
  coordinate reached the graph;
- the immutable reference `l_out-0` hash remains exact;
- all ten original Rung-2C negative controls reject;
- router IDs remain exact and no checkpoint, shape, operation, payload order,
  or configuration field disappears;
- the C producer emits exactly two unique graph-completion markers;
- donor graph executions equal two and reference graph executions equal zero;
- source and artifact state equal the committed apparatus state.

## Decision and stop rule

- `PASS_ENGINE_REFERENCE_GENERIC_PROPAGATION` only if every checkpoint,
  continuity, cache, identity, intervention, and negative-control gate passes.
- `FAIL_ENGINE_REFERENCE_GENERIC_PROPAGATION` for a valid run with at least
  one remaining numerical or exact-I32 failure.
- `VOID_ENGINE_REFERENCE_GENERIC_PROPAGATION` for an apparatus, identity,
  source, intervention, graph-count, manifest, or control failure.

PASS authorizes a separately frozen next-layer expansion; it does not prove
full-model logits, generation, quality, RAM, or rate. A valid FAIL closes this
global-Q4 propagation cell and makes its earliest failing dependency the next
diagnostic boundary. VOID authorizes only repair of the failed apparatus
coordinate. No outcome changes `SPEED_LEDGER.md`.

Planned raw directories:

- apparatus:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_reference_generic_propagation_apparatus_20260923/`;
- scientific:
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_reference_generic_propagation_20260923/`.
