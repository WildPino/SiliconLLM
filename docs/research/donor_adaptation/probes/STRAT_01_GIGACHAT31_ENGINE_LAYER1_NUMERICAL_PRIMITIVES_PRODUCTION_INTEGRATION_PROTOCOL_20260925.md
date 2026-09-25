# STRAT-01 layer-1 numerical-primitives production-integration protocol

**FROZEN BEFORE IMPLEMENTATION OR EXECUTION:** 2026-09-25

**Cell:** `STRAT-01-ENGINE-LAYER1-NUMERICAL-PRIMITIVES-PRODUCTION-INTEGRATION`

## Question and non-duplication

The closed [compile-parity result](STRAT_01_GIGACHAT31_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_COMPILE_PARITY_RESULT_20260925.md)
proves exact local replacements for the first remaining router residual and
both routed/shared SwiGLU residuals. It does not prove that the standard
`--strat01-gguf-rung2c` path invokes them or that their composition remains
exact through Q6, weighting, reduction and terminal addition.

Question: does installing only those independently qualified primitives in
normal Rung-2C production make the accepted artifact byte-exact to the
immutable two-schedule reference through every block-0 and layer-1 checkpoint?

Do not rerun compile parity, a cross-input cell, Q4/Q6/F16 work, any local
alternative, or a reference graph.

## Immutable bindings

- accepted GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- local-primitives adjudication SHA-256
  `ac04b6daeadb0ae6e71d19746585cf04fffeb74620217ed211e56484578b045a`;
- Q6 production-integration recovery adjudication SHA-256
  `cb2a2d9cbfc80d8eb08dfc32d0f850fae717cf9a76525a6f81a9f600269a3b93`;
- local-primitives execution commit
  `2923b78371ffff0c166bf6e0f6ae27eefa39ee8b`;
- pinned llama.cpp revision
  `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`;
- immutable Rung-2C repair-1 reference tree, tokens
  `[1,72,14,14129,14,2135,1512,2015]`, positions
  `[0,1,2,3,4,5,6,7]`, schedules `prefill8` and `cached7p1`.

## Changed coordinates

Only three normal Rung-2C numerical sites may change:

1. F32 router rows must delegate to the qualified non-inlined
   `no-avx,no-avx2,no-fma` helper implementing F32 product, source-order F64
   accumulation and one final F32 conversion;
2. each routed 1280-value gate/up slice must use the already qualified
   four-lane SSE2/no-FMA SwiGLU helper;
3. the 10,240-value shared gate/up population must use that same helper.

The helper calls must fail closed. Q4, Q5, Q6, F16, RMSNorm, attention,
sigmoid/bias/top-4 routing, normalized weights, expert selection, down
projections, routed reduction, residual order, caches, tensor layouts and the
artifact remain unchanged. Normal Rung-2C `CONFIG` must declare both primitive
implementations, and validators must bind the complete appended configuration
rather than an obsolete literal prefix.

## Model-free apparatus gate

Before opening the artifact, prove that:

- production router and diagnostic candidate have one shared implementation,
  while the independent oracle remains a separate translation unit;
- production routed/shared SwiGLU and the closed diagnostic use the same
  qualified helper, with no scalar production body remaining at those sites;
- exact synthetic candidate/oracle and SSE2 fixtures pass, including widths
  1280 and 1536 where applicable;
- float-accumulator, promoted-product, scalar-SwiGLU, mutation, swap,
  short-read, invalid-count, ownership and source controls fire;
- Q6 reference-generic dispatch and all unchanged operator selections remain
  intact;
- complete registered Python regression and all relevant C self-tests pass;
- model, payload, diagnostic, producer, donor-graph and reference-graph
  counters are all zero.

Only a qualified committed apparatus authorizes production.

## Scientific execution and gates

Run exactly one normal producer command:

```text
engine --strat01-gguf-rung2c <accepted.gguf> --out-dir <candidate>
```

It must complete exactly the `prefill8` and `cached7p1` schedules and no
reference graph. Validate source identities, full configuration, schema,
tokens, positions, tensor types/shapes/order, finiteness, helper ownership,
execution counts, causal controls, route equality, cache accounting and
schedule continuity.

The decisive gates are byte-exact equality to immutable reference for:

- `l_out-0` in both schedules;
- all 32 layer-1 checkpoints in each schedule, including router logits,
  probabilities, ordered top-4 IDs, normalized weights, routed/shared
  up/gate/SwiGLU/down/output and terminal `l_out-1`;
- all six cache comparisons and every prefill/cached token-7 continuity
  comparison.

The old scalar hashes must be displaced by the three closed primitive hashes:
router `8188ff0d...a0bbf5`, routed SwiGLU `5097dc85...e29e8`, and shared
SwiGLU `fa0d1b4f...64c6003`. The production report and sources must prove that
normal production, not a diagnostic arm, emitted them.

## Decision and stop rule

- `PASS_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_PRODUCTION_INTEGRATION` only if all
  apparatus, identity, accounting, source, exact checkpoint, continuity,
  routing, cache and causal gates pass;
- `FAIL_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_PRODUCTION_INTEGRATION` if the run
  is valid but any scientific exactness gate fails; report the first boundary
  and open no later coordinate;
- `VOID_ENGINE_LAYER1_NUMERICAL_PRIMITIVES_PRODUCTION_INTEGRATION` for any
  build, identity, schema, source, control, finiteness or accounting failure.

Exactly one non-VOID producer invocation is allowed. A VOID may be repaired
model-free or recovered offline only if preserved immutable outputs suffice;
it never silently authorizes another producer.

This cell establishes fidelity only on the frozen two-schedule trace. It does
not establish tokenizer/logits, free generation, task quality, RAM or accepted
token throughput. Timing is inadmissible and must not update
`SPEED_LEDGER.md`.

## Apparatus repair addendum

**Frozen after apparatus VOID 1 and before repair execution: 2026-09-25.**
The first model-free apparatus compiled successfully and ran 311 Python tests,
but four historical assertions required layer-1 SwiGLU to remain scalar and
therefore rejected the exact coordinate this protocol preregistered. No model,
payload, diagnostic, producer or graph ran.

One apparatus repair is authorized. It may change only those four historical
assertions so that they continue to require every older Q6, block-0 SwiGLU and
layer-2 control while explicitly recognizing the new layer-1 router/SwiGLU
production integration. It must not weaken the new cell's own positive source
controls or alter production arithmetic. Preserve the original apparatus
directory as VOID and use a distinct `apparatus_repair1` directory. Scientific
execution remains unauthorized until the complete repaired suite and all C
self-tests pass with every execution counter at zero.
