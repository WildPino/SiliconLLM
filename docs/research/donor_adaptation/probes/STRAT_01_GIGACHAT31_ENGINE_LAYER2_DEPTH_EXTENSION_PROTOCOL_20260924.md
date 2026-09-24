# STRAT-01 layer-2 depth-extension protocol

**FROZEN BEFORE IMPLEMENTATION OR EXECUTION:** 2026-09-24

**Cell:** `STRAT-01-ENGINE-LAYER2-DEPTH-EXTENSION`

## Question and non-duplication

Can the accepted C-engine path consume its already accepted `l_out-1` and
execute complete GigaChat layer 2 within the unchanged Rung-2C numerical
gates for both frozen schedules?

The nearest prior cell is
[`PASS_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION`](STRAT_01_GIGACHAT31_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION_RESULT_20260924.md),
which closes all 32 checkpoints through `l_out-1` but contains no layer-2
weights, cache, checkpoint, or reference payload. The changed coordinate is
depth only: add `blk.2` execution and evidence. Layer 0, layer 1, every
accepted operator implementation, token, position, schedule, threshold, and
ordering rule remain fixed.

Do not rerun or alter Q4/F16/SwiGLU/Q6 diagnostics, cross-input splits,
block-0 integration, layer-1 integration, AVX2/FMA variants, tokenizer,
quality, or rate cells.

## Immutable bindings

| object | identity |
|---|---|
| accepted GGUF | 6,474,702,976 bytes; SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| accepted engine producer commit | `cb0f5260044bc209175edd324e1580db272370a8` |
| accepted production adjudication | SHA-256 `d543c56d7e463cfec23e4fa3a7ec7fd5bfecbdccfdf84351ddf12085fcc3cc94` |
| pinned llama.cpp | `5b335f413e4f73b0809c4fe39af894efbcc6a0d2` |
| tokens | `[1,72,14,14129,14,2135,1512,2015]` |
| positions | `[0,1,2,3,4,5,6,7]` |
| schedules | `prefill8` and `cached7p1` |
| model depth | 26 blocks; one leading dense block |

The [static audit](../audits/STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_AUDIT_20260924.md)
is the descriptor and no-duplication record for this cell.

## Changed coordinate

Add a new Rung-2D-style command and evidence schema that:

1. executes the accepted block 0 and layer 1 without changing their source
   semantics;
2. copies accepted `l_out-1` into a fresh layer-2 attention arm;
3. resolves the sixteen `blk.2.*` tensors under the same exact type/shape
   contract as layer 1;
4. reuses the accepted attention and MoE compute primitives on those tensor
   pointers;
5. emits `l_out-1` plus the complete 31-checkpoint layer-2 surface, and
   cache witnesses for layers 0, 1, and 2;
6. declares layer 2 and the unchanged layer-1 semantics in `CONFIG`.

Do not change layer-1 SwiGLU, thresholds, helper arithmetic, residual order,
or routing in this cell. Combining such a change with added depth would make
the first failure causally ambiguous.

## Reference extension

Create a separate pinned-reference build variant; do not modify or overwrite
the immutable Rung-2C reference tree. Its callback surface must contain
`l_out-1`, all layer-2 logical checkpoints, and `Kcur-0/1/2` cache witnesses.
It must retain explicit operation-qualified selection, logical-rank rules,
prefix7/final1 composition, F16 cache round-trip witnesses, and source/build
identity.

After apparatus qualification and an exact-source commit, exactly one
reference invocation may execute both schedules, producing two reference
graphs. The new reference must reproduce the old immutable `l_out-1` and
layer-0/1 cache witnesses within their frozen gates before it can adjudicate
layer 2. No historical reference output may be regenerated or overwritten.

## Apparatus gate

Before either accepted-artifact producer runs, an apparatus-only invocation
must establish:

- exact `blk.2` tensor names, types, ranks, shapes, and count;
- unchanged block-0 and layer-1 source semantics;
- a separate command, output schema, directories, and reference build
  variant that cannot overwrite Rung-2C;
- complete 32-checkpoint layer-2 and three-layer cache schemas for both
  schedules;
- expected helper totals of `6912` QK and `786432` value invocations;
- fail-closed source, protocol, predecessor, artifact, compiler, manifest,
  shape/type/order, finiteness, count, and path-containment controls;
- full STRAT-01 Python regression and every registered C self-test;
- zero GGUF access and zero donor/reference graph executions.

Only a qualified, committed apparatus authorizes the scientific run.

## Scientific execution and gates

Run exactly one pinned-reference invocation and one C-engine invocation. Each
must complete exactly `prefill8` and `cached7p1`, for two reference graphs and
two C graphs total. Neither producer may be rerun after a non-VOID result.

For each schedule, compare `l_out-1` and every layer-2 checkpoint using the
existing Rung-2C gates unchanged:

- general F32 checkpoints: NRMSE `<= 0.002`, normalized maximum `<= 0.01`;
- `ffn_inp-2` and `l_out-2`: NRMSE `<= 0.001`, normalized maximum `<= 0.005`;
- `ffn_moe_topk-2`: exact I32 equality;
- prefill/cached token-7 continuity: NRMSE `<= 2e-6` and normalized maximum
  `<= 1e-5`, or exact equality for I32 routing;
- all nine cache comparisons: NRMSE `<= 0.002`, normalized maximum `<= 0.01`.

Require exact schedule accounting, the frozen token/position vectors, finite
payloads, path containment, accepted artifact identity, committed source
identity, and helper counts `6912` / `786432`. Reuse the ten Rung-2C causal
controls with layer-2 targets; every mutation must reject.

## Decision rule

- **`PASS_ENGINE_LAYER2_DEPTH_EXTENSION`** if apparatus, reference lineage,
  predecessor consistency, all layer-2 checkpoint/continuity/cache/route
  gates, helper counts, identities, controls, and execution counts pass.
- **`FAIL_ENGINE_LAYER2_DEPTH_EXTENSION`** if execution is otherwise valid
  but a numerical gate fails. Report the first failed layer-2 boundary and
  do not edit an operator in the same cell.
- **`VOID_ENGINE_LAYER2_DEPTH_EXTENSION`** for any apparatus, binding,
  source, build, schema, identity, count, mutation, finiteness, path,
  predecessor-consistency, or execution-accounting failure.

A VOID may be repaired only model-free unless the immutable outputs prove
that no producer rerun is required.

## Non-claims and stop rule

This cell can establish fidelity only through layer 2 on one frozen
eight-token trace. It does not establish layers 3–25, output norm/head,
tokenizer, logits, generation, quality, RAM, or accepted-token throughput.
Producer timing is inadmissible and must not enter `SPEED_LEDGER.md`.

If PASS, document before deciding whether to generalize repeated MoE depth.
If FAIL, freeze a separate diagnostic at the earliest failed checkpoint.
