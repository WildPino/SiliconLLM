# STRAT-01 GigaChat 3.1 Rung-2C attention cross-input diagnostic protocol

**Frozen:** 2026-09-23, before implementation or adjudication

**Cell:** `STRAT-01-ENGINE-RUNG2C-ATTENTION-CROSS-INPUT-DIAGNOSTIC`

**Purpose:** explain the first valid Rung-2C failure without rerunning either
donor producer.

## Prior result and changed estimand

[Rung 2C](STRAT_01_GIGACHAT31_ENGINE_RUNG2C_RESULT_20260923.md) is closed as
`FAIL_ENGINE_RUNG2C`. In both schedules, `Qcur-1`, `Kcur-1`, and `Vcur-1`
pass, while the first failed checkpoint is `kqv_out-1`. Continuity is exact,
the cache checks pass, and the discrete router and route weights pass. The two
producer traces are immutable and must not be repeated.

This diagnostic changes one coordinate: the existing C causal-attention plus
V-B operator receives the already captured C or pinned-reference query and KV
boundary tensors in all four combinations. It asks whether exact reference
inputs pass through the C operator and, if they do, whether the admitted
query residual, KV residual, or their joint interaction is sufficient to
cross the frozen `kqv_out-1` gate.

It executes no embedding, block 0, layer-1 Q/KV projections, output
projection, residual, MoE, logits, tokenizer, or generation graph.

## Immutable evidence and inputs

Raw inputs come only from the completed producer directory:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_repair1_20260923/`

The recovered Rung-2C adjudication has SHA-256
`3742bc5982dd47be36a8e422f7b79c9e6716dc628153e44e6f9e90553c9ba62c`;
its run manifest has SHA-256
`898f614a8baaf40716a7aab66e4e31ba51c92da84e133ff24a55e7ccafa60dd4`.
The C prefill manifest has SHA-256
`bdaa49fb9493f891f66fa65c48dd703bae7636b35fd173f54c135326355923c3`;
the reference prefill manifest has SHA-256
`d7506adfd7cb20a54da2d406446c5c261452ecf0eecaa3663ef6a4a7acca1451`.

Only `prefill8` is read. Rung 2C proved exact prefill-versus-7+1 continuity and
the same first-failure surface in both arms, so reading `cached7p1` would
duplicate the estimand rather than add an independent schedule test.

| producer | object | logical shape | bytes | SHA-256 |
|---|---|---:|---:|---|
| C | `Qcur-1` | `[576,32,8]` | 589,824 | `3faa0a37833c9d9b9cf92d3bf985d13bf6f09b9c278f81e8813258e86e7d2b2f` |
| C | `Kcur-1` | `[576,1,8]` | 18,432 | `febfcaa2ca7192a71b7ffca1af26252c9e81b010068478928ed6d56fa78fb0d4` |
| C | `Vcur-1` | `[512,1,8]` | 16,384 | `58fa280015b838d47df9b1a0ba9064d6cc8eb25000e454330d988ec48d537b57` |
| C | `kqv_out-1` | `[6144,8]` | 196,608 | `7e6d44661f21dee62b449b44e1939b59d8ad0799c4ccf6779d77a01a48e3ee41` |
| reference | `Qcur-1.full` | `[576,32,8]` | 589,824 | `9cb50fb41e0d33549eb53bd4ae19605705331562d36f975bb2ac82cc4c41cb16` |
| reference | `Kcur-1.full` | `[576,1,8]` | 18,432 | `73febde321c2ea45a56a5d27a3605b820e08f8b038420a379afc4b9c3f0c72b4` |
| reference | `Vcur-1.full` | `[512,1,8]` | 16,384 | `3f3c00b3fe008aecda5e1164b5da2b295d349270b1e339831da1544a849a9387` |
| reference | `kqv_out-1.full` | `[6144,8]` | 196,608 | `fa1006c4e2c365d3a5540baebbe4c88180f524d5f69c610fcc3b4205a856d489` |

The accepted GGUF is 6,474,702,976 bytes with SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
The only admitted weight is `blk.1.attn_v_b.weight`, Q4_K with logical shape
`[512,192,32]`. The executable must report and the runner must bind its parsed
descriptor, including offset, file offset, and span.

For each producer, `Vcur-1` is byte-identical to the first 512 F32 components
of every corresponding 576-component `Kcur-1` row. The runner must revalidate
this relation before execution. There is no separate V cache in this MLA
contract; a synthetic independent V arm is forbidden.

Every path, byte count, digest, type, shape, payload order, model identity,
manifest identity, and source hash is fail-closed.

## Frozen operator and four arms

The diagnostic must reuse, without semantic edits:

1. `strat01_r2a_cache_write`, which rounds each 576-component KV row to F16;
2. `strat01_r2a_attend_one`, including F16-rounded query, scalar F16×F16
   accumulation, pinned scale, stable softmax, F16-rounded probabilities, and
   the first 512 cached components as values;
3. `strat01_r2a_vb_batch`, including the production Q4_K×Q8_K V-B path and
   token/head/output layout.

The four outputs are:

| arm | query | KV/cache and value | role |
|---|---|---|---|
| `c_q__c_kv` | C | C | native replay |
| `ref_q__ref_kv` | reference | reference | exact-input operator test |
| `c_q__ref_kv` | C | reference | query-side residual test |
| `ref_q__c_kv` | reference | C | KV-side residual test |

All arms write exactly 196,608 F32LE bytes in token-major order. The command
may self-certify only `OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION`; it may
never emit a scientific PASS or causal verdict.

Compile with Clang C11, `-O3 -mavx2 -mfma`, no fast-math, and the existing
FP-contraction-off source contract. The diagnostic records zero donor graph
executions and no timing or rate measurement.

## External adjudication and gates

Before interpreting any arm, the runner must establish:

- `c_q__c_kv` is byte-identical to the captured C `kqv_out-1` target;
- all four outputs are finite and have the required size and digest record;
- the two V↔K-prefix equalities hold;
- the accepted model and both source manifests retain their pinned identities;
- the executable report binds the engine and diagnostic source hashes.

Replay identity uses exact bytes. Any optional numerical replay report uses
NRMSE `<= 2e-6` and normalized maximum `<= 1e-5`, but cannot replace the exact
hash gate.

Each of the four outputs is also compared with the captured reference
`kqv_out-1.full` using the unchanged Rung-2C general limits: NRMSE `<= 0.002`
and normalized maximum `<= 0.01`. Both limits must pass; no token, head, or arm
average may rescue a failure. Per-token metrics are descriptive and must be
reported to expose concentration hidden by the aggregate.

The runner must recompute the accepted C-versus-reference metrics and agree
with the frozen Rung-2C values within `1e-12`; otherwise the cell is VOID.

## Causal and apparatus controls

All controls are frozen before observing outputs:

1. a one-byte mutation of each of the six input payloads must fail identity
   validation before operator execution;
2. substituting block-0 or a descriptor-incompatible tensor for
   `blk.1.attn_v_b.weight` must be refused;
3. an internally generated control that negates all 32 query heads of token 7
   in the reference/reference arm must fail at least one reference-target
   gate;
4. an internally generated control that swaps complete reference KV rows 0
   and 7 before F16 cache write must fail at least one reference-target gate;
5. swapping the two mixed-arm labels in a copied report must be detected by
   output identity/provenance validation.

A model-free self-test must cover F32 input decoding, V↔K-prefix checking,
four-arm routing, cache reset between arms, query mutation, KV-row swap,
descriptor refusal, and report/output path separation.

## Decision rule

Apply the following rules in order:

1. **`VOID_RUNG2C_ATTENTION_CROSS_INPUT`** if any identity, native replay,
   source binding, self-test, or planted-control requirement fails. It is also
   VOID if `c_q__c_kv` no longer reproduces the frozen Rung-2C failure against
   the same reference target.
2. **`EXACT_REFERENCE_FAILS_ATTENTION_VB_OPERATOR`** if
   `ref_q__ref_kv` fails. The composed C attention/V-B operator is not
   sufficiently faithful on the exact boundary; mixed arms are descriptive
   only and cannot attribute upstream residual amplification. Freeze a
   narrower attention-versus-V-B boundary diagnostic before repair.
3. If `ref_q__ref_kv` passes and `c_q__ref_kv` alone fails, emit
   **`QUERY_RESIDUAL_SUFFICIENT`**.
4. If `ref_q__ref_kv` passes and `ref_q__c_kv` alone fails, emit
   **`KV_RESIDUAL_SUFFICIENT`**.
5. If `ref_q__ref_kv` passes and both mixed arms fail, emit
   **`QUERY_AND_KV_RESIDUALS_INDEPENDENTLY_SUFFICIENT`**.
6. If `ref_q__ref_kv` and both mixed arms pass while `c_q__c_kv` fails, emit
   **`JOINT_QUERY_KV_INTERACTION_SUFFICIENT`**.

No other non-VOID surface is admissible. In particular, all four arms passing
would contradict the exact native replay plus frozen Rung-2C target and is
therefore an apparatus inconsistency, not a success.

Exactly one non-VOID execution is allowed. Gates must not be changed after
values are observed.

## Non-claims and stop rule

This cell cannot repair or promote Rung 2C. It makes no independent claim
about output projection, residual, MoE experts, routing, later layers,
tokenizer, logits, generation quality, RAM, or rate. It cannot update
`SPEED_LEDGER.md`.

After a non-VOID result, document and index it before changing production
semantics. A production repair is authorized only by the narrowest boundary
identified by this decision tree; rerunning either Rung-2C producer is not
authorized.
