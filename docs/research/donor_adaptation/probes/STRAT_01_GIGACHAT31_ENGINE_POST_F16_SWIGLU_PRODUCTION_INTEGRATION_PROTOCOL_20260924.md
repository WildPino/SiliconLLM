# STRAT-01 post-F16 SwiGLU production-integration protocol

**FROZEN BEFORE IMPLEMENTATION OR EXECUTION:** 2026-09-24

**Cell:** `STRAT-01-ENGINE-POST-F16-SWIGLU-PRODUCTION-INTEGRATION`

## Question and non-duplication

Does installing the already proven no-FMA four-lane SSE2 SwiGLU semantics in
the shared block-0 production primitive make the ordinary accepted-artifact
`--strat01-gguf-rung2c` path pass block 0 and all 32 complete layer-1
checkpoints in both frozen schedules?

The closed
[SSE2-semantics result](STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_SWIGLU_SSE2_SEMANTICS_RESULT_20260924.md)
establishes `POST_F16_BLOCK0_SWIGLU_SSE2_EXACT_REPAIR` on immutable reference
gate/up operands: the candidate is byte-exact to captured reference SwiGLU and
passes the complete downstream layer-1 chain. It does not establish that the
ordinary Rung-2B/Rung-2C path calls that primitive on its own computed
operands. That production installation and end-to-end propagation are the
only new coordinates here.

Do not rerun any cross-input cell, reference producer, Q4/F16 diagnostic,
gate/up split, Q6 split, AVX2/FMA sweep, or isolated SSE2 diagnostic.

## Immutable bindings

| object | identity |
|---|---|
| accepted GGUF | 6,474,702,976 bytes; SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| pinned llama.cpp | `5b335f413e4f73b0809c4fe39af894efbcc6a0d2` |
| SSE2-semantics adjudication | `46f127215e4bfc00b80ba4eb151d20c294482d2cb5f3c976616e953d21852c0e` |
| post-F16 propagation adjudication | `512e3ea7754c5e8fa067dab5692a8a192879be26895547cbedbb6023a896edcf` |
| reference root manifest | `9a00ea08a47482323055a7fa8fcfa6c0e079ea2ed4957665bd491c1abfec21eb` |
| reference prefill manifest | `d7506adfd7cb20a54da2d406446c5c261452ecf0eecaa3663ef6a4a7acca1451` |
| reference cached manifest | `ad214e8d38ee51a3b0f0629bdadf7cc692f6806499e1347c90d46aba937a39b9` |
| scalar predecessor `l_out-0` | `7fb5fe52df684e4af82413004f6845133528241e3b030530d41595f74298bdc4` |
| tokens | `[1,72,14,14129,14,2135,1512,2015]` |
| positions | `[0,1,2,3,4,5,6,7]` |
| schedules | `prefill8` and `cached7p1` |

The immutable reference tree is
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_repair1_20260923/pinned_reference/`.
It may be read and hash-validated but never regenerated in this cell.

## Changed coordinate

The production change is limited to these operations:

1. factor the exact no-FMA four-lane exponential/SwiGLU implementation out
   of diagnostic-only ownership into one shared STRAT-01 primitive;
2. make `strat01_r2b_run` call that shared primitive for block-0
   `ffn_swiglu-0` and refuse a non-multiple-of-four feature count;
3. make the closed SSE2 diagnostic call the same shared primitive, with no
   copied polynomial implementation;
4. declare the installed semantics explicitly in the Rung-2B and Rung-2C
   production `CONFIG` strings.

The default reference-generic Q4_K×Q8_K primitive, pinned-generic-F64 F16
reductions, double RMSNorm, Q5_0×Q8_0 K-B, Q6_K down projection, residual
order, layer-1 attention, routing, experts, and all tensor layouts remain
unchanged. Layer-1 SwiGLU sites are not changed in this cell: the prior
downstream experiment already qualified them under the frozen gates, and
changing them would add an unmeasured coordinate.

## Apparatus gate

Before accepted-artifact access, an apparatus-only run must establish:

- one shared implementation and no diagnostic duplicate;
- explicit `_mm_add_ps` / `_mm_mul_ps` ordering and no FMA intrinsic;
- four-lane execution with scalar-tail refusal;
- the frozen synthetic scalar-versus-SSE2 distinguisher and mutation control;
- production Rung-2B delegation to the shared primitive;
- diagnostic delegation to the same primitive;
- unchanged default Q4 and F16 helper selection;
- exact source, protocol, predecessor-adjudication, and reference-manifest
  bindings;
- successful full STRAT-01 Python regression and every registered C
  self-test;
- zero GGUF access and zero accepted-artifact/reference graph executions.

Only a qualified, committed apparatus authorizes the production run.

## Scientific execution and gates

Run exactly one standard C producer invocation:

```text
engine --strat01-gguf-rung2c <accepted.gguf> --out-dir <candidate>
```

That invocation must complete exactly two graph schedules, `prefill8` and
`cached7p1`. No reference producer may run. Validate the complete existing
Rung-2C schema: all 32 checkpoints per schedule, both layer caches, manifest
shape/type/order, finite payloads, fixed tokens/positions, artifact identity,
source identity, helper counts (`4608` QK and `524288` value), and exact
schedule accounting.

Use the existing Rung-2C gates unchanged:

- every F32 checkpoint except the two named terminals: NRMSE `<= 0.002` and
  normalized maximum `<= 0.01`;
- `ffn_inp-1` and `l_out-1`: NRMSE `<= 0.001` and normalized maximum
  `<= 0.005`;
- `ffn_moe_topk-1`: exact integer equality;
- prefill/cached token-7 continuity for every checkpoint: NRMSE `<= 2e-6`
  and normalized maximum `<= 1e-5`, or exact equality for integer routing;
- all six cache comparisons: NRMSE `<= 0.002` and normalized maximum
  `<= 0.01`.

The candidate must no longer reproduce the scalar predecessor `l_out-0`
hash. The ordinary production `CONFIG` and source controls must prove that
the shared SSE2 primitive, not a diagnostic-only arm, produced the candidate.
All pre-existing Rung-2C causal controls must continue to reject.

## Decision rule

- **`PASS_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION`** if the apparatus
  is qualified, the one production invocation is valid, all checkpoint,
  continuity, cache, route, helper-count, identity, and causal gates pass,
  and the scalar predecessor hash is displaced.
- **`FAIL_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION`** if the run is
  otherwise valid but any scientific gate fails. Report the first failed
  boundary without opening a new coordinate.
- **`VOID_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION`** for any source,
  binding, build, schema, count, identity, mutation, finiteness, apparatus,
  or execution-accounting failure.

Exactly one non-VOID production invocation is allowed. A VOID may be repaired
only model-free unless immutable outputs prove that no additional producer
execution is required.

## Non-claims and stop rule

This cell can close production fidelity only through layer 1 on the frozen
eight-token trace. It does not establish later layers, tokenizer/logits,
generation, task quality, RAM, or accepted-token throughput. Timing from this
diagnostic is inadmissible and must not enter `SPEED_LEDGER.md`.

Document and index the result before selecting the next depth boundary.
