# STRAT-01 Q6_K×Q8_K reference-generic production-integration protocol

**FROZEN BEFORE IMPLEMENTATION OR EXECUTION:** 2026-09-24

**Cell:** `STRAT-01-ENGINE-Q6K-Q8K-REFERENCE-GENERIC-PRODUCTION-INTEGRATION`

## Question and non-duplication

The closed
[compile-parity result](STRAT_01_GIGACHAT31_ENGINE_BLOCK0_Q6K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_RESULT_20260924.md)
proves that the isolated pinned `GGML_CPU_GENERIC` operation is byte-exact to
the independent oracle, immutable block-0 `ffn_out-0`, `l_out-0`, and the
complete downstream amplifier. It does not prove that the ordinary
`--strat01-gguf-rung2c` path delegates every Q6_K×Q8_K product to that helper.

Question: does installing only that exact helper in the normal shared Q6
dispatch make one standard accepted-artifact Rung-2C invocation reproduce the
immutable two-schedule reference through block 0 and complete layer 1?

Do not rerun compile parity, either historical arm, the independent full-
matrix oracle, any cross-input diagnostic, Q4/F16/SwiGLU work, or a reference
producer.

## Immutable bindings

- accepted GGUF: 6,474,702,976 bytes, SHA-256
  `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- reference-generic scientific adjudication SHA-256
  `cb811b19c58d3a31688967c18dc9119451ed01faee0e56098166efd4f46ff066`;
- post-SwiGLU production-integration adjudication SHA-256
  `d543c56d7e463cfec23e4fa3a7ec7fd5bfecbdccfdf84351ddf12085fcc3cc94`;
- pinned llama.cpp revision
  `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`;
- reference root/prefill/cached manifests remain the immutable Rung-2C repair-1
  tree under
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_repair1_20260923/pinned_reference/`;
- tokens `[1,72,14,14129,14,2135,1512,2015]`, positions
  `[0,1,2,3,4,5,6,7]`, schedules `prefill8` and `cached7p1`.

## Changed coordinate

Only shared production Q6 dispatch may change:

1. retain `strat01_q6k_q8k_dot_reference_generic` as the single exact
   implementation, non-inlined and targeted `no-avx,no-avx2,no-fma`;
2. replace the independent scalar body of `strat01_q6k_q8k_dot` with direct
   delegation to that helper;
3. thereby change both dense block-0 and routed/shared layer-1 Q6 calls made
   through the existing shared dispatch, without changing quantization,
   layouts, matrices, routing, residual order, or any other operator;
4. declare `q6kq8k=reference-generic-noavx-noavx2-nofma-noinline` in normal
   Rung-2B/Rung-2C `CONFIG` records.

Q4_K×Q8_K, F16 reductions, RMSNorm, Q5_0×Q8_0, SwiGLU, attention, routing,
weighting/sum, caches, tensor layouts, and numerical gates remain unchanged.

## Model-free apparatus gate

Before opening the artifact, the apparatus must prove:

- production and diagnostic ownership converge on the same exact helper and
  no second production arithmetic body remains;
- candidate/oracle bit parity at lengths 256, 512, **1280**, 1536, and 8960;
- the existing pairwise-distinct fixture, target attributes, noinline/source,
  short-read, mutation, and invalid-mode controls still fire;
- Rung-2B and Rung-2C source delegation plus both explicit `CONFIG` markers;
- unchanged Q4/F16/SwiGLU selections;
- complete registered Python regression and all C self-tests pass;
- all model, payload, matrix, diagnostic, donor-graph, and reference-graph
  counters are zero.

Only a qualified committed apparatus authorizes the scientific invocation.

## Scientific execution and gates

### Pre-implementation observability addendum

The standard Rung-2C schema emits `l_out-0`, not the internal
`ffn_out-0` buffer. Expanding that schema merely to restate the already closed
direct `ffn_out-0` result would add an unnecessary coordinate. Therefore the
production gate below observes byte-exact `l_out-0`, binds the unchanged
`ffn_inp-0` source path and residual order, and requires source-level direct
delegation to the exact helper. The direct `ffn_out-0` hash remains a binding
from compile parity, not a second production payload requirement. This
clarification was frozen before implementation.

Run exactly one normal producer command:

```text
engine --strat01-gguf-rung2c <accepted.gguf> --out-dir <candidate>
```

It must complete exactly the `prefill8` and `cached7p1` donor schedules and no
reference graph. Validate the existing Rung-2C schema, source identities,
tokens, positions, tensor order/types/shapes, helper counts, route equality,
cache accounting, finiteness, causal controls, and schedule continuity.

The decisive repair gates are byte-exact equality to immutable reference for:

- block-0 `l_out-0` in both schedules, with unchanged `ffn_inp-0` production
  and residual-add source controls;
- every one of the 32 complete layer-1 checkpoints in both schedules;
- all six cache comparisons and all integer routing payloads;
- every prefill/cached token-7 continuity comparison.

The old production block-0 terminal hash
`a71c814dc43360f73b49cb165e7cdf7128f2501015685ff16e7e524dd1f05f11`
must be displaced, while new `l_out-0` must equal
`385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa`.
The production report and source controls must prove the shared reference-
generic helper, not a diagnostic arm, produced the candidate.

## Decision and stop rule

- **`PASS_ENGINE_Q6_REFERENCE_GENERIC_PRODUCTION_INTEGRATION`** if apparatus,
  accounting, identity, source, causal, exact checkpoint, continuity, route,
  and cache gates all pass;
- **`FAIL_ENGINE_Q6_REFERENCE_GENERIC_PRODUCTION_INTEGRATION`** if the run is
  valid but any scientific exactness gate fails; report the first boundary and
  open no further coordinate;
- **`VOID_ENGINE_Q6_REFERENCE_GENERIC_PRODUCTION_INTEGRATION`** for any build,
  apparatus, identity, schema, count, source, mutation, finiteness, or
  execution-accounting failure.

Exactly one non-VOID producer invocation is allowed. A VOID may be repaired
model-free or adjudicated offline only when immutable outputs suffice; it may
not silently authorize a second producer run.

This cell establishes fidelity only on the frozen two-schedule trace. It does
not establish tokenizer/logits, generation, HumanEval, later layers, RAM, or
accepted-token throughput. Timing is inadmissible and must not enter
`SPEED_LEDGER.md`. Document the result before choosing the next boundary.
