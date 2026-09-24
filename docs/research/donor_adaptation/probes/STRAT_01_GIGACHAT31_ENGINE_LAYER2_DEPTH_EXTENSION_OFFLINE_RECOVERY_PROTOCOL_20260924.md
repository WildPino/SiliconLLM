# STRAT-01 layer-2 depth-extension offline-recovery protocol

**State:** FROZEN BEFORE RECOVERY IMPLEMENTATION OR EXECUTION

## Trigger and immutable source run

The sole scientific run authorized by the
[layer-2 protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_PROTOCOL_20260924.md)
completed from commit `9360378657a74f764b0374424f3cb325c61fd764` with one
reference invocation and one C invocation, each completing `prefill8` and
`cached7p1`. Its raw adjudication is immutable at:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer2_depth_extension_20260924/adjudication.json`

SHA-256: `65cf9f30bdf4c2028f5c13f91cf4dfc44d28b3c04d07bc254c395e0c74e0693c`.

The runner emitted `FAIL_ENGINE_LAYER2_DEPTH_EXTENSION`, but that label is
not canonical. Twelve checkpoint comparisons failed, while the inherited
aggregate `omit_shared_expert` causal mutation remained inside the general
gate: NRMSE `0.0009646364838712824`, normalized maximum
`0.0004489167347438313`. The frozen decision rule classifies any
non-rejecting causal mutation as VOID. The raw run must therefore be treated
as `VOID_ENGINE_LAYER2_DEPTH_EXTENSION_CAUSAL_CONTROL` until this recovery is
adjudicated.

Neither producer may be rerun. Their payloads and logs are immutable recovery
inputs.

## Why the control is repaired

The aggregate mutation compares the routed-only output with the routed plus
shared sum. At layer 2 the shared branch is nonzero but small relative to that
sum, so omission is masked by the denominator. Token permutation, token
reversal, and sign inversion of the same branch also remain within the frozen
aggregate gate. Relaxing the requirement or tightening a threshold after
seeing the data is forbidden.

The repaired `omit_shared_expert` control is branch-local: compare an all-zero
array with the immutable reference checkpoint `ffn_shexp-2`. Apply the same
unchanged general limits, NRMSE `<= 0.002` and normalized maximum `<= 0.01`,
to both schedules. The control rejects only if the captured shared branch is
materially nonzero in its own coordinate. The other nine inherited controls
remain byte-for-byte unchanged. This repairs observability; it does not alter
a checkpoint threshold, producer output, or scientific hypothesis.

## Offline-only recovery contract

The recovery must:

1. bind the raw adjudication by exact path, byte count, and SHA-256;
2. require raw `errors=[]`, one invocation per producer, and exactly two
   completion markers per producer;
3. revalidate all reference and C manifests, payload hashes, path
   containment, shapes, types, operations, artifact identity, committed
   engine/header hashes, helper counts, predecessor-reference equality, and
   all checkpoint/cache/continuity comparisons from the existing payloads;
4. preserve the original aggregate omission metric as a disclosed diagnostic;
5. replace only its decision role with the branch-local omission control;
6. execute zero reference/C producers and zero donor/reference graphs;
7. record recovery-source identity, input key-file hashes, and both the raw
   and recovered classifications.

No accepted-model hash, build, or graph execution is required. Resolving the
already recorded model path for manifest identity is not producer access.

## Decision rule

- `PASS_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERED` only if every original
  numerical gate and every repaired integrity/control gate passes.
- `FAIL_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERED` if integrity and all ten
  causal controls pass but at least one frozen checkpoint, continuity, or
  cache gate fails.
- `VOID_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERY` for any identity, source,
  count, schema, marker, predecessor, control, or recovery-accounting failure.

If recovered FAIL, report the first failing checkpoint in frozen execution
order and freeze a separate diagnostic there. Do not edit an operator inside
this recovery and do not rerun either producer.

## Non-claims

This recovery can only canonicalize the existing layer-2 fidelity evidence.
It makes no claim about later layers, tokenizer, logits, generation, quality,
RAM, or throughput and cannot update `SPEED_LEDGER.md`.
