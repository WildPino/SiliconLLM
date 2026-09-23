# STRAT-01 GigaChat 3.1 engine Rung 2C result

**Verdict:** `FAIL_ENGINE_RUNG2C`.

The first complete block-1 routed/shared MoE cell is valid and fails its frozen
numerical gates. The earliest failing dependency boundary is the layer-1
attention output `kqv_out-1`, not the discrete router. Do not repeat Rung 2C or
tune its gates.

## Result

Both `prefill8` and `cached7p1` produce the same pass/fail surface:

| surface | result |
|---|---:|
| C/reference checkpoints | `34/64` PASS (`17/32` in each arm) |
| C/reference continuity checks | `64/64` PASS |
| cache comparisons | `6/6` PASS |
| causal negative controls | `10/10` reject as required |
| reference producer | `1` invocation, `2` completed schedules |
| C producer | `1` invocation, `2` completed schedules |

The exact accepted block-0 start hashes recur in both arms. Layer-1 attention
checkpoints through `Qcur-1`, `Kcur-1`, and `Vcur-1` pass. The first failure is:

| checkpoint | NRMSE | normalized maximum | frozen limits | result |
|---|---:|---:|---:|---|
| `Qcur-1` | `0.0010587355` | `0.0033672295` | `0.002 / 0.01` | PASS |
| `Kcur-1` | `0.0009183710` | `0.0013156948` | `0.002 / 0.01` | PASS |
| `Vcur-1` | `0.0012282778` | `0.0015361116` | `0.002 / 0.01` | PASS |
| `kqv_out-1` | `0.0061361676` | `0.0030491103` | `0.002 / 0.01` | **FAIL** |
| `ffn_inp-1` | `0.0026648083` | `0.0011663754` | `0.001 / 0.005` | **FAIL** |
| `l_out-1` | `0.0059407696` | `0.0029673050` | `0.001 / 0.005` | **FAIL** |

All router checkpoints pass in both arms: F32 logits, sigmoid probabilities,
selection-only bias, exact ordered top-4 IDs, selected unbiased weights, and
normalized weights. This does not certify the expert kernels independently:
their inputs already contain the failed attention residual. It does show that
the first material residual is upstream of the routed expert computation and
that a generic router rewrite would attack the wrong boundary.

## Canonical record and recovery

The producers ran from execution HEAD
`00d894c14c6f2bf75e2391283b6a37c09876f8ae`. Their immutable raw directory is:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_repair1_20260923/`.

Both producers completed and emitted valid manifests and unique completion
markers. The original runner then raised while JSON-encoding in-memory NumPy
cache copies after adjudication. It neither wrote a verdict nor reran a
producer. Commit `fd23b59a87b29687899363daf7295725fa3fd815` adds a tested
metadata-only serializer and a recovery mode that revalidates every manifest,
payload digest, source control, model identity, and producer marker without
compilation or producer execution.

Canonical recovered record:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_repair1_recovery1_20260923/`.

| item | SHA-256 |
|---|---|
| adjudication | `3742bc5982dd47be36a8e422f7b79c9e6716dc628153e44e6f9e90553c9ba62c` |
| run manifest | `898f614a8baaf40716a7aab66e4e31ba51c92da84e133ff24a55e7ccafa60dd4` |
| accepted GGUF | `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |

The recovery directory is the canonical adjudication; the source directory is
the immutable producer evidence. Neither should be overwritten.

## Consequence and next coordinate

Close Rung 2C as a measured FAIL. Do not rerun either producer, relax the
frozen limits, or interpret downstream expert failures as independent defects.
The next admissible coordinate is a zero-producer, offline cross-input
diagnostic at `kqv_out-1` using the already captured C/reference `Qcur`,
F16-rounded cache rows, and V-B weights. It must separate:

1. attention/V-B operator semantics on exact reference inputs;
2. amplification of the individually admissible Q/K/V input residuals;
3. query-side versus cache-side contribution.

Only a frozen result from that diagnostic may authorize a production repair.
This result establishes no later-layer parity, tokenizer/logit/generation
quality, HumanEval, RAM, or rate claim and does not update `SPEED_LEDGER.md`.
