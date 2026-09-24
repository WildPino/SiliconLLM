# STRAT-01 GigaChat 3.1 layer-1 routed-Q6 residual propagation result

**Canonical verdict:** `LAYER1_ROUTED_SWIGLU_RESIDUAL_SUFFICIENT`

The sole scientific diagnostic authorized by the frozen
[protocol](STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_Q6_RESIDUAL_PROPAGATION_PROTOCOL_20260924.md)
completed from qualified producer commit
`1bdfda67d64f68dd5b9576862f535cdb7873a474`. It reports no error, one
diagnostic invocation, and zero Q6, donor-graph, or reference-graph executions.

Canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_layer1_routed_q6_residual_propagation_20260924/`

| record | SHA-256 |
|---|---|
| `adjudication.json` | `2eee409b8f007b9e53136e7a3d14d250af93a4d951b1ce859d4080fcfe6a9bc2` |
| C binary | `26efea9da3387baa6ca939ff3dbd3d8e05015375f4ecb9f10c3418dcb4c58cca` |

The orchestration took `38.975` seconds. This is diagnostic cost, not
emitted-token throughput.

## Scientific outcome

The immutable old Q6 output produced from exact reference routed SwiGLU passes
the newly established downstream gate exactly. The captured production C down
payload still reproduces the predecessor failure.

| arm | routed-output NRMSE | downstream NRMSE | normalized maximum | result |
|---|---:|---:|---:|---|
| captured reference down | `0` | `0` | `0` | PASS, byte-exact anchor |
| old Q6(reference-SwiGLU) down | `6.97121e-8` | `0` | `0` | PASS |
| captured production C down | `1.35468e-5` | `0.002642086799405445` | `0.004687597394884091` | **FAIL** |
| token-7-negated Q6 control | — | `0.20962964201860176` | `0.3668577386863162` | expected rejection |

At the direct selected-expert output boundary, old Q6(reference-SwiGLU) versus
reference is `7.1095157598044e-8` NRMSE, while captured C down versus reference
is `7.95119111602408e-5`. The former therefore remains far below the local
gate and, crucially, remains below the complete downstream gate as well.

## Replay and controls

- exact-reference and production-C anchors replay their predecessor outputs;
- all immutable input, output, schedule-twin, source, and predecessor hashes
  match;
- every admitted payload refuses a one-byte mutation and arm-label swapping is
  rejected;
- the planted token-7 sign control fails decisively;
- all new, inherited, legacy, and Python tests pass;
- counters certify zero Q6 and zero graph executions.

## Interpretation and stop rule

The selected Q6 down projection is not sufficient to explain the layer-1
failure, even under the sensitive downstream gate. Close Q6, expert down,
router selection and normalization, routed reduction, shared expert, terminal
addition, and layer 2. Do not repeat this propagation.

The residual is already present in the routed SwiGLU input to Q6. The sole
non-duplicate successor is a frozen cross-input split on the same captured
layer-1 tensors: reference versus production gate input, up input, and
production expression semantics. Each arm must traverse the already validated
current Q6 plus the same closed downstream amplifier. It may execute Q6 but
must execute neither donor nor reference graph.

No repair, later-layer, tokenizer, logits, generation, task quality, RAM, or
rate claim follows. No `SPEED_LEDGER.md` update is due.
