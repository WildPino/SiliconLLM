# STRAT-01 Rung-2C layer-1-start cross-input apparatus

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`.

The frozen layer-1-start diagnostic is qualified at commit
`b0d89f9d247c02366097ecbe02160b9ec3e49ba5`. No GGUF or scientific payload
was opened and no donor graph was executed.

| check | result |
|---|---:|
| Clang production build | PASS |
| C diagnostic self-test | `8/8` PASS |
| Python contract tests | `4/4` PASS |
| legacy kernel checks | `73,024/73,024` PASS |
| donor graph executions | `0` |

Canonical apparatus directory:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_layer1_start_cross_input_apparatus_20260923/`

| record | SHA-256 |
|---|---|
| adjudication | `dd4d2a9ba1703e5e998fd28603d2f7f27ebe601fb61508155f7231144ea7163d` |
| run manifest | `e7febc91f8385e239e61cdd807fd8ac4892ac03ac2b7bf8fb86b6fd4fb8fac36` |

The first direct script launch failed before argument parsing because the
repository root was absent from `sys.path`; it created no output directory and
read no model or payload. Commit `b0d89f9` repairs only import bootstrapping.
This is apparatus engineering, not a scientific invocation or a VOID result.

Exactly one non-VOID scientific invocation is authorized by the
[frozen protocol](STRAT_01_GIGACHAT31_ENGINE_RUNG2C_LAYER1_START_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md).
This qualification makes no numerical, repair, quality, or rate claim.
