# STRAT-01 GigaChat 3.1 Rung-2C attention cross-input apparatus

**Status:** `APPARATUS_READY_NO_DONOR_EXECUTION`.

The implementation at commit
`2a9fbb9a0a1ec847178555bdb6f945299aba5cc5` passes its frozen apparatus
qualification. No GGUF was opened, no captured boundary payload was consumed,
and no donor graph was executed.

## Qualified surface

The C command implements the six frozen boundary-only arms:

- native C-query/C-KV replay;
- exact reference-query/reference-KV operator test;
- C-query/reference-KV and reference-query/C-KV cross-input arms;
- the preregistered token-7 query-negation and KV-row-swap controls.

Each arm resets its compact F16 cache and calls the unchanged production
`strat01_r2a_cache_write`, `strat01_r2a_attend_one`, and
`strat01_r2a_vb_batch` functions. The executable can emit only
`OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION`; the Python runner owns all
scientific gates and the frozen decision tree.

## Qualification evidence

| check | result |
|---|---:|
| Clang C11 `-O3 -mavx2 -mfma` build | PASS |
| diagnostic C self-test | `11/11` PASS |
| Python schema/gate/control tests | `5/5` PASS |
| legacy kernel checks | `73,024/73,024` PASS |
| donor graph executions | `0` |

Canonical apparatus directory:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_cross_input_apparatus_20260923/`

| record | SHA-256 |
|---|---|
| adjudication | `51817957cc510b3232e99aebe33a3e54033dacf0e62550aa86aa68a9ab05ab97` |
| run manifest | `b9e226f1ab635f64d4b38d60b63bf7c9d44ce5fa8194070a495e31e51296e275` |

## Authorization and non-claims

The apparatus authorizes exactly one non-VOID scientific invocation under the
[frozen protocol](STRAT_01_GIGACHAT31_ENGINE_RUNG2C_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md).
That invocation may read only the accepted GGUF V-B tensor and the immutable
Rung-2C prefill Q/K/V payloads. It must execute zero donor graphs.

This qualification is not a numerical result and makes no claim about repaired
Rung 2C, output projection, MoE, later layers, quality, generation, RAM, or
rate. Do not update `SPEED_LEDGER.md` and do not rerun either Rung-2C producer.
