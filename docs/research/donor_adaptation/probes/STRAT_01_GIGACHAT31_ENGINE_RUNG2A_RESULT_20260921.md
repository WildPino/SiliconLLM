# STRAT-01 GigaChat 3.1 engine rung 2A — first attempt voided by reference apparatus

**Date:** 2026-09-21

**Outcome:** `VOID_ENGINE_RUNG2A`

**Cell:** `STRAT-01-ENGINE-RUNG2A`

**Apparatus revision:** `dbcbb01e10e2df883fe799589bce7e23e5fdeeee`

**Narrow repair revision:** `5cdc826`
**Raw evidence:** [`strat01_gigachat_engine_rung2a_20260921`](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_20260921/)

**Claim labels.** The process exit, C producer state, test outcomes, and missing reference trace are **MEASURED**. The stack-buffer diagnosis is **SOURCE-DERIVED** from the failed revision and Windows exit status. The repaired donor adjudication remains **PROPOSED**; only its model-free apparatus readiness has been measured.

## Question and frozen scope

This is the first execution of the frozen [Rung 2A protocol](STRAT_01_GIGACHAT31_ENGINE_RUNG2A_PROTOCOL_20260921.md), changing the fidelity coordinate from the scalar stored-row parity measured in [Rung 1](STRAT_01_GIGACHAT31_ENGINE_RUNG1_RESULT_20260921.md) to composed block-0 MLA attention semantics through `benchmarks/phase60/engine.c`. The planned paired arms are `prefill8` and `cached7p1`, with intermediate tensors, compact F16 K-cache, continuity, and numerical gates fixed by the protocol.

The target was the same accepted GigaChat 3.1 Q4 artifact: 6,474,702,976 bytes, SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`. The pinned reference is llama.cpp commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.

## Execution record

The apparatus and regressions completed: **21/21 model-free tests passed**, the C rung-2A self-test passed **22 checks**, and the legacy engine self-test passed **73,024 checks**. The raw manifest records the artifact's actual byte count and SHA-256 equal to the frozen identity.

The C arm completed in **26.6748359 s** and emitted `ENGINE_OUTPUT_READY_PENDING_REFERENCE`. This is an intermediate producer state, not a parity result. The pinned reference process exited after **1.9933109 s** with Windows status **3221225725 (`0xC00000FD`, stack overflow)**. Both captured reference stdout and stderr are empty, and the expected `pinned_reference` trace directory was not created. Consequently, the paired numerical adjudication did not run and no comparison value exists.

The authoritative raw `run_manifest.json` labels the attempt `VOID_ENGINE_RUNG2A`; the raw C output is retained. The elapsed times above are operational diagnostics only: they are not throughput measurements and do not belong in `SPEED_LEDGER.md`.

## Source-backed apparatus diagnosis

At the failed apparatus revision, the pinned-reference helper's `sha256_file` declared `std::array<uint8_t, 1 << 20> buf{}` as an automatic local object. The Windows default thread stack is also 1 MiB. The helper hashes and validates the accepted GGUF before model loading, so this oversized stack frame explains the immediate stack-overflow exit and the absence of reference traces. This diagnosis is supported by the exact source at commit `dbcbb01e10e2df883fe799589bce7e23e5fdeeee` and the process status/raw logs. The donor was **not evaluated by the reference**; no statement about model loading, callback extraction, MLA parity, or numerical failure follows.

The narrow repair at commit `5cdc826` changes the streaming hash buffer from the automatic 1 MiB array to a heap-backed `std::vector<uint8_t>` and adds a file-hash self-test. Its separate apparatus-only record is [`strat01_gigachat_engine_rung2a_apparatus_repair1_20260921`](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_apparatus_repair1_20260921/): status `APPARATUS_READY_NO_DONOR_EXECUTION`, empty errors, and 21/21 model-free tests passed. This establishes apparatus readiness only; it is not a donor execution or a repair of the voided attempt.

## Adjudication and limits

The frozen protocol classifies an incomplete reference dump/process failure as `VOID_ENGINE_RUNG2A`. Therefore this first attempt is **VOID, not FAIL**. Its state is preserved even if a later, narrowly repaired donor execution produces a valid result. Rung 2A is currently neither a numerical `PASS_ENGINE_RUNG2A` nor a numerical `FAIL_ENGINE_RUNG2A`.

This record makes no claim about numeric agreement or disagreement, full-model execution, dense FFN or MoE semantics, tokenizer parity, logits/generation, C-path quality, RAM fit, or accepted-token speed. It does not modify any frozen threshold.

## Next gate

The repaired apparatus has not yet executed the donor. The narrowly scoped reference hash-buffer/self-test fix is committed at `5cdc826`; the next step is the single permitted repaired execution in a **new empty raw directory**, retaining this first void and all frozen protocol gates. No speed-ledger update is due unless a separately valid rate experiment is later run.
