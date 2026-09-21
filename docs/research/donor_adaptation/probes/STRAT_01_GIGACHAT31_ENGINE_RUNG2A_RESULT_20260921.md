# STRAT-01 GigaChat 3.1 engine rung 2A — two reference-apparatus voids, executable repair ready

**Date:** 2026-09-21

**Outcome:** `VOID_ENGINE_RUNG2A`

**Cell:** `STRAT-01-ENGINE-RUNG2A`

**Apparatus revision:** `dbcbb01e10e2df883fe799589bce7e23e5fdeeee`

**Hash-buffer repair revision:** `5cdc826`

**Resolved-context repair revision:** `31bc68a`

**Raw evidence:** [first void](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_20260921/), [second void](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_repair1_20260921/), [current apparatus-only validation](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_apparatus_repair2_20260921/)

**Claim labels.** Process exits, C producer states, test outcomes, runtime dimension logs, and missing reference traces are **MEASURED**. The stack-buffer diagnosis and llama.cpp context-padding rule are **SOURCE-DERIVED** from the exact failed revisions and pinned source. Donor numerical adjudication remains **UNMEASURED**; only apparatus readiness has been measured.

## Question and frozen scope

This is the first execution of the frozen [Rung 2A protocol](STRAT_01_GIGACHAT31_ENGINE_RUNG2A_PROTOCOL_20260921.md), changing the fidelity coordinate from the scalar stored-row parity measured in [Rung 1](STRAT_01_GIGACHAT31_ENGINE_RUNG1_RESULT_20260921.md) to composed block-0 MLA attention semantics through `benchmarks/phase60/engine.c`. The planned paired arms are `prefill8` and `cached7p1`, with intermediate tensors, compact F16 K-cache, continuity, and numerical gates fixed by the protocol.

The target was the same accepted GigaChat 3.1 Q4 artifact: 6,474,702,976 bytes, SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`. The pinned reference is llama.cpp commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.

## First execution record: stack overflow

The apparatus and regressions completed: **21/21 model-free tests passed**, the C rung-2A self-test passed **22 checks**, and the legacy engine self-test passed **73,024 checks**. The raw manifest records the artifact's actual byte count and SHA-256 equal to the frozen identity.

The C arm completed in **26.6748359 s** and emitted `ENGINE_OUTPUT_READY_PENDING_REFERENCE`. This is an intermediate producer state, not a parity result. The pinned reference process exited after **1.9933109 s** with Windows status **3221225725 (`0xC00000FD`, stack overflow)**. Both captured reference stdout and stderr are empty, and the expected `pinned_reference` trace directory was not created. Consequently, the paired numerical adjudication did not run and no comparison value exists.

The authoritative raw `run_manifest.json` labels the attempt `VOID_ENGINE_RUNG2A`; the raw C output is retained. The elapsed times above are operational diagnostics only: they are not throughput measurements and do not belong in `SPEED_LEDGER.md`.

## Source-backed apparatus diagnosis

At the failed apparatus revision, the pinned-reference helper's `sha256_file` declared `std::array<uint8_t, 1 << 20> buf{}` as an automatic local object. The Windows default thread stack is also 1 MiB. The helper hashes and validates the accepted GGUF before model loading, so this oversized stack frame explains the immediate stack-overflow exit and the absence of reference traces. This diagnosis is supported by the exact source at commit `dbcbb01e10e2df883fe799589bce7e23e5fdeeee` and the process status/raw logs. The donor was **not evaluated by the reference**; no statement about model loading, callback extraction, MLA parity, or numerical failure follows.

The narrow repair at commit `5cdc826` changes the streaming hash buffer from the automatic 1 MiB array to a heap-backed `std::vector<uint8_t>` and adds a file-hash self-test. Its separate apparatus-only record is [`strat01_gigachat_engine_rung2a_apparatus_repair1_20260921`](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_apparatus_repair1_20260921/): status `APPARATUS_READY_NO_DONOR_EXECUTION`, empty errors, and 21/21 model-free tests passed. This establishes apparatus readiness only; it is not a donor execution or a repair of the voided attempt.

## Second execution record: impossible resolved-context assertion

The one repaired donor execution is preserved at [`strat01_gigachat_engine_rung2a_repair1_20260921`](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_repair1_20260921/). The C producer again completed successfully, in **24.85 s**, and emitted `ENGINE_OUTPUT_READY_PENDING_REFERENCE`. The pinned reference now passed artifact hashing, loaded the exact 6,474,702,976-byte model, and constructed its CPU context before exiting with code 2 after **32.7977285 s**. Its runtime log records `n_ctx=256`, `n_batch=8`, and `n_ubatch=8`; no reference trace directory was emitted, so numerical adjudication again remained `NOT_RUN`.

The exact pinned source explains the divergence: `src/llama-context.cpp:288` applies `GGML_PAD(cparams.n_ctx, 256)`. Therefore the frozen request `n_ctx=8` deterministically resolves to a 256-slot allocation at the very revision selected as reference. The failed helper nevertheless demanded `llama_n_ctx(ctx)==8`. That assertion was internally impossible and did not test model semantics. The second result is consequently another apparatus `VOID_ENGINE_RUNG2A`, not a numerical failure.

Commit `31bc68a` repairs the contract without changing the estimand: it retains `requested_n_ctx=8`, requires and records `resolved_n_ctx=256`, and still requires `n_batch=n_ubatch=8`. The protocol now states this clarification explicitly. Token IDs, positions, logical eight-token tensor shapes, occupied cache rows, metrics, and thresholds are unchanged. The fresh apparatus-only record [`strat01_gigachat_engine_rung2a_apparatus_repair2_20260921`](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_apparatus_repair2_20260921/) is `APPARATUS_READY_NO_DONOR_EXECUTION`, has empty errors, passes **22/22** model-free tests, the C Rung-2A self-test's **22 checks**, and all **73,024** legacy checks.

## Adjudication and limits

The protocol classifies an incomplete reference dump/process failure as `VOID_ENGINE_RUNG2A`. Therefore both preserved donor attempts are **VOID, not FAIL**. Neither produced a reference payload or any paired comparison value. Rung 2A is currently neither a numerical `PASS_ENGINE_RUNG2A` nor a numerical `FAIL_ENGINE_RUNG2A`.

This record makes no claim about numeric agreement or disagreement, full-model execution, dense FFN or MoE semantics, tokenizer parity, logits/generation, C-path quality, RAM fit, or accepted-token speed. It does not modify any frozen threshold.

## Next gate

The resolved-context repair has not executed the donor. The next step is one execution in a **new empty raw directory** at commit `31bc68a`, retaining both voids and every numerical gate. If that execution is again apparatus-void, stop further donor retries and audit the complete reference path before another attempt. No speed-ledger update is due unless a separately valid rate experiment is later run.
