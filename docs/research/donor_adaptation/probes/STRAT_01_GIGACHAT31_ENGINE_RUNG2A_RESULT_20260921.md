# STRAT-01 GigaChat 3.1 engine rung 2A — numerical FAIL from preserved captured payloads

**Date:** 2026-09-21

**Outcome:** `FAIL_ENGINE_RUNG2A`

**Cell:** `STRAT-01-ENGINE-RUNG2A`

**Apparatus revision:** `dbcbb01e10e2df883fe799589bce7e23e5fdeeee`

**Hash-buffer repair revision:** `5cdc826`

**Resolved-context repair revision:** `31bc68a`

**Offline adjudicator revision:** `89b15d5`

**Raw evidence:** [first void](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_20260921/), [second void](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_repair1_20260921/), [resolved-context apparatus validation](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_apparatus_repair2_20260921/), [third captured run](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_repair2_20260921/), [offline adjudication](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_offline_adjudication_20260921/)

**Claim labels.** Process exits, C/reference producer states, test outcomes, runtime dimensions, payload hashes, and numerical metrics are **MEASURED**. The stack-buffer diagnosis, llama.cpp context-padding rule, and lowercase `ggml_type_name()` diagnosis are **SOURCE-DERIVED** from exact revisions, raw manifests, and pinned source. The localization to full-matrix projection semantics rather than cache sequencing is an **INFERENCE** from the measured first-failure boundary and exact within-implementation continuity.

## Question and frozen scope

The frozen [Rung 2A protocol](STRAT_01_GIGACHAT31_ENGINE_RUNG2A_PROTOCOL_20260921.md) changes the fidelity coordinate from the scalar stored-row parity measured in [Rung 1](STRAT_01_GIGACHAT31_ENGINE_RUNG1_RESULT_20260921.md) to composed block-0 MLA attention semantics through `benchmarks/phase60/engine.c`. Its paired arms are `prefill8` and `cached7p1`, with intermediate tensors, compact F16 K-cache, continuity, and numerical gates fixed by the protocol. This record preserves the complete attempt chain and the eventual adjudication.

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

## Third captured run and offline adjudication

The third run is preserved unchanged at [`strat01_gigachat_engine_rung2a_repair2_20260921`](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_repair2_20260921/). Both producer processes returned zero: the C arm completed in **25.50 s**, and the pinned reference completed in **43.39 s** and emitted both complete trace trees. The external runner nevertheless marked the raw attempt `VOID_ENGINE_RUNG2A` because `ggml_type_name()` serialized callback types as lowercase `f32`, while the validator admitted only uppercase `F32`/`F16`. All 36 selected callback sources had the frozen name, operation, ordinal, shape, and 8 versus 7+1 decomposition; the type spelling was uniformly the sole identity mismatch.

Commit `89b15d5` makes type-name validation case-insensitive while still admitting only F32/F16, adds a negative control that refuses `q4_K`, and adds a fail-closed offline mode. The offline record hashes the original run manifest (`0f91db875cada58b054dfcfd57161668a92d17bfc2d27c43af206c0652ef372f`) and original void adjudication (`d7c530db7d6fca08e3755bdbdbc34f0762cdec35f71a1603bd31106c1dcf16d6`), revalidates every existing payload, and executes no donor process (`donor_executions=0`). Both C and reference outputs validate with empty errors; the frozen numerical gates then produce `FAIL_ENGINE_RUNG2A`.

## Numerical result

Of 24 arm/tensor comparisons, 6 pass and 18 fail. In both `prefill8` and `cached7p1`, `attn_norm-0` passes at NRMSE `7.965057458369789e-07`, `q_nope_absorbed_perm-0` passes at `9.347833248820601e-04`, and `Qcur-0` passes at `1.3294316405033932e-03`. The first failures occur immediately after the passing normalized input: `q-0` NRMSE is `0.0029148289138709186` and `kv_cmpr_pe-0` is `0.004722034291199872`, both above the general `0.002` gate.

Downstream, `kv_cmpr-0`/`Vcur-0` reach NRMSE `0.012175256746571883`; `Kcur-0` is `0.003814485105464076`; `kqv_out-0` is `0.01734043888612236` with normalized maximum `0.010648486532351408`, failing both general limits; and terminal `ffn_inp-0` is `0.007459962254437984`, above its `0.001` NRMSE limit. All three compact-cache checkpoints fail the `0.002` NRMSE gate: prefill/final and cached/final are `0.003816122325463294`, while cached/prefix7 is `0.00392033117785943`.

The prefill-versus-7+1 continuity checks pass exactly inside both implementations: NRMSE and normalized maximum are both zero for the C engine and for the pinned reference. The two arms also produce identical per-tensor parity metrics. This rules out the tested cache sequencing split as the source of the measured disagreement. Given that `attn_norm-0` passes and the first failures are projection outputs, the evidence points to full-matrix quantized projection/dequantization/accumulation semantics as the next diagnostic boundary; this is localization evidence, not yet a proved root cause.

## Adjudication and limits

The first two attempts and the raw third runner record remain preserved apparatus `VOID`s and must not be rewritten. The third run's producer payloads are nevertheless complete and immutable, and their separate offline validation/adjudication is a valid numerical `FAIL_ENGINE_RUNG2A` with zero additional donor executions. Under the frozen stop rule, Rung 2A is closed as FAIL and must not be rerun or have its thresholds tuned.

This record makes no claim about numeric agreement or disagreement, full-model execution, dense FFN or MoE semantics, tokenizer parity, logits/generation, C-path quality, RAM fit, or accepted-token speed. It does not modify any frozen threshold.

## Next gate

Do not proceed directly to dense SwiGLU Rung 2B: the attention block is not yet numerically faithful. Freeze a separate diagnostic that starts at the measured boundary after passing `attn_norm-0` and distinguishes weight decode, per-row scale handling, dot-product accumulation order, and full-matrix layout for `attn_q` and `attn_kv_a_mqa`. It should consume the already captured inputs/outputs where possible and must not reuse the Rung-2A acceptance cell. No speed-ledger update is due.
