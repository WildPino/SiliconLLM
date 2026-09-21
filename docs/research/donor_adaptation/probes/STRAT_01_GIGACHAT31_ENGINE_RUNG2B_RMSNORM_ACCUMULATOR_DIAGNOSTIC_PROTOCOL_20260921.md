# STRAT-01 GigaChat 3.1 Rung-2B RMSNorm accumulator diagnostic protocol

**Frozen:** 2026-09-21, before implementation or adjudication

**Cell:** `STRAT-01-ENGINE-RUNG2B-RMSNORM-ACCUMULATOR-DIAGNOSTIC`

**Purpose:** determine whether matching pinned llama.cpp's double RMSNorm sum is sufficient to restore the block-0 up/gate projection gates, without executing the donor graph.

## Source-backed hypothesis

The [cross-input result](STRAT_01_GIGACHAT31_ENGINE_RUNG2B_CROSS_INPUT_DIAGNOSTIC_RESULT_20260921.md) proves that the current Q4_K×Q8_K up/gate path passes near `1.5e-7` NRMSE on exact reference normalized input, while the accepted C normalized input reproduces the Rung-2B FAIL.

At pinned llama.cpp commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`, `ggml_compute_forward_rms_norm_f32` uses `ggml_float sum = 0.0`; `ggml_float` is double. Each term is formed as F32 `x[i]*x[i]`, cast to double for accumulation; `sum/ne00` is converted to F32 `mean`; `scale` and `x*scale*weight` remain F32. The current `strat01_r2a_rmsnorm` instead accumulates `ss` in float. The changed estimand is exactly this accumulator type.

## Immutable objects

| object | bytes | SHA-256 |
|---|---:|---|
| accepted GGUF | 6,474,702,976 | `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| reference `ffn_inp-0`, `[1536,8]` | 49,152 | `baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1` |
| accepted C `ffn_inp-0`, `[1536,8]` | 49,152 | `bd00c9c5b01726980ab0865e27860f73b7c4186940362919d16dee7e75ee8d82` |
| reference `ffn_norm-0` target | 49,152 | `7bbfd9c2f0ed17c12a429b5dff83f75d47a45dd97008ad88251a7ec8d64dc588` |
| accepted C float-sum `ffn_norm-0` replay target | 49,152 | `ea5d60791c4404fbb98e7839fa73ff8112c54abba6060a7fbc3d5e83b951e3e0` |
| reference `ffn_up-0` target | 286,720 | `2b608af95db90fcde83c29946ba2ac680fb6465cbaaf518fc12874ab510b29e4` |
| reference `ffn_gate-0` target | 286,720 | `5c30c0ada2e96ce43a92b22057b4d593b0b9ca342feade7092bfec084e8c6c2a` |
| accepted C up replay target | 286,720 | `7ebb2858834cd390e6563333b4e0da3f4f95098180ea64e1c0be9be79c4e74b3` |
| accepted C gate replay target | 286,720 | `83d26be14a2349a74f25bd5c56fd3499fe62e20daf41ed7fdedb0ea6a70c354f` |

The only admitted weights are `blk.0.ffn_norm.weight` F32 `[1536]` at GGUF offset `305786880`, byte span `6144`, file offset `311889792`, plus the already frozen up/gate Q4_K descriptors. Prefill/cached copies are hash-identical at this boundary; prefill is reused to avoid duplicate computation.

## Diagnostic implementation

Add a diagnostic-only command to `benchmarks/phase60/engine.c`. It must hash and parse the accepted GGUF, read the two captured `ffn_inp-0` payloads, and execute both RMSNorm variants:

1. current float-sum semantics;
2. pinned semantics: F32 product, double sum, F32 mean/scale/output.

For each input, write both normalized outputs. Then feed the double-sum normalized C input through the unchanged production Q4_K×Q8_K up/gate path and write those outputs. The diagnostic may also replay float-sum C up/gate as an apparatus control. It must report `donor_graph_executions=0` and never self-certify PASS.

Build flags remain Clang C11, `-O3 -mavx2 -mfma`, no fast-math, with the existing FP-contraction prohibition. Fail closed on every identity, descriptor, compiler, count, finiteness, or source-hash mismatch.

## Gates and controls

Primary repair-sufficiency gates use the unchanged Rung-2B general limits, separately for double-sum C-input up and gate versus their reference targets:

- NRMSE `<= 0.002`;
- normalized maximum `<= 0.01`.

The diagnostic is valid only if:

- float-sum C `ffn_norm`, up, and gate replay their accepted payloads byte-for-byte;
- double-sum reference-input `ffn_norm` is compared to the reference target and reported;
- all frozen baseline metrics reproduce within `1e-12`;
- a one-byte input mutation is refused;
- swapping up/gate targets rejects;
- source and artifact identities validate.

Report, without additional gates, all four norm comparisons, double-C up/gate comparisons, Q8_K change census between float-C and double-C normalized inputs, and output deltas.

## Decision rule

- **`DOUBLE_RMSNORM_SUFFICIENT_FOR_PROJECTION_GATES`** if both double-C up/gate primary gates pass and all controls pass. Freeze a changed-coordinate production confirmation that changes only RMSNorm accumulation, reuses the immutable pinned reference traces, and executes the full C artifact once.
- **`DOUBLE_RMSNORM_INSUFFICIENT_FOR_PROJECTION_GATES`** if the diagnostic is valid but either primary gate fails. Do not patch production. Use the measured double-C normalized boundary to localize the remaining upstream attention-residual discrepancy.
- **`VOID_RMSNORM_ACCUMULATOR_DIAGNOSTIC`** for any identity, replay, or planted-control failure.

Exactly one non-VOID diagnostic execution is allowed. It reads three block-0 weight tensors but executes no donor graph. It does not reopen Rung 2A, Rung 2B, or the cross-input diagnostic.

## Non-claims and stop rule

This cell is not a production repair and makes no claim about complete Rung 2B, Rung 2C, quality, logits, generation, RAM, or rate. Document and index its result before any production edit. No `SPEED_LEDGER.md` entry is permitted.

