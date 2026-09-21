# STRAT-01 GigaChat 3.1 Rung-2B cross-input Q4_K diagnostic protocol

**Frozen:** 2026-09-21, before implementation or adjudication

**Cell:** `STRAT-01-ENGINE-RUNG2B-CROSS-INPUT-DIAGNOSTIC`

**Purpose:** explain the first measured Rung-2B failure without rerunning the accepted donor graph.

## Prior result and changed estimand

[Rung 2B](STRAT_01_GIGACHAT31_ENGINE_RUNG2B_RESULT_20260921.md) is closed as `FAIL_ENGINE_RUNG2B`. Both schedules pass `ffn_norm-0` and first fail at the parallel Q4_K `ffn_up-0` and `ffn_gate-0` projections. Schedule continuity is exact and all semantic controls reject.

This diagnostic changes one coordinate: the C Q4_K×Q8_K projection path receives either the captured pinned-reference `ffn_norm-0` or the captured C-engine `ffn_norm-0`. It asks whether the projection path passes when both implementations start from the exact same normalized input. It does not execute embeddings, attention, cache, RMSNorm, SwiGLU, Q6_K down, residual, later layers, logits, or generation.

## Immutable inputs

All payloads come from the accepted numerical run `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2b_repair2_20260921/`. Prefill and cached payload hashes are identical at `ffn_norm-0`, so only the prefill copy is read; this is deduplication, not schedule selection.

| object | bytes | SHA-256 |
|---|---:|---|
| accepted GGUF | 6,474,702,976 | `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| reference `ffn_norm-0`, shape `[1536,8]` | 49,152 | `7bbfd9c2f0ed17c12a429b5dff83f75d47a45dd97008ad88251a7ec8d64dc588` |
| C `ffn_norm-0`, shape `[1536,8]` | 49,152 | `ea5d60791c4404fbb98e7839fa73ff8112c54abba6060a7fbc3d5e83b951e3e0` |
| reference `ffn_up-0`, shape `[8960,8]` | 286,720 | `2b608af95db90fcde83c29946ba2ac680fb6465cbaaf518fc12874ab510b29e4` |
| reference `ffn_gate-0`, shape `[8960,8]` | 286,720 | `5c30c0ada2e96ce43a92b22057b4d593b0b9ca342feade7092bfec084e8c6c2a` |
| accepted C `ffn_up-0` replay target | 286,720 | `7ebb2858834cd390e6563333b4e0da3f4f95098180ea64e1c0be9be79c4e74b3` |
| accepted C `ffn_gate-0` replay target | 286,720 | `83d26be14a2349a74f25bd5c56fd3499fe62e20daf41ed7fdedb0ea6a70c354f` |

The only admitted matrices are `blk.0.ffn_up.weight` and `blk.0.ffn_gate.weight`, both Q4_K with logical shape `[1536,8960]`, from that GGUF. The executable must fail closed on every byte count, hash, type, shape, offset, compiler, or artifact mismatch.

## Implementation contract

Extend `benchmarks/phase60/engine.c` with a diagnostic-only command that:

1. parses and hashes the accepted GGUF but executes no donor graph;
2. reads the two pinned F32 normalized-input payloads;
3. applies the existing production `strat01_r2a_matmul_batch` Q4_K×Q8_K path to up and gate for each input;
4. writes four F32LE outputs: `{reference_input,c_input} × {up,gate}`;
5. records complete Q8_K-buffer hashes for each input and a byte/block change census between them;
6. self-certifies only `OUTPUT_READY_PENDING_EXTERNAL_ADJUDICATION`, never PASS.

Compile with the accepted production flags: Clang C11, `-O3 -mavx2 -mfma`, no fast-math, and FP contraction disabled by the existing source pragma. One model-free self-test must exercise payload hashing, Q8 change census, descriptor refusal, and the four-output routing.

## External adjudication

The runner must validate source hashes, exact input identities, output sizes/hashes, compiler identity, and `donor_graph_executions=0`. It then computes NRMSE and normalized maximum with the frozen Rung-2B definitions.

Primary comparisons:

1. `C_Q4Q8(reference_norm, up)` versus captured reference `ffn_up-0`;
2. `C_Q4Q8(reference_norm, gate)` versus captured reference `ffn_gate-0`.

The unchanged general gate is NRMSE `<= 0.002` and normalized maximum `<= 0.01`, separately for both matrices. No aggregate average may rescue a failed matrix.

Required controls and descriptive measurements:

- `C_Q4Q8(c_norm, up/gate)` must be byte-identical to the two accepted C replay targets. Failure makes the diagnostic `VOID`.
- The accepted Rung-2B C-input-versus-reference comparisons must be recomputed and agree with the frozen metrics within `1e-12`; failure is `VOID`.
- A one-byte mutation of either input must be refused by identity validation.
- Gate/up descriptor swap must be refused or fail its corresponding target gate.
- Report input NRMSE/normalized maximum, Q8_K serialized hashes, changed-block count, changed-byte count, and the C-output delta between the two inputs. These are localization measurements, not acceptance gates.

## Decision rule

- **`REFERENCE_INPUT_PASSES_Q4K_PATH`** if both primary comparisons pass. Together with exact C-input replay and the already frozen Rung-2B failures, this establishes that the accepted upstream perturbation is sufficient to move the composed FFN projections outside their gate. Q8_K census may localize the discontinuity but must not be promoted to a proof of the exclusive mechanism.
- **`REFERENCE_INPUT_FAILS_Q4K_PATH`** if either primary comparison fails. Freeze a narrower matrix/kernel diagnostic for Q4_K decode, scales, layout, Q8_K quantization, and accumulation; do not repair production from intuition.
- **`VOID_CROSS_INPUT_DIAGNOSTIC`** for any identity, apparatus, replay, or planted-control failure.

Exactly one non-VOID execution of this diagnostic is allowed. It reads donor weights but has `donor_graph_executions=0`; it does not consume or reopen the Rung-2B accepted-artifact execution. Do not change gates after observing values.

## Non-claims and stop rule

This cell makes no claim about repaired Rung 2B, Rung 2C, MoE routing, quality, logits, generation, RAM, or rate. It cannot promote the engine path. After a non-VOID result, document and index it before implementing any repair. No `SPEED_LEDGER.md` entry is permitted.

