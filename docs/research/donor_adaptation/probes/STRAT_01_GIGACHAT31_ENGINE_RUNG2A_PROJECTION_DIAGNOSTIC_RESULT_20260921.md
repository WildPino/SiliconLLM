# STRAT-01 GigaChat 3.1 engine Rung-2A projection diagnostic result

**Final label:** `ATTRIBUTED_Q8K_ACTIVATION_QUANTIZATION`
**Date:** 21 September 2026
**Protocol:** [frozen projection diagnostic](STRAT_01_GIGACHAT31_ENGINE_RUNG2A_PROJECTION_DIAGNOSTIC_PROTOCOL_20260921.md)
**Implementation/run commit:** `a963538aa450ad98e361d708f45e9c8602a63302`
**Raw record:** [`strat01_gigachat_engine_rung2a_projection_diagnostic_20260921`](../../../../benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_projection_diagnostic_20260921/)
**Adjudication SHA-256:** `624116ba35e645f5ced3e9b02ed40450061103596b07cf07d4183fae3018c16f`

## Outcome

Pinned GGML Q4_K×Q8_K multiplication reproduces every captured reference Q and KV projection value **bit for bit**: NRMSE and normalized maximum are both exactly zero for both tensors.  The current engine's dequantized-Q4_K/F32 multiplication independently reproduces its captured outputs within the frozen tight gate, while the same computation on the reference input retains the original Rung-2A failures.  The first projection mismatch is therefore attributed to activation quantization and dot-product semantics, not weight decode, row layout, input drift, or ordinary accumulation precision.

This result does not rewrite the closed `FAIL_ENGINE_RUNG2A`; it identifies the engine defect that caused the first failing boundary.

## Identity and apparatus

The runner verified the 6,474,702,976-byte GGUF SHA-256 `68a8732f...0b8deb`, the immutable source-run manifest SHA-256 `0f91db87...f372f`, all six preregistered payload hashes and sizes, exact Q4_K descriptors/weight spans, and clean pinned llama.cpp commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.  It executed no donor process (`donor_executions=0`).

Six model-free tests passed, including payload-byte mutation, shape/order substitution, token swap, wrong/dirty pinned revision, non-finite metric input, and frozen-classifier controls.  The linked GGML helper self-test passed after explicit CPU-table initialization.  One deterministic Q8_K activation scale-byte mutation fired independently for Q and KV and missed the tight gate by orders of magnitude.  All control flags are true and the error list is empty.

The diagnostic binary SHA-256 is `c603314fe3bb83445e45c7c2101e2a10b12050edd0eee3d5ea232f42e68561c5`.

## Frozen numerical adjudication

| Tensor / path | NRMSE | Normalized max | Frozen interpretation |
|---|---:|---:|---|
| Q D32 on C input vs captured C | `7.817554510208904e-08` | `1.401131927780056e-07` | tight control PASS |
| KV D32 on C input vs captured C | `9.107736850241328e-08` | `9.721429520656726e-08` | tight control PASS |
| Q D32 on reference input vs captured reference | `0.002914761031330664` | `0.0027936110089160737` | old Rung-2A NRMSE gate FAIL |
| KV D32 on reference input vs captured reference | `0.004721964177339217` | `0.002155137458634987` | old Rung-2A NRMSE gate FAIL |
| Q pinned Q4_K×Q8_K vs captured reference | `0.0` | `0.0` | exact/tight PASS |
| KV pinned Q4_K×Q8_K vs captured reference | `0.0` | `0.0` | exact/tight PASS |
| Q F64 accumulation audit vs captured reference | `0.0029147593566815917` | `0.0027940776529225076` | old gate FAIL |
| KV F64 accumulation audit vs captured reference | `0.004721995166470306` | `0.0021551617388645768` | old gate FAIL |
| Q mutated-Q8_K control vs captured reference | `0.09223368240237316` | `0.20813699286762635` | control fires |
| KV mutated-Q8_K control vs captured reference | `0.11095347547502002` | `0.25895399022749893` | control fires |

The D32/reference values reproduce the measured Rung-2A discrepancies to the expected small difference from using the exact reference `attn_norm-0` rather than the near-identical C input.  Binary64 accumulation does not materially reduce either discrepancy.  Exact Q8_K reproduction, together with exact weight-span identity and the passing D32/C controls, rules out the registered alternatives at this boundary.

## Scientific interpretation

The project C path currently computes a mathematically reasonable dequantized-weight/F32 dot, but that is not the numerical operator used by pinned llama.cpp for Q4_K matrices.  GGML first quantizes each F32 activation row to Q8_K and then evaluates Q4_K×Q8_K.  That lossy activation conversion is part of the reference operator's semantics and accounts for both first failing tensors exactly.

The result is narrow:

- it establishes the cause of the first Q and KV projection mismatch for this artifact, layer, inputs, pinned CPU backend, and kernel path;
- it does not establish downstream attention parity after a repair;
- it does not test FFN/MoE, tokenizer, logits, generation, quality, RAM, or speed;
- it makes no claim that Q8_K activation conversion is universally preferable in accuracy terms—only that matching it is required for parity with this accepted reference path.

## Next controlled step

Implement exact Q8_K activation quantization plus Q4_K×Q8_K stored-row multiplication in the project engine for the affected full-matrix path.  First require model-free codec/dot tests and an offline confirmation against these immutable captured Q/KV targets.  Only after that repair is numerically closed should a separately frozen changed-coordinate attention confirmation decide whether any downstream Rung-2A mismatch remains.  Do not repeat the old acceptance cell, tune its thresholds, or proceed directly to 2B.

No `SPEED_LEDGER.md` entry is due.
