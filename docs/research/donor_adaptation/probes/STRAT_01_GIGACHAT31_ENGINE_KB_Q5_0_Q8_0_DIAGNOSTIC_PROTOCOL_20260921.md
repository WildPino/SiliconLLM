# STRAT-01 GigaChat 3.1 K-B Q5_0×Q8_0 diagnostic protocol

**Frozen:** 2026-09-21, before implementation or adjudication

**Cell:** `STRAT-01-ENGINE-KB-Q5_0-Q8_0-DIAGNOSTIC`

**Purpose:** determine whether pinned Q5_0×Q8_0 runtime semantics close the first material residual at `q_nope_absorbed_perm-0`, without executing a donor graph.

## Nearest evidence and non-duplication

The [upstream RMSNorm propagation result](STRAT_01_GIGACHAT31_ENGINE_UPSTREAM_RMSNORM_PROPAGATION_DIAGNOSTIC_RESULT_20260921.md) makes `attn_norm-0` exact and leaves `q-0` at NRMSE `6.44789694114499e-8`, but `q_nope_absorbed_perm-0` remains at `0.00031511401613621`. This is the first material residual after the changed coordinate.

Rung 1 already proved canonical Q5_0 stored-row decoding and a scalar matvec over dequantized F32 weights. It did not test the runtime activation format used by pinned llama.cpp. At commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`, the CPU type traits bind `GGML_TYPE_Q5_0` matrix multiplication to `GGML_TYPE_Q8_0` activation rows and `ggml_vec_dot_q5_0_q8_0`. The current `strat01_r2a_kb_batch` instead dequantizes Q5_0 blocks and multiplies F32 activations directly. The changed estimand is therefore full-matrix Q5_0×Q8_0 execution, not codec decoding, tensor identity, or another Rung-1 row test.

## Immutable objects

| object | bytes | SHA-256 |
|---|---:|---|
| accepted GGUF | 6,474,702,976 | `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| reference `q-0`, `[192,32,8]` | 196,608 | `4dc1424d3f93651acaa152bd57222754c132ab3e43330d286c61e9739caff64b` |
| upstream-double `q-0`, `[192,32,8]` | 196,608 | `6255f40d5a717af2a742c484b0ab75ff549b37acea70f73556698c794c8b3366` |
| accepted-float `q-0`, `[192,32,8]` | 196,608 | `cbc263ed903c9a7d992ebd2f14795ea60f60b999efc3e5365be6b7d84fe1252b` |
| reference `q_nope_absorbed_perm-0`, `[512,32,8]` | 524,288 | `94ddcff3f90faca65dc7237222939f407751f5c29e1c3f145e817de1e9c45ee9` |
| upstream-double current-path replay target | 524,288 | `11adb3548a0b69320cc4a38c45e1350763a35ebd298a8f1bd0e6efe7c5ee1dee` |
| accepted-float current-path replay target | 524,288 | `715e12661c6934f82ae9eec437da5713a83405edac64cb4bb04f86482d3f3a99` |

The only admitted matrix is `blk.0.attn_k_b.weight`: Q5_0, rank 3, logical dimensions `[128,512,32]`, tensor offset `272421888`, byte span `1441792`, GGUF file offset `278524800`. Each token/head consumes the first 128 of its 192 direct-Q components and produces 512 absorbed components; output order remains token, head, feature.

## Diagnostic arms

Implement a diagnostic-only engine command and an independent pinned-llama helper. Neither may execute embeddings, RMSNorm, Q/KV projections, attention, caches, FFN, or any other donor graph component.

For exact-reference and upstream-double `q-0` inputs, execute:

1. the current dequantize-Q5_0-to-F32 path;
2. pinned Q8_0 activation quantization plus Q5_0×Q8_0 dot semantics.

Also replay the accepted-float input through the current path as an apparatus control. The pinned helper must be built from the clean frozen llama.cpp checkout, expose Q8_0 activation bytes and output payloads, and never self-certify PASS. The C command must independently serialize the same Q8_0 rows and outputs. Hash and validate all sources, inputs, matrix descriptors, output sizes, and artifact identity before arithmetic.

## Gates and controls

The diagnostic is valid only if:

- current-path upstream-double and accepted-float outputs replay their frozen targets byte-for-byte;
- C and pinned-helper Q8_0 activation bytes agree exactly for both primary inputs;
- C and pinned-helper Q5_0×Q8_0 outputs agree at NRMSE `<= 2e-6` and normalized maximum `<= 1e-5`;
- a one-byte input mutation, wrong head stride, omitted Q8_0 quantization, and high-bit-mask corruption all reject or miss the tight oracle gate;
- the pinned checkout is clean at the frozen revision;
- `donor_graph_executions=0` and no timing/rate claim is emitted.

Primary comparisons use the C pinned-semantic outputs against immutable reference `q_nope_absorbed_perm-0`:

| input | NRMSE | normalized maximum | role |
|---|---:|---:|---|
| exact reference `q-0` | `<= 2e-6` | `<= 1e-5` | operator/layout proof |
| upstream-double `q-0` | `<= 0.002` | `<= 0.01` | changed-coordinate sufficiency |

Report current-path, pinned-path, cross-input, C-versus-helper, and reference comparisons separately, plus Q8_0 changed-block/byte censuses among accepted-float, upstream-double, and exact-reference inputs. Do not change gates after observing values.

## Decision rule

- **`Q5_0_Q8_0_SUFFICIENT_FOR_KB_BOUNDARY`** if both primary comparisons and every apparatus/oracle control pass. Freeze one combined upstream-double-RMSNorm plus Q5_0×Q8_0 propagation confirmation; do not edit production yet.
- **`Q5_0_Q8_0_REFERENCE_ONLY`** if exact-reference input passes the tight operator gate but upstream-double input fails the general gate. Preserve the operator finding but do not integrate; localize the remaining input residual.
- **`Q5_0_Q8_0_FAILS_REFERENCE_INPUT`** if the run is valid but exact-reference input misses the tight gate. Do not patch production; use C/helper agreement and controls to distinguish runtime semantics from layout/orientation.
- **`VOID_KB_Q5_0_Q8_0_DIAGNOSTIC`** for any identity, replay, build, helper, completeness, or planted-control failure.

Exactly one non-VOID diagnostic execution is allowed. Because all arms are frozen-payload operator calls, `donor_graph_executions` must remain zero. Apparatus-only runs do not consume the cell.

## Non-claims and stop rule

This cell does not repair production or establish complete Rung 2A, Rung 2B, Rung 2C, later-layer parity, tokenizer, logits, generation, quality, RAM, or rate. Document and index the result before any combined propagation run. No `SPEED_LEDGER.md` entry is permitted.
