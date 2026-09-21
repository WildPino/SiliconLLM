# STRAT-01 GigaChat 3.1 upstream RMSNorm propagation diagnostic protocol

**Frozen:** 2026-09-21, before implementation or adjudication

**Cell:** `STRAT-01-ENGINE-UPSTREAM-RMSNORM-PROPAGATION-DIAGNOSTIC`

**Purpose:** determine whether pinned double accumulation at the two layer-0 attention RMSNorm sites removes enough inherited `ffn_inp-0` error to close the unchanged FFN up/gate projection gates.

## Nearest evidence and changed coordinate

The [Rung-2B cross-input result](STRAT_01_GIGACHAT31_ENGINE_RUNG2B_CROSS_INPUT_DIAGNOSTIC_RESULT_20260921.md) clears the existing Q4_K×Q8_K up/gate path on exact reference input. The [FFN RMSNorm diagnostic](STRAT_01_GIGACHAT31_ENGINE_RUNG2B_RMSNORM_ACCUMULATOR_DIAGNOSTIC_RESULT_20260921.md) then proves that pinned double accumulation exactly reproduces reference normalization but is insufficient when applied only after the accepted C `ffn_inp-0`: up/gate remain at NRMSE `0.00306126`/`0.00269141`.

The accepted attention path still calls the float-sum helper at two earlier sites:

1. layer-0 attention-input RMSNorm over the exact decoded token embeddings, width 1536;
2. compressed-KV RMSNorm over the first 512 components of `kv_cmpr_pe-0`.

At pinned llama.cpp commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`, both use the same `ggml_float` double accumulator already validated by the FFN diagnostic. The new coordinate is double accumulation at exactly these two upstream sites. The final FFN normalization also uses pinned double semantics, but that component is a frozen measured control rather than a new coordinate. Quantized operators, F16 conversion, RoPE, attention order, cache layout, tensor descriptors, schedules, and numerical gates do not change.

## Immutable bindings

| object | identity |
|---|---|
| accepted GGUF | 6,474,702,976 bytes; SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| pinned llama.cpp | `5b335f413e4f73b0809c4fe39af894efbcc6a0d2` |
| accepted attention-repair adjudication | SHA-256 `e99f0b3e05286b02546a9d9c83f1a8745ba486d0c05087db313f23f9d87f0627` |
| reused pinned-reference manifest | SHA-256 `0f91db875cada58b054dfcfd57161668a92d17bfc2d27c43af206c0652ef372f` |
| accepted Rung-2B run manifest | SHA-256 `5fa1ca6317c3afb466df937f4a9f2e622c38f7d3787d84d2874b2e8afb6ca0b7` |
| FFN RMSNorm diagnostic adjudication | SHA-256 `88bb7f63cd832328c1a4043e457f905304ef520a6144093811d5ebbbe98e8960` |

The fixed tokens, positions, tensor inventory, two schedules (`prefill8`, `cached7p1`), reference tensors/caches, FFN targets, compiler family, and artifact descriptors are inherited by hash from those records. No reference producer may run.

## Diagnostic implementation

Add a diagnostic-only engine entry point; do not alter the existing production Rung-2A or Rung-2B commands. It must:

1. hash and parse the accepted GGUF and validate all inherited identities;
2. execute the existing block-0 attention graph for both schedules, changing only the two upstream RMSNorm accumulators from float to pinned double semantics;
3. preserve the repaired F16 attention and Q4_K×Q8_K V-B paths exactly;
4. emit the same 12 Rung-2A tensors and three cache checkpoints as the accepted attention repair;
5. apply pinned double FFN RMSNorm to candidate `ffn_inp-0`, then run the unchanged Q4_K×Q8_K up and gate projections and emit all three payloads;
6. report source hashes, payload hashes, schedule continuity, the first failing boundary, and `donor_graph_executions=1`.

The runner must reuse immutable reference payloads and independently recompute all metrics. The engine never self-certifies PASS. Build flags remain Clang C11, `-O3 -mavx2 -mfma`, no fast-math, with FP contraction disabled where the pinned arithmetic requires it.

## Gates and controls

The diagnostic is valid only if builds, existing model-free self-tests, inherited artifact/source identities, output completeness, finiteness, shapes, schedule composition, and negative controls pass. The accepted float-path adjudication and FFN-only diagnostic hashes must be revalidated before execution; they are not rerun.

For each schedule, retain the frozen Rung-2A limits:

- every nonterminal tensor and cache: NRMSE `<= 0.002`, normalized maximum `<= 0.01`;
- terminal `ffn_inp-0`: NRMSE `<= 0.001`, normalized maximum `<= 0.005`;
- token-7 prefill/cached continuity: NRMSE `<= 1e-7`, normalized maximum `<= 1e-6`.

The primary propagation decision uses the unchanged Rung-2B limits on double-normalized candidate up and gate, separately in both schedules:

- NRMSE `<= 0.002`;
- normalized maximum `<= 0.01`.

Report without post-hoc gates:

- every candidate Rung-2A metric and delta from the accepted float baseline;
- exactness and metrics at `attn_norm-0` and `kv_cmpr-0`;
- candidate `ffn_inp-0`, `ffn_norm-0`, up, and gate metrics;
- Q8_K block/byte change censuses at both attention projections and both FFN projections;
- first failure by schedule and exact prefill/cached continuity.

A one-byte artifact mutation, a mutated inherited payload identity, swapped up/gate targets, and an omitted upstream RMSNorm coordinate must reject or fail closed.

## Decision rule

- **`UPSTREAM_DOUBLE_RMSNORM_SUFFICIENT_FOR_PROJECTION_GATES`** if every inherited Rung-2A tensor/cache/continuity gate and all four schedule-specific up/gate gates pass. Freeze a production repair confirmation that changes the shared RMSNorm accumulator to pinned semantics, reuses all immutable references, and executes the accepted C artifact once.
- **`UPSTREAM_DOUBLE_RMSNORM_INSUFFICIENT_FOR_PROJECTION_GATES`** if the run is valid but any inherited Rung-2A or primary up/gate gate fails. Do not patch production; use the first changed/failing tensor to freeze a narrower operator-boundary diagnostic.
- **`VOID_UPSTREAM_RMSNORM_PROPAGATION_DIAGNOSTIC`** for any identity, build, completeness, replay, continuity-control, planted-control, or source-binding failure.

Exactly one non-VOID accepted-artifact C execution is allowed. It does not reopen Rung 2A, Rung 2B, the cross-input cell, or the FFN-only RMSNorm cell. Apparatus-only runs execute no donor graph and do not consume the cell.

## Non-claims and stop rule

This is a propagation diagnostic, not a production repair. It makes no claim about complete Rung 2B, Rung 2C, later layers, tokenizer, logits, generation, task quality, RAM, or rate. Document and index the outcome before any shared-helper edit. No `SPEED_LEDGER.md` entry is permitted.
