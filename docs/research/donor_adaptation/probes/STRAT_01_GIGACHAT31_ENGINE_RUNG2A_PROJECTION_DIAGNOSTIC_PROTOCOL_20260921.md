# STRAT-01 GigaChat 3.1 engine Rung-2A projection diagnostic protocol

**State:** EXECUTED; CLOSED as `ATTRIBUTED_Q8K_ACTIVATION_QUANTIZATION`
**Date frozen:** 21 September 2026  
**Scope:** offline attribution of the already measured Rung-2A Q/KV projection mismatch; no donor execution, no acceptance-rung repetition, and no speed claim

The frozen gates below are unchanged.  See the [canonical result](STRAT_01_GIGACHAT31_ENGINE_RUNG2A_PROJECTION_DIAGNOSTIC_RESULT_20260921.md).

## Question and nearest evidence

The accepted STRAT-01 artifact already has a valid `FAIL_ENGINE_RUNG2A`.  In both frozen arms, `attn_norm-0` passes while the first projection outputs fail: `q-0` has NRMSE `0.0029148289138709186` and `kv_cmpr_pe-0` has NRMSE `0.004722034291199872`.  Prefill-versus-7+1 continuity is exact inside each implementation, so the next unresolved boundary is the full Q4_K matrix multiplication rather than cache sequencing.

This diagnostic asks whether that boundary is explained by one exact implementation difference:

- the current C path dequantizes each Q4_K weight block to F32 and accumulates products against the F32 activation; while
- pinned llama.cpp maps Q4_K `vec_dot_type` to Q8_K, converts each F32 activation row to Q8_K, and evaluates the Q4_K×Q8_K dot-product kernel.

That mechanism is **SOURCE-DERIVED**, not yet a measured root cause.  The diagnostic must distinguish it from wrong Q4_K decode/scales, wrong row/layout interpretation, or a residual kernel/accumulation difference.

## Immutable evidence and identity

No model graph, tokenizer, donor inference, or generation process may run.  The only scientific inputs are the exact GGUF weight spans and immutable tensors captured by the third Rung-2A run:

- source run: `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_repair2_20260921/`;
- source run manifest SHA-256: `0f91db875cada58b054dfcfd57161668a92d17bfc2d27c43af206c0652ef372f`;
- captured-run git head: `a653aeef48156b92ae471e5daa28675ffb6fef2d`;
- pinned llama.cpp commit: `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`;
- GGUF size: `6,474,702,976` bytes;
- GGUF SHA-256: `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- `blk.0.attn_q.weight`: Q4_K, dimensions `[1536,6144]`, file offset `285780864`, byte span `5308416`;
- `blk.0.attn_kv_a_mqa.weight`: Q4_K, dimensions `[1536,576]`, file offset `279966592`, byte span `497664`.

The frozen prefill-8 payloads are:

| Producer | Logical tensor | Shape / order | SHA-256 |
|---|---|---|---|
| pinned reference | `attn_norm-0` | `[1536,8]`, token-major after decoding the GGML payload | `c8c7bd47772b1f153f28183892795b9bc91be978f5473bb322f869b2c10c1efd` |
| pinned reference | `q-0` | `[192,32,8]`, token/head/feature | `4dc1424d3f93651acaa152bd57222754c132ab3e43330d286c61e9739caff64b` |
| pinned reference | `kv_cmpr_pe-0` | `[576,8]`, token/feature | `6a364dd45c12142fb45ab90287caa14874ed40160089e716e23bd58d3785653c` |
| C engine | `attn_norm-0` | `[1536,8]`, token/feature | `f746d41ff1d909d70d09b241c2a3f2d66fd837b51dbeec5ca95ba618d3456d9e` |
| C engine | `q-0` | `[192,32,8]`, token/head/feature | `78dafb9fff7edd265a81693707cfd3d300b41cb8eeb9aab3d9a6c15c81ae383d` |
| C engine | `kv_cmpr_pe-0` | `[576,8]`, token/feature | `5325d4dca9deb7fa528895530e73871676653764156a0c363f46c7ff350e4caf` |

Every payload, source manifest, weight descriptor, pinned source revision, and model size/hash must validate before numerical adjudication.  Failure is apparatus `VOID_PROJECTION_DIAGNOSTIC`; it is not evidence for or against a hypothesis.

## Frozen computations

For each of the two registered weight matrices and all eight tokens, compute these paths without invoking the donor model:

1. **D32/C-control.** Decode every Q4_K block to binary32 and accumulate binary32 products in stored-row order, using the captured C `attn_norm-0` as input.  Compare with the captured C projection output.  This checks that the diagnostic represents the measured C path.
2. **D32/reference-input.** Run the same dequantized-F32 computation using the captured reference `attn_norm-0`.  Compare with the captured reference projection output.  This isolates input drift from multiplication semantics.
3. **Q8K/reference reproduction.** Quantize each captured reference activation row with the exact pinned `quantize_row_q8_K` implementation and evaluate every stored row with the exact pinned `ggml_vec_dot_q4_K_q8_K` implementation.  Compare with the captured reference projection output.
4. **F64 audit.** Decode the Q4_K blocks once and accumulate products in binary64 before final binary32 storage, using the reference input.  This is descriptive attribution for accumulation error only; it cannot by itself decide the primary label.

The Q8_K codec and Q4_K×Q8_K dot product must be linked from the clean pinned source.  They may not be copied, reimplemented, or imported from the project C engine.  The D32 path may reuse the already validated project Q4_K decoder, but the runner must record this dependency and its source hash.

All output arrays must be finite and have exact token-major logical shapes `[8,6144]` for Q and `[8,576]` for KV.  The reshaped reference Q payload must be proven to be a view of that same contiguous `[8,6144]` order; no transpose chosen from observed errors is allowed.

## Metrics and frozen gates

For candidate `c` and captured target `r`:

`NRMSE = sqrt(mean((c-r)^2)) / max(sqrt(mean(r^2)), 1e-12)`

`normalized_max = max(abs(c-r)) / max(max(abs(r)), 1e-6)`

The following gates are fixed before execution:

| Gate | Acceptance |
|---|---|
| Identity/completeness | Every frozen hash, descriptor, shape, element count, source revision, and finite-value check passes. |
| D32/C-control | For Q and KV independently: NRMSE `<= 2e-6` and normalized max `<= 1e-5` against the captured C output. |
| Q8K/reference reproduction | For Q and KV independently: NRMSE `<= 2e-6` and normalized max `<= 1e-5` against the captured pinned-reference output. |
| Old-gate counterfactual | D32/reference-input fails the old general Rung-2A gate (`NRMSE <= 2e-3` and normalized max `<= 1e-2`) for both Q and KV. |
| Negative controls | Each registered corruption is detected and at least one of the two tight reproduction metrics exceeds its limit. |

The tight reproduction gate is the already preregistered Rung-2A within-implementation continuity tolerance.  It is not derived from this diagnostic's future observations.

## Hypotheses and adjudication

| Label | Frozen rule |
|---|---|
| `ATTRIBUTED_Q8K_ACTIVATION_QUANTIZATION` | Identity and D32/C controls pass; exact Q8K/reference reproduction passes for both Q and KV; D32/reference-input fails the old gate for both Q and KV; all negative controls fire.  This establishes that activation quantization/dot semantics account for the first measured projection mismatch. |
| `PARTIAL_Q8K_ATTRIBUTION` | Apparatus is valid and controls pass, but exact Q8K reproduction closes only one projection or D32 fails the old gate for only one projection.  Record per-tensor outcomes; do not patch the engine globally. |
| `REJECT_Q8K_AS_SUFFICIENT` | Apparatus is valid and D32/C controls pass, but exact Q8K/reference reproduction misses the tight gate for either projection.  Investigate graph/kernel dispatch, layout, or another source-derived difference under a new frozen diagnostic. |
| `VOID_PROJECTION_DIAGNOSTIC` | Any identity, build, hash, shape, finiteness, D32/C-control, or negative-control requirement is invalid or incomplete. |

No threshold may be relaxed and no path may be changed after seeing accepted-artifact metrics.  Preserve a void under a distinct raw directory.  Run the accepted offline cell once; after a non-VOID result, close this protocol.

## Registered negative controls

Before the accepted offline cell, model-free tests must prove that the adjudicator rejects:

1. a one-byte source-payload hash mutation;
2. a Q/KV row-count or payload-order substitution;
3. a token-0/token-1 output swap;
4. a deterministic mutation of one Q8_K quantized activation byte before a dot product;
5. a wrong pinned llama.cpp revision or dirty pinned checkout.

These are apparatus tests, not donor measurements.  Their output is recorded separately.

## Mandatory eight-item control card

1. **Nearest prior cell and canonical artifact link:** measured and closed `FAIL_ENGINE_RUNG2A`; [canonical result](STRAT_01_GIGACHAT31_ENGINE_RUNG2A_RESULT_20260921.md).
2. **Exact changed coordinate:** composed attention acceptance rung -> offline first-projection attribution on immutable captured inputs/outputs and exact Q4_K weight spans.
3. **Why prior does not answer this cell:** Rung 2A localizes the first failure after `attn_norm-0` but does not distinguish F32 activation use from GGML's Q8_K activation conversion, row/layout errors, or accumulation differences.
4. **Frozen identity and paired controls:** exact GGUF identity, two weight descriptors, six payload hashes, captured-run manifest hash, pinned llama.cpp revision, D32/C and Q8K/reference paired reproductions are fixed above.
5. **Separate gates:** this diagnostic measures numerical attribution only.  It does not measure Rung-2A acceptance, downstream attention, quality, RAM, or speed.
6. **Claim labels:** Rung-2A evidence is **MEASURED**; kernel mapping and proposed mechanism are **SOURCE-DERIVED**; this protocol is **PROPOSED** until executed; invalid apparatus is **VOID**.
7. **Void and stop rule:** preserve every void; permit only narrow apparatus repair in a new raw directory; stop after one non-VOID offline adjudication.  Never rerun the donor or retune Rung-2A.
8. **Raw and canonical records:** planned raw directory `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_projection_diagnostic_20260921/`; this protocol will be amended with a clearly separated result section or paired canonical result after adjudication.

## Consequence boundary

Only `ATTRIBUTED_Q8K_ACTIVATION_QUANTIZATION` authorizes an engine implementation step that replaces the affected Q4_K dequant-F32 full-matrix path with exact Q8_K activation quantization and Q4_K×Q8_K dot semantics.  That implementation still requires model-free tests and a new parity confirmation; this diagnostic itself does not reopen or convert the closed Rung-2A result.

No speed-ledger entry is due for any outcome.

## Pinned source references

Verified against clean llama.cpp commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`:

- `ggml/src/ggml-cpu/ggml-cpu.c:307-317` — Q4_K selects `ggml_vec_dot_q4_K_q8_K` and `GGML_TYPE_Q8_K`;
- `ggml/src/ggml-cpu/quants.h:32,54` — active Q8_K quantizer and Q4_K×Q8_K dot declarations;
- `ggml/src/ggml-quants.h:34,63` — exported reference Q8_K quantizer and dequantizer;
- `ggml/src/ggml-cpu/quants.c:696-764` — scalar/generic Q4_K×Q8_K dot semantics;
- `benchmarks/phase60/strat01_gguf_rung2a.h:173-182` — current C stored-row dequantized-F32 matrix multiplication.
