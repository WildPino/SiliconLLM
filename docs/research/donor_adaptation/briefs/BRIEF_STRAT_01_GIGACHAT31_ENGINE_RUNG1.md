# STRAT-01 GigaChat 3.1 base — C-engine quantized numerical parity rung 1

**Frozen:** 2026-09-21, before implementation or execution on the accepted artifact  
**State:** PROPOSED  
**Purpose:** prove, through `benchmarks/phase60/engine.c`, the byte-level decode and scalar row-matvec interpretation of every material quantized layout in the accepted GGUF. Optimized kernels, model operators, generation, quality, and speed are out of scope.

## 1. Nearest prior cell

The nearest prior is [engine rung 0](../probes/STRAT_01_GIGACHAT31_ENGINE_RUNG0_RESULT_20260921.md), which proves exact artifact identity and agreement on all 414 GGUF descriptors. Its [compatibility audit](../audits/STRAT_01_GIGACHAT31_ENGINE_COMPATIBILITY_AUDIT_20260921.md) explicitly leaves dequantization, matrix orientation, accumulation, and operators open.

Rung 0 does not answer this cell: matching a tensor's name, dimensions, type, and byte span cannot detect a swapped nibble, wrong high-bit plane, incorrect FP16 conversion, or transposed row interpretation. No prior STRAT-01 C path has numerically consumed the accepted weights.

## 2. Changed coordinate and falsifiable hypothesis

**Changed coordinate:** engine fidelity, from descriptor-only inspection to numerical consumption of real payload bytes by a reusable scalar C row-matvec. Donor, artifact, quantization, model graph, and quality evidence remain fixed.

**Hypothesis:** for one preregistered real tensor of each material quantized type, the C engine can decode every quant block to the same canonical float32 stream as pinned `llama.cpp` `gguf-py`, interpret GGUF `ne[0]` as the contiguous row length, and produce fixed-input row-matvec outputs within the frozen normalized error bound.

One type cannot stand in for another. Rung 1 passes only if `Q4_K`, `Q5_0`, and `Q6_K` each pass independently.

## 3. Frozen artifact, reference, tensors, and inputs

- Accepted artifact: `benchmarks/donor_adaptation/density/results/strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf`.
- Required byte size: `6,474,702,976`.
- Required SHA-256: `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Reference implementation: `gguf-py/gguf/quants.py` at pinned `llama.cpp` commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.
- C build: Clang C11, `-O3 -mavx2 -mfma`, no `-ffast-math`; existing `--kselftest` and rung-0 tests must remain passing.

The three fixed cells are:

| Cell | Tensor | Type | GGUF dimensions | Relative offset | Byte span | Absolute file offset |
|---|---|---:|---:|---:|---:|---:|
| G-R1-Q4 | `blk.0.attn_q.weight` | `Q4_K` | `[1536, 6144]` | `279677952` | `5308416` | `285780864` |
| G-R1-Q5 | `blk.0.attn_k_b.weight` | `Q5_0` | `[128, 512, 32]` | `272421888` | `1441792` | `278524800` |
| G-R1-Q6 | `blk.0.ffn_down.weight` | `Q6_K` | `[8960, 1536]` | `286755840` | `11289600` | `292858752` |

For every tensor, `row_length = dims[0]` and `row_count = product(dims[1:])`; rows are contiguous in GGUF payload order. The rank-3 Q5 cell is therefore 16,384 stored rows of length 128. This is a storage/parity interpretation, not yet a claim about the eventual MLA operator call shape.

The input for a row length `K` is generated without a PRNG:

```text
x[i] = (((i * 73 + 19) mod 257) - 128) / 128.0f,  0 <= i < K
```

Every value is exactly representable in binary32. The same vector is applied to every stored row in a cell. This intentionally isolates payload decode and orientation from tokenizer, activation, routing, and model-state effects.

## 4. Apparatus and independent comparison

The engine receives an explicit early-dispatched command, provisionally:

```text
engine --strat01-gguf-rung1 <accepted.gguf> --out-dir <directory>
```

Before reading selected payloads it must reuse the rung-0 fail-closed parser and enforce the compiled size/hash identity. It must look up the three tensors by exact name and re-check type, rank, dimensions, offset, span, and bounds against this brief.

The C implementation must:

1. decode blocks with the exact pinned layouts: Q4_K = 256 values/144 bytes, Q5_0 = 32 values/22 bytes, and Q6_K = 256 values/210 bytes;
2. convert GGML little-endian FP16 scales explicitly, without assuming host struct padding;
3. stream every decoded binary32 value into a canonical little-endian SHA-256 digest, so the comparison covers every weight rather than a sample;
4. perform a scalar row-matvec in increasing element order with a binary32 accumulator and write all row outputs as canonical little-endian float32;
5. emit JSON with artifact identity, source hashes/revision, selected descriptors, block counts, decoded-value counts, per-cell dequant digest, output path/hash, finite counts, extrema, and gate state.

The independent Python adjudicator must read the same raw spans, use the pinned `gguf-py` dequantizer rather than C-derived expected values, generate the frozen input independently, compute the same full-stream digest, and compare every row output. It must also validate the selected descriptors through the pinned GGUF reader.

Tiny synthetic fixture tests must exercise at least one block of each type and include deliberate low/high-nibble and high-bit-plane patterns. Negative controls must show that wrong tensor name/type/dimensions and a one-byte artifact substitution cannot pass. The production command may not accept a caller-supplied expected hash, dimensions, type, or tensor list.

## 5. Gates

### G-R1A — apparatus and regression

- registered C build succeeds;
- pre-existing `--kselftest` and all rung-0 tests pass;
- deterministic Q4_K/Q5_0/Q6_K synthetic codec tests pass, including planted bit-layout traps;
- malformed/substituted identities and descriptor mismatches fail closed.

### G-R1B — complete codec parity

For each of the three real tensors:

- decoded count equals the descriptor element count;
- all C and reference outputs are finite;
- the SHA-256 of the complete canonical float32 dequant stream is identical between C and pinned `gguf-py`.

Digest equality is the primary decode gate. A sampled or aggregate-only match is insufficient.

### G-R1C — row orientation and scalar matvec parity

For every stored row, compare C output `c_r` with the pinned-reference output `p_r`. Record maximum absolute error and the maximum normalized residual

```text
abs(c_r - p_r) / (1 + sum_i abs(w_ri * x_i)).
```

The cell passes when every output is finite and the maximum normalized residual is at most `2e-6`. The report must also record max absolute error, RMSE, worst row, and both output SHA-256 values; output digest equality is informative but not required because reduction implementations may round differently.

Only `PASS_ENGINE_RUNG1` requires G-R1A and all three G-R1B/G-R1C cells. Any genuine decode, descriptor, orientation, or numerical-gate failure is `FAIL_ENGINE_RUNG1` and stops promotion. A compiler/reference-import/filesystem failure before a selected tensor is evaluated is `VOID_APPARATUS`; repair only that apparatus defect and preserve the void record.

## 6. Claim labels, stop rule, and non-claims

- **MEASURED after a valid run:** exact whole-stream codec parity and scalar stored-row matvec parity for the three named accepted-artifact tensors.
- **SOURCE-DERIVED:** quant block definitions and descriptor semantics inherited from the pinned `llama.cpp` source.
- **PROPOSED until measured:** this brief and its gates.
- **VOID:** only a failure that prevents the estimand from being reached, never a failed numerical gate.

Stop immediately after adjudicating rung 1. Do not optimize with AVX2, implement MLA/MoE/model layers, run generation, repeat quality gates, or measure tok/s in this cell. A pass authorizes a separately frozen optimized-kernel or operator rung; it does not prove full-layer semantics, tokenizer parity, logits, quality through C, RAM fit, or accepted-token throughput. Rung 1 must not enter `SPEED_LEDGER.md`.

## 7. Artifact destinations

- C implementation: `benchmarks/phase60/engine.c` and a narrowly named reusable header/source under `benchmarks/phase60/`.
- Tests/runner/reference adjudicator: `benchmarks/donor_adaptation/engine/`.
- Raw accepted run: `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung1_20260921/`.
- Canonical adjudication: `docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG1_RESULT_20260921.md`.

After adjudication, update `docs/research/RESEARCH_INDEX.md`, `docs/research/donor_adaptation/INDEX.md`, and `docs/research/STRATEGIC_10B_20260916/STATUS_20260921.md`. Update `SPEED_LEDGER.md` only after a later registered rate measurement.
