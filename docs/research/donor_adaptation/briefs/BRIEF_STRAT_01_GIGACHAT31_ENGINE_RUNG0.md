# STRAT-01 GigaChat 3.1 base — C-engine parity rung 0

**Frozen:** 2026-09-21, before implementation or execution on the accepted artifact  
**State:** PROPOSED  
**Purpose:** establish the exact accepted-GGUF identity and loader contract inside `benchmarks/phase60/engine.c`; no numerical operator, generation, quality, or speed claim is in scope.

## 1. Nearest prior cell

The nearest prior is the bounded [engine compatibility audit](../audits/STRAT_01_GIGACHAT31_ENGINE_COMPATIBILITY_AUDIT_20260921.md), whose rung 0 requires a machine-readable inventory through the actual C-engine path. The accepted artifact and lineage are pinned by the [source-binding result](../probes/STRAT_01_GIGACHAT31_SOURCE_BINDING_PROTOCOL_20260919.md), [fresh BPB result](../probes/STRAT_01_GIGACHAT31_FRESH_BPB_PROTOCOL_20260919.md), [PIQA result](../probes/STRAT_01_GIGACHAT31_PIQA_RESULT_20260920.md), and [document rollout](../probes/STRAT_01_GIGACHAT31_DOCUMENT_ROLLOUT_RESULT_20260921.md).

The earlier Python header census is source inspection, not this cell: it never exercised `phase60/engine.c`, did not create a reusable C loader boundary, and could not prevent that engine from accepting a substituted file.

## 2. Changed coordinate and falsifiable hypothesis

**Changed coordinate:** execution path, from Python/llama.cpp inspection to native parsing and rejection inside `benchmarks/phase60/engine.c`; the artifact, donor, quantization, and model graph remain fixed.

**Hypothesis:** a portable C extension can read the accepted GGUF without loading its 6.47 GB payload, hash and identify the exact file, enumerate every metadata value and tensor descriptor needed by later rungs, and reject every planted identity/layout violation before any model allocation or inference.

Failure of any required identity, metadata, tensor, bounds, or negative-control check falsifies rung 0. It says nothing about whether the operators can later achieve parity or speed.

## 3. Frozen inputs and execution contract

- Accepted artifact: `benchmarks/donor_adaptation/density/results/strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf`.
- Required byte size: `6,474,702,976`.
- Required SHA-256: `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- Required GGUF architecture/version: `deepseek2`, GGUF v3, quantization version 2, file type 15.
- Required base boundary: 26 blocks and 414 tensors; no MTP block 26 tensors.
- Required tensor-type census: 233 `Q4_K`, 26 `Q5_0`, 26 `Q6_K`, and 129 `F32`.
- Required primary dimensions: vocabulary 128,256; width 1,536; 26 layers; dense FFN 8,960; expert width 1,280; 64 routed experts; top-4; one shared expert; 32 heads; KV-LoRA rank 512; query head 128 non-RoPE + 64 RoPE; BOS 1; EOS 2.
- Build baseline: Clang C with `-O3 -mavx2 -mfma`, no `-ffast-math`; the existing synthetic modes must remain buildable and their `--kselftest` must still pass.

The new CLI surface shall be explicit and early-dispatched, provisionally:

```text
engine --strat01-gguf-inspect <accepted.gguf> --json <inventory.json>
```

It must not fall through to the synthetic `E1M1`/`E4M1` loader. The accepted hash is compiled into the frozen STRAT-01 inspection mode; accepting an arbitrary `deepseek2` file is not a pass.

## 4. Output schema and controls

The JSON output must record at least: engine source hash or Git revision, command, input path, byte size, SHA-256, GGUF magic/version/alignment/data offset, metadata count, tensor count, all tensor names/types/dimensions/offsets/byte spans, type census, required scalar metadata, explicit MTP exclusion, and final gate status. Integer fields must remain integers; large offsets may not pass through floating-point JSON values.

Bounds checks must prove for every tensor that:

1. the descriptor and strings are fully inside the file;
2. rank and dimension products cannot overflow `uint64_t`/`size_t`;
3. type block divisibility and byte-size arithmetic are valid;
4. aligned payload start plus tensor offset/span remains inside the file;
5. names are unique and tensor payload spans do not overlap;
6. unknown GGUF value or tensor types fail closed.

Required controls:

- deterministic parser self-tests on small generated fixtures;
- truncated header, invalid string length, duplicate tensor name, out-of-bounds tensor, wrong type/block divisibility, and unsupported type must all be refused;
- a valid tiny fixture must parse independent of the host working directory;
- an exact copy of the accepted file may pass; a one-byte-changed copy or wrong expected size/hash must be refused before promotion;
- existing `engine --kselftest` remains passing after integration.

The full accepted-file hash may be computed in a separate pass from header parsing. Peak additional memory for inspection must be bounded independently of file size; mapping or allocating the whole GGUF is forbidden for rung 0.

## 5. Gates

### G-R0A — build and regression

- `engine.c` builds with the registered Clang flags;
- parser self-tests pass;
- the pre-existing `--kselftest` passes unchanged.

### G-R0B — exact artifact identity

- byte size and SHA-256 equal the frozen values;
- architecture/version/base boundary equal the frozen values;
- no substituted or modified file is silently accepted.

### G-R0C — complete layout inventory

- exactly 414 unique descriptors are emitted;
- complete names, ranks, dimensions, types, offsets, and spans are present;
- the frozen type census and required metadata all match;
- every descriptor passes overflow, alignment, block-size, bounds, and non-overlap checks;
- MTP tensors are absent and the report states that exclusion.

Only `PASS_ENGINE_RUNG0` requires all three gates. Any failed scientific invariant is `FAIL_ENGINE_RUNG0`; malformed fixtures and intentionally damaged copies are expected refusals, not model failures. A compiler/toolchain failure before the parser can run is `VOID_TOOLCHAIN` and may receive only a narrow apparatus repair.

## 6. Stop rule and non-claims

Stop immediately after rung 0 adjudication. Do not implement Q4_K/Q5_0/Q6_K matvecs in this cell. A pass authorizes a separately frozen rung 1 for one named tensor and fixed numerical inputs; it does not authorize full-model implementation by inference, a quality rerun, or timing.

This cell cannot establish tokenizer parity, dequantization correctness, tensor orientation, MLA/YaRN/MoE semantics, logits, greedy decoding, quality, memory fit, or accepted-token throughput. It must not enter `SPEED_LEDGER.md`.

## 7. Artifact destinations

- Implementation: `benchmarks/phase60/engine.c` plus a narrowly named reusable C header/source under `benchmarks/phase60/` if separation is needed.
- Tests/runner: under `benchmarks/donor_adaptation/engine/`, with generated tiny fixtures confined to a temporary directory.
- Raw accepted run: `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung0_20260921/`.
- Canonical adjudication: `docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG0_RESULT_20260921.md`.

After adjudication, update `docs/research/RESEARCH_INDEX.md` and `docs/research/STRATEGIC_10B_20260916/STATUS_20260921.md`. Update `SPEED_LEDGER.md` only after a later registered rate measurement.
