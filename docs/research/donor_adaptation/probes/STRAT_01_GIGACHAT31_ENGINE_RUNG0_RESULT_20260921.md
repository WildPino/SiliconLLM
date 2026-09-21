# STRAT-01 GigaChat 3.1 base — C-engine parity rung 0 result

**Date:** 2026-09-21

**Outcome:** `PASS_ENGINE_RUNG0`

**Scope:** exact accepted-Q4 file identity and complete GGUF metadata/tensor-layout inventory through the early-dispatched `benchmarks/phase60/engine.c` path. This result is not a numerical kernel, operator, tokenizer, generation, quality, or throughput result.

## Frozen question and lineage

The preregistered [rung-0 brief](../briefs/BRIEF_STRAT_01_GIGACHAT31_ENGINE_RUNG0.md) changed only the execution path: prior Python/llama.cpp inspection became a fail-closed native C boundary in `phase60/engine.c`. It did not change donor, artifact, quantization, model graph, corpus, or quality threshold.

- Brief/index commit: `d6145f3`.
- Frozen implementation/run commit: `10a44b5332afa23b754e6ca43d400bb915632124`.
- Accepted Q4 GGUF: 6,474,702,976 bytes; SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
- C source SHA-256: `df199820c56cb76674ade4e127fd7e206eb8fe5f90d88ae5b1d673a8a27a05e1`.
- Inspector header SHA-256: `3d577150ea442f5fd5ae993621024b421e000a9f66727e67b109184aa74fd711`.
- Compiled binary SHA-256: `f87fe9522c32a2f77edce4c838443404f72c97b734898adf52bc0c0ebeffe5c4`.

The implementation adds an explicit `--strat01-gguf-inspect <path> --json <output>` dispatch before the historical E1M1/E4M1 option parser and loader. It streams SHA-256 in bounded memory, parses GGUF v3 metadata and descriptors, validates integer overflow, quantization block divisibility, alignment, file bounds, unique names, non-overlap, frozen metadata/census, and explicit MTP exclusion, then emits all 414 descriptors as integer-valued JSON.

## Pre-run apparatus review

The delegated first draft passed its synthetic fixtures but was not run on the accepted file. Independent review found two fixture omissions: the accepted GGUF omits `general.alignment` and therefore uses the GGUF default 32, and it contains scalar `FLOAT32`/`BOOL` metadata. The draft incorrectly required a serialized alignment key and rejected those known scalar types. Both were repaired before the only accepted-artifact run; the fixtures were expanded to cover absent/default alignment, floats, bools, strings, string arrays, and integer arrays.

This is pre-estimand apparatus history, not a void model run. No accepted artifact was inspected by the defective draft and no failed result was erased.

## G-R0A — build and regression

**PASS.** The final apparatus was independently rerun with:

```powershell
.\.venv\Scripts\python.exe -m unittest -v benchmarks\donor_adaptation\engine\test_strat01_engine_rung0.py
```

All 5 test methods passed. They cover the registered Clang C11 build, the pre-existing `--kselftest`, a valid fixture from an unrelated working directory, truncated/invalid strings, duplicate names, out-of-bounds spans, wrong quantization block divisibility, unsupported tensor and metadata types, exact-copy acceptance, one-byte hash tampering, and wrong compile-time size/hash identities. The production binary has no CLI or environment identity override.

The accepted run compiled with `clang -std=c11 -O3 -mavx2 -mfma ... -lm`, return code 0. Compilation took 2.704022 seconds. These durations are operational diagnostics, not rate measurements.

## G-R0B — exact artifact identity

**PASS.** Native C inspection reports:

- byte size 6,474,702,976;
- SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- magic `GGUF`, version 3, effective alignment 32;
- architecture `deepseek2`, base block count 26;
- no identity substitution or MTP tensor accepted.

The run returned 0 and printed `PASS_ENGINE_RUNG0`. Its 25.962140-second duration is dominated by full-file identity hashing and must not be interpreted as model loading latency, inference throughput, or a clean-box speed observation.

## G-R0C — complete layout inventory

**PASS.** The C path emits:

- metadata count 47;
- tensor count 414;
- data offset 6,102,912;
- type census 129 F32, 26 Q5_0, 233 Q4_K, and 26 Q6_K;
- every tensor name, rank, dimension, type, relative offset, absolute file offset, and byte span;
- `mtp_excluded=true`.

The independent verifier compares the C inventory against the GGUF reader from pinned llama.cpp commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`. All 414 descriptors match exactly; both descriptor streams have SHA-256 `c64f56486ecae3231b30d041ed1bf20f6a5a94942817164b586fff22ffe0d67c`, with zero mismatches. A second comparison against the 414-tensor BF16 source-binding report finds zero name/shape mismatches.

## Evidence classes and adjudication

- **MEASURED:** the committed C engine path accepts the exact pinned Q4 file, rejects planted malformed/substituted inputs, and emits an inventory exactly matching the pinned reader.
- **SOURCE-DERIVED:** the architectural interpretation of the names and dimensions remains inherited from the source binding, pinned DeepSeek2 implementation, and compatibility audit.
- **PROPOSED:** rung 1, one named tensor and fixed-input numerical parity for each material quantized layout, is now authorized but not yet implemented or measured.
- **VOID:** none for the accepted-artifact estimand. Malformed fixtures are planted expected refusals; the two pre-run review findings were corrected before execution.

All three preregistered gates pass; the result is therefore `PASS_ENGINE_RUNG0`.

## Raw artifacts

Directory: `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung0_20260921/`.

| Artifact | SHA-256 |
|---|---|
| `inventory.json` | `aa2eb7f0ade5d82ddeec9d5621fd8b58cebc4d74920a43fb915204d81e315791` |
| `run_manifest.json` | `c0375c9c9fb9bee0b5b0a829675a423224585c3dfba26c1bcde804cab69cba96` |
| `reference_compare.json` | `feb3ecab0530640bdaddd6f5281fb8e637685a8e3f331afac8ae7b09cbfeeedc` |
| `run.stderr.log` | `26d5d8aec1124ac2519c9c0440e0bf27f5303ef31fedc9f103235d9282fa45e7` |

The runner preserves compiler/version, commands, return codes, environment, revision, source/binary hashes, and explicit non-claims. Empty compile/stdout logs are retained locally. The generated executable is reproducible from the committed sources and is not required as canonical evidence.

## Boundary and next gate

Rung 0 proves only identity and layout. It does not prove dequantization, matrix orientation, accumulation precision, Q4_K/Q5_0/Q6_K kernels, MLA, YaRN, routing, shared experts, tokenizer parity, logits, greedy decoding, quality through C, RAM, or accepted-token rate. It does not enter `SPEED_LEDGER.md`.

The next non-duplicate coordinate is rung 1: a frozen named tensor, deterministic fixed input, scalar reference built from the exact GGUF blocks, and numerical comparison through a reusable C matvec. Q4_K, Q5_0, and Q6_K must each receive direct evidence before their behavior is generalized.
