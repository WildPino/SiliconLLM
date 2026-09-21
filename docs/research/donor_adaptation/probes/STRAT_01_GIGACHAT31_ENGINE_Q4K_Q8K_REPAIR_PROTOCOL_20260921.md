# STRAT-01 GigaChat 3.1 engine Q4_K×Q8_K repair protocol

**State:** EXECUTED; CLOSED `PASS_ENGINE_Q4K_Q8K_REPAIR` — see [result](STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REPAIR_RESULT_20260921.md)
**Date frozen:** 21 September 2026
**Scope:** implement and confirm the project-engine projection operator authorized by the closed Rung-2A projection diagnostic; no donor graph, downstream attention, quality, or speed

## Authorized change

The measured [projection diagnostic](STRAT_01_GIGACHAT31_ENGINE_RUNG2A_PROJECTION_DIAGNOSTIC_RESULT_20260921.md) establishes that pinned llama.cpp Q4_K×Q8_K reproduces both captured reference projections bit-for-bit, while the current dequantized-Q4_K/F32 operator reproduces the engine's failing outputs.  This protocol authorizes exactly one changed coordinate:

> For Q4_K stored-row matrix multiplication, quantize each F32 activation row to Q8_K and evaluate Q4_K×Q8_K dot semantics instead of dequantizing weights and multiplying directly by F32.

The implementation must be standalone C inside the project engine path.  It may be derived from the pinned MIT-licensed GGML algorithms with an attribution comment, but final engine execution may not depend on or link against llama.cpp/GGML.

This cell initially confirms only the two first-failure projections `blk.0.attn_q.weight` and `blk.0.attn_kv_a_mqa.weight`.  It does not silently promote other quantized types or declare downstream attention repaired.

## Immutable accepted-artifact inputs

Reuse without modification:

- GGUF size `6,474,702,976`, SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`;
- source Rung-2A run manifest SHA-256 `0f91db875cada58b054dfcfd57161668a92d17bfc2d27c43af206c0652ef372f`;
- captured reference `attn_norm-0` SHA-256 `c8c7bd47772b1f153f28183892795b9bc91be978f5473bb322f869b2c10c1efd`;
- captured reference Q target SHA-256 `4dc1424d3f93651acaa152bd57222754c132ab3e43330d286c61e9739caff64b`;
- captured reference KV target SHA-256 `6a364dd45c12142fb45ab90287caa14874ed40160089e716e23bd58d3785653c`;
- Q4_K Q descriptor `[1536,6144]`, file offset `285780864`, byte span `5308416`;
- Q4_K KV descriptor `[1536,576]`, file offset `279966592`, byte span `497664`;
- pinned independent oracle: clean llama.cpp `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`.

All identities, byte counts, shapes, and finite-value checks are mandatory.  No donor model process may execute.

## Required implementation and controls

The project engine must add bounded-storage representations and functions for:

1. one Q8_K block: binary32 scale, 256 signed quants, and sixteen signed 16-element sums;
2. Q8_K activation quantization with the pinned signed-maximum/tie behavior and nearest-integer rule;
3. Q4_K×Q8_K stored-block dot semantics, including packed Q4 values, packed 6-bit scales/minima, Q8_K block sums, and binary32 reduction;
4. batch stored-row multiplication that quantizes each activation row once and reuses it across output rows;
5. a diagnostic-only CLI that reads the two exact GGUF spans plus the immutable reference input and writes token-major Q/KV F32LE outputs without running embeddings or the donor graph.

Production code must reject unsupported dimensions, non-Q4_K tensors, non-multiples of 256, allocation overflow/failure, short reads, and non-finite activation input.  The old dequant-F32 calculation may remain only as an explicitly named test/control path; the repaired Q4_K full-matrix path must not silently fall back to it.

## Model-free oracle tests

Before accepted-artifact confirmation, an independent test helper linked to pinned GGML must compare project C against the oracle on fixed synthetic fixtures:

- all-zero activation block;
- positive and negative signed maxima;
- equal-absolute-value tie ordering;
- half-integer rounding neighborhoods;
- deterministic nonperiodic finite vectors;
- planted Q4_K scale/minimum/high-nibble layouts.

For every fixture:

- Q8_K `d`, all 256 `qs`, and all 16 `bsums` must be byte-identical to pinned `quantize_row_q8_K_ref`;
- project Q4_K×Q8_K dot versus pinned active `ggml_vec_dot_q4_K_q8_K` must satisfy NRMSE `<= 2e-6` and normalized maximum `<= 1e-5` over the fixture population;
- a one-byte Q8_K scale mutation, wrong packed-scale interpretation, and token/row transpose must each make the relevant gate fail;
- legacy Rung 0, Rung 1, Rung 2A model-free tests and the 73,024-check kernel self-test must remain passing.

## Accepted offline confirmation

Run the committed project C diagnostic CLI once on the immutable captured reference input and exact Q/KV weight spans.  Compare its token-major outputs with the captured pinned-reference targets using:

`NRMSE = sqrt(mean((c-r)^2)) / max(sqrt(mean(r^2)), 1e-12)`

`normalized_max = max(abs(c-r)) / max(max(abs(r)), 1e-6)`

Both Q and KV must independently pass NRMSE `<= 2e-6` and normalized maximum `<= 1e-5`.  The runner must preserve source/binary hashes, build logs, test logs, command records, payload hashes, per-tensor metrics, and an empty-or-explicit error list.  It must record `donor_executions=0`.

## Adjudication

| Label | Frozen rule |
|---|---|
| `PASS_ENGINE_Q4K_Q8K_REPAIR` | Identity, all model-free oracle/negative/legacy controls, exact Q8_K bytes, and both accepted Q/KV numerical gates pass. |
| `FAIL_ENGINE_Q4K_Q8K_REPAIR` | Apparatus and controls are valid, but either accepted Q or KV misses a frozen numerical gate. |
| `VOID_ENGINE_Q4K_Q8K_REPAIR` | Any identity, source revision, build, test/control, shape, hash, output, or finiteness requirement is invalid or incomplete. |

Preserve every void in a distinct directory.  A narrow apparatus repair may not change the algorithm, inputs, metrics, or thresholds.  Stop after one non-VOID accepted offline confirmation; do not tune from observed errors.

## Mandatory eight-item control card

1. **Nearest prior cell and canonical artifact:** measured `ATTRIBUTED_Q8K_ACTIVATION_QUANTIZATION`; [canonical result](STRAT_01_GIGACHAT31_ENGINE_RUNG2A_PROJECTION_DIAGNOSTIC_RESULT_20260921.md).
2. **Exact changed coordinate:** engine Q4_K full-matrix dequant-F32 multiplication -> standalone exact Q8_K activation plus Q4_K×Q8_K multiplication.
3. **Why prior does not answer this cell:** the prior diagnostic used pinned GGML as an oracle and identifies the defect; it does not prove the new standalone C implementation matches that oracle.
4. **Frozen identity and paired controls:** exact artifact/spans/captured tensors, pinned revision, byte-level Q8_K tests, dot tests, negative controls, and Q/KV accepted targets are fixed above.
5. **Separate gates:** operator implementation fidelity only; no downstream attention, quality, RAM, or rate claim.
6. **Claim labels:** root cause is **MEASURED**; pinned algorithm inspection is **SOURCE-DERIVED**; this implementation/protocol is **PROPOSED** until adjudicated; invalid apparatus is **VOID**.
7. **Void and stop rule:** preserve voids; only narrow apparatus repair; one non-VOID accepted offline cell; no threshold tuning or donor execution.
8. **Raw and canonical records:** planned raw directory `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_q4k_q8k_repair_20260921/`; future canonical result beside this protocol.

## Consequence boundary

A pass authorizes a separately frozen changed-coordinate block-0 attention confirmation using the repaired operator.  It does not convert the old Rung-2A result, authorize 2B, or establish full-engine quality/rate.  A fail keeps the implementation boundary open and requires a new source-derived diagnostic rather than threshold changes.

No `SPEED_LEDGER.md` entry is due.

## Apparatus repair A — pre-execution source-directory correction

The first launch at commit `3d1861a` produced `VOID_ENGINE_Q4K_Q8K_REPAIR`
before any command record, model read, or projection.  The runner compared the
frozen successful-manifest hash against the older raw directory
`strat01_gigachat_engine_rung2a_20260921/`, while that hash and every frozen
captured tensor above belong to
`strat01_gigachat_engine_rung2a_repair2_20260921/`, as already recorded by the
closed projection diagnostic.  Repair A changes only that directory constant
and sends the next immutable record to
`strat01_gigachat_engine_q4k_q8k_repair1_20260921/`.  Inputs, algorithms,
metrics, thresholds, labels, and the one-non-VOID stopping rule are unchanged.
