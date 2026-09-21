# STRAT-01 GigaChat 3.1 engine rung 2A protocol

**Date:** 2026-09-21  
**Status:** `PROPOSED` / preregistered; no result  
**Cell:** `STRAT-01-ENGINE-RUNG2A`  
**Adjudication labels:** `PASS_ENGINE_RUNG2A`, `FAIL_ENGINE_RUNG2A`, `VOID_ENGINE_RUNG2A`

## Question and boundary

Does the accepted GigaChat 3.1 Q4 artifact, executed through the `phase60/engine.c` C path, reproduce the pinned llama.cpp reference's block-0 MLA attention semantics—including its compact F16 cache—on two fixed, paired execution forms: one eight-token prefill and a seven-token prefill followed by one cached token?

The nearest prior cell is the measured [`PASS_ENGINE_RUNG1`](STRAT_01_GIGACHAT31_ENGINE_RUNG1_RESULT_20260921.md). Its exact changed coordinate is **isolated scalar stored-row matvec parity → composed block-0 MLA attention semantic parity through the phase60 C engine path**. Rung 1 cannot answer RMSNorm placement, projection composition, YaRN, F16 compact-cache rounding/layout, causal softmax, V-B expansion, output projection, or residual order. This is falsifiable: if C composes any of those operations differently enough to cross a frozen tensor, cache, or continuity gate below, this cell will not pass.

This cell establishes only block-0 attention semantics. It does not establish dense FFN, MoE, full logits, generation, tokenizer parity, quality, RAM, or speed. No rate/timing observation is admissible evidence here.

## Frozen identity and execution configuration

| Item | Frozen value |
|---|---|
| Accepted artifact | `benchmarks/donor_adaptation/density/results/strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf` |
| Artifact size | 6,474,702,976 bytes |
| Artifact SHA-256 | `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| Reference | Clean llama.cpp commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`; CPU only; 1 thread |
| Reference execution | Flash attention disabled; K/Q/V offload disabled; `n_ctx=8`; `n_batch=8`; `n_ubatch=8`; `type_k=F16`; `type_v=F16` |
| C engine entry point | Dedicated `--strat01-gguf-rung2a` mode in `benchmarks/phase60/engine.c` (planned; not asserted to exist) |
| C build semantics | Clang C11, `-O3 -mavx2 -mfma`, with no `-ffast-math`; record the resolved compiler path and complete `clang --version` output. Floating-point contraction choices used by the implementation must be explicit in `CONFIG`. |
| Configuration record | Exact resolved `CONFIG` is mandatory in the raw JSON and execution log. It must record all reference settings above and the C engine's matching semantic settings; any unlisted or mismatched setting makes the run void. |
| Source token lineage | First item of `benchmarks/donor_adaptation/density/results/strat01_gigachat_rollout_v1/q4_rollout.jsonl.core.jsonl`; `item_id=file:data/external/the_stack_python/django/django/db/backends/mysql/introspection.py` |
| Exact token IDs | `[1,72,14,14129,14,2135,1512,2015]` |
| Exact positions | `[0,1,2,3,4,5,6,7]` |
| Token input rule | Use the IDs directly. No text tokenization or token substitution. |

The reference and C engine each run both arms:

| Arm | Decode schedule | Required terminal comparison |
|---|---|---|
| A — `prefill8` | Submit all eight IDs in one decode, at positions 0–7. | Token 7 tensors and terminal `ffn_inp-0`. |
| B — `cached7p1` | Submit IDs 0–6 at positions 0–6; retain the resulting cache; then submit ID 2015 at position 7. | Token 7 tensors and terminal `ffn_inp-0`, plus cache checks after the prefix and after the final token. |

No throughput, elapsed-time, or rate statistic may enter adjudication.
Any operational wall time is diagnostic only and must not be added to `SPEED_LEDGER.md`.

## Source-derived MLA cache contract

For this artifact's MLA layout, implement one compact K-cache row per layer and token, with 576 contiguous IEEE binary16 values in physical logical-vector order: `[512 normalized latent KV, 64 RoPE K]`. There is no separate V-cache allocation for MLA. The cached K representation is viewed as V over its first 512 components. The required local C layout is **layer-major, then slot, then 576 contiguous F16 values**. This is the exact layout for this eight-position experiment only; it makes no claim about long-context allocation.

The pinned source shows the MLA latent and RoPE dimensions and their concatenation/slicing in `src/models/deepseek2.cpp:426-431,517-538,569-580`; MLA omits V storage in `src/llama-kv-cache.cpp:163-164,230-244`; and the context defaults/cache constraints are visible in `src/llama-context.cpp:3672-3677,3727-3729`. The attention graph's relevant construction and output projection are in `src/llama-graph.cpp:2839-2908`; the architecture graph sequence and `ffn_inp` residual are in `src/models/deepseek2.cpp:469-640`. These references support the source-derived architecture statements above, not a claim that the experiment or local C implementation already exists.

For each cache checkpoint, record occupied slot indices and corresponding absolute positions; every occupied row must have length 576 and storage type F16, with no separate V cache. Compare dequantized C cache rows against the reference logical `Kcur-0` values under the general tensor gates below. **Do not require byte equality:** reference cache extraction may expose logical values rather than serialized physical memory.

## Compared block-0 tensors

The reference dump must identify each callback occurrence using at least callback `name`, per-name `ordinal`, operation (`op`), data `type`, and logical `shape`; selection is by the specified post-operation occurrence, not by callback name alone. Because the pinned graph reuses callback names at different points, an absent or ambiguous required occurrence is an apparatus void, not a numerical failure. Both implementations emit raw little-endian float32 payloads for selected logical tensors and a JSON manifest giving each payload's path, name, ordinal/selection, op, type, exact logical shape, byte count, and SHA-256. The C engine emits the same logical tensor set and manifest fields.

| Required logical tensor | Selection meaning |
|---|---|
| `attn_norm-0` | Block-0 attention RMSNorm output. |
| `q-0` (direct Q after reshape) | Direct query projection after its registered reshape; retain this name as the logical tensor label for this protocol. |
| `kv_cmpr_pe-0` | Combined compressed latent and positional-key projection, before splitting. |
| `k_pe-0` | Post-RoPE positional key component. |
| `kv_cmpr-0` | Compressed latent KV after its RMSNorm. |
| `q_pe-0` | Post-RoPE positional query component. |
| `q_nope_absorbed_perm-0` | Absorbed non-positional query projection after its registered permutation. |
| `Qcur-0` | Composed query tensor entering attention. |
| `Kcur-0` | Compact MLA key tensor entering/stored for attention. |
| `Vcur-0` | Latent value view used for MLA attention; not a separate cached V row. |
| `kqv_out-0` | Attention result after causal attention and V-B expansion, before output projection. |
| `ffn_inp-0` | Attention output projection added to the block input residual. |

The reference manifest additionally preserves all callback identity fields so an independent adjudicator can verify the exact post-op occurrence selected for every logical tensor. The fixed set of shapes must be recorded from the reference and match exactly in C; no shape may be inferred from a flattened payload alone.

## Frozen numerical and semantic gates

For compared tensor vectors `c` (C) and `r` (reference), define:

`NRMSE = sqrt(sum((c-r)^2) / max(sum(r^2), 1e-30))`

`normalized_max = max(abs(c-r)) / max(max(abs(r)), 1e-6)`

All tensors below must have identical logical shapes, be finite (no NaN or Inf), and pass independently in **both arms**. For every compared tensor, `NRMSE <= 2e-3` and `normalized_max <= 1e-2`. The terminal `ffn_inp-0` has the stricter limits `NRMSE <= 1e-3` and `normalized_max <= 5e-3`.

| Gate | Frozen acceptance |
|---|---|
| General tensor parity | Every required tensor, each arm: NRMSE ≤ `2e-3` **and** normalized max ≤ `1e-2`. |
| Terminal residual parity | `ffn_inp-0`, each arm: NRMSE ≤ `1e-3` **and** normalized max ≤ `5e-3`. |
| Prefill/cache continuity | Within each implementation, token-7 `ffn_inp-0` from `prefill8` vs `cached7p1`: NRMSE ≤ `2e-6` **and** normalized max ≤ `1e-5`. Apply separately to reference and C. |
| Cache invariants | After cached prefix and final token: exact occupied slots and positions; row length 576; F16 storage; no separate V cache. Dequantized C cache rows vs reference logical `Kcur-0` pass the general tensor gates. |
| Completeness and validity | Every required tensor exists exactly once after occurrence disambiguation, has exact logical shape, and is finite. Missing/ambiguous callback occurrence or missing dump is VOID. |

No tolerance may be relaxed, no failed numerical cell may be repaired by tuning, and no arm may be selectively omitted.

## Mandatory eight-item control card

1. **Nearest prior cell and canonical artifact link:** measured `PASS_ENGINE_RUNG1`; canonical result [STRAT-01 GigaChat 3.1 engine rung 1 result](STRAT_01_GIGACHAT31_ENGINE_RUNG1_RESULT_20260921.md).
2. **Exact changed coordinate:** isolated scalar stored-row matvec parity → composed block-0 MLA attention semantic parity through `benchmarks/phase60/engine.c`.
3. **Why prior does not answer this cell:** Rung 1 does not test RMSNorm placement, composed projections, YaRN, compact F16 cache rounding/layout, causal softmax, V-B expansion, output projection, or residual ordering; this cell can fail those independently frozen comparisons.
4. **Frozen identity and paired controls:** exact accepted artifact, size, SHA-256, pinned reference commit, source-derived fixed token IDs/positions, `CONFIG`, and paired `prefill8` / `cached7p1` arms are specified above. Both engines execute both arms.
5. **Separate gates:** this rung has block-0 engine parity and cache/continuity gates only. Quality, rank/rollout, and rate are not measured here and cannot be inferred from this protocol.
6. **Claim labels:** prior rung-1 outcome is **MEASURED**; pinned source layout interpretation is **SOURCE-DERIVED**; this protocol and all future implementation/run details are **PROPOSED** until adjudicated; apparatus failures are **VOID**. There is no rung-2A result yet.
7. **Void and stop rule:** identity/config/reference-dump/shape/missing-output apparatus failure is `VOID_ENGINE_RUNG2A`. Preserve every void. A narrow apparatus repair requires a new raw run directory and must not erase or overwrite prior evidence. Stop after one accepted-artifact execution and adjudication. Do not tune thresholds or rerun the donor to repair a scientific failure.
8. **Raw and canonical records:** planned raw directory `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2a_20260921/`; future canonical adjudication `docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2A_RESULT_20260921.md`.

## Apparatus negative controls

The test suite must include these controls. They test the apparatus; they are not donor repetitions.

| Control | Required behavior |
|---|---|
| Wrong artifact hash | Reject before scientific comparison. |
| Changed token ID | Reject the altered input identity or make a synthetic fixture fail. |
| Changed position | Reject the altered position identity or make a synthetic fixture fail. |
| Changed cache precision/layout | Reject the altered cache contract or make a synthetic fixture fail. |
| Omitted residual | Make a synthetic fixture fail the terminal residual comparison. |

Negative-control outcomes must be recorded separately from the accepted-artifact execution and must not be represented as donor measurements.

## Execution and evidence contract

Implementation and test filenames may be planned by the implementer; this protocol does not assert that any rung-2A code, tests, raw directory, or result already exists. The execution wrapper must validate artifact bytes and hash, fixed token IDs and positions, pinned reference revision, and exact `CONFIG` before accepting outputs. It must preserve stdout/stderr, exit status, executable/build identity, source revision, full raw JSON, both JSON manifests, and every float32 payload under the planned raw directory. Each payload hash is checked against its manifest before numerical adjudication. The adjudicator records per-tensor/per-arm shape, finiteness, NRMSE, normalized maximum, cache checkpoints/invariants, continuity statistics, control outcomes, and one final label in the future canonical result.

The accepted artifact is executed once. If apparatus identity/config/dump/shape requirements fail, adjudicate VOID and preserve the run. Any narrow apparatus repair uses a new raw directory and retains the prior void. If apparatus is valid but any frozen numerical or semantic gate fails, adjudicate FAIL; do not modify thresholds or repeat the donor execution. Only if both arms, all required tensors, cache invariants, and the within-implementation continuity gate pass may the result be `PASS_ENGINE_RUNG2A`.

## Adjudication and next step

| Label | Rule |
|---|---|
| `PASS_ENGINE_RUNG2A` | Both arms pass every required tensor gate, cache invariant/comparison, and within-implementation terminal continuity gate; apparatus and negative controls are valid. |
| `FAIL_ENGINE_RUNG2A` | Apparatus is valid, but any frozen numerical or semantic gate fails. |
| `VOID_ENGINE_RUNG2A` | Artifact identity, configuration, reference dump/selection, shape, or required-output apparatus is invalid or incomplete. |

After a pass only, the next cell is a separately frozen **2B block-0 dense SwiGLU** protocol; after that, and only after its own adjudication, **2C one MoE block**. Neither is part of this cell. A 2A pass remains limited to block-0 attention semantics and is not evidence of full-model execution, generation, quality, RAM fit, or speed.

## Pinned source references

Line references below were verified against clean pinned llama.cpp commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`:

- `C:/Users/giosa/AppData/Local/Temp/siliconllm-llama-bind-5b335f4/src/models/deepseek2.cpp:426-431,469-640` — MLA ranks/dimensions, attention graph composition, and residual output.
- `C:/Users/giosa/AppData/Local/Temp/siliconllm-llama-bind-5b335f4/src/llama-graph.cpp:2839-2908` — cache writes, attention result callback, and attention output projection in the generic attention builder.
- `C:/Users/giosa/AppData/Local/Temp/siliconllm-llama-bind-5b335f4/src/llama-kv-cache.cpp:163-164,230-244` — MLA determination and K-only cache allocation (no V allocation for MLA).
- `C:/Users/giosa/AppData/Local/Temp/siliconllm-llama-bind-5b335f4/src/llama-context.cpp:3672-3677,3727-3729` — default F16 cache types and MLA K/V type compatibility constraint.
