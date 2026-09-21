# STRAT-01 GigaChat 3.1 engine rung 2B protocol

**Date:** 2026-09-21  
**Status:** `PROPOSED` / preregistered; no result  
**Cell:** `STRAT-01-ENGINE-RUNG2B`  
**Adjudication labels:** `PASS_ENGINE_RUNG2B`, `FAIL_ENGINE_RUNG2B`, `VOID_ENGINE_RUNG2B`

## Question and boundary

Does the accepted GigaChat 3.1 Q4 artifact, executed through the
`benchmarks/phase60/engine.c` C path, reproduce pinned llama.cpp block-0 dense
SwiGLU semantics from the already accepted `ffn_inp-0` through FFN RMSNorm,
parallel Q4_K gate/up projections, SiLU gating, Q6_K down projection, and the
block residual output?

The nearest prior changed-coordinate cell is the measured
[`PASS_ENGINE_ATTENTION_VB_REPAIR`](STRAT_01_GIGACHAT31_ENGINE_ATTENTION_VB_REPAIR_RESULT_20260921.md),
which closes Rung 2A through `ffn_inp-0`.  The exact changed coordinate here is
**accepted block-0 attention residual input -> complete block-0 dense SwiGLU and
residual output**.  Rung 2A cannot answer FFN RMSNorm placement, Q4_K gate/up
operator semantics, parallel SwiGLU operand order, Q6_K down-projection
semantics, or the final residual order.  This is falsifiable at six frozen
checkpoints and by four causal apparatus controls below.

This cell establishes only block-0 dense FFN semantics.  It does not establish
the routed/shared MoE branch used from block 1 onward, later blocks, logits,
generation, tokenizer parity, quality, RAM, or speed.  No timing observation is
admissible evidence.

## Frozen identity and execution configuration

| Item | Frozen value |
|---|---|
| Accepted artifact | `benchmarks/donor_adaptation/density/results/strat01_gigachat_q4_97045b2/GigaChat3.1-10B-A1.8B-q4_K_M.gguf` |
| Artifact size | 6,474,702,976 bytes |
| Artifact SHA-256 | `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| Reference | Clean llama.cpp commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`; CPU only; 1 thread |
| Reference execution | Flash attention disabled; K/Q/V offload disabled; requested `n_ctx=8`, required resolved allocation `resolved_n_ctx=256`; `n_batch=8`; `n_ubatch=8`; `type_k=F16`; `type_v=F16` |
| C engine entry point | Dedicated Rung-2B mode extending the accepted `benchmarks/phase60/engine.c` STRAT-01 path; exact flag is implementation detail but must be recorded in `CONFIG` |
| C build semantics | Clang C11, `-O3 -mavx2 -mfma`, no `-ffast-math`; record compiler path, complete `clang --version`, source revision, and contraction choices |
| Source token lineage | First item of `benchmarks/donor_adaptation/density/results/strat01_gigachat_rollout_v1/q4_rollout.jsonl.core.jsonl`; `item_id=file:data/external/the_stack_python/django/django/db/backends/mysql/introspection.py` |
| Exact token IDs | `[1,72,14,14129,14,2135,1512,2015]` |
| Exact positions | `[0,1,2,3,4,5,6,7]` |
| Token input rule | Use IDs directly; no tokenization or substitution |

The reference and C engine each execute both frozen schedules:

| Arm | Decode schedule | Required comparison |
|---|---|---|
| A — `prefill8` | IDs 0-7 at positions 0-7 in one decode | All six block-0 FFN checkpoints; terminal token 7 emphasized |
| B — `cached7p1` | IDs 0-6, retain cache, then ID 2015 at position 7 | All six token-7 checkpoints and `l_out-0` schedule continuity |

The reference trace may be generated once for this newly extended callback
set.  The accepted artifact may be executed once through C after apparatus
validation.  Existing Rung-2A raw tensors are immutable inputs/evidence and
must not be regenerated merely to support this cell.

## Source-derived dense-FFN contract

Pinned `src/models/deepseek2.cpp:640-687` establishes this sequence for block
0: add attention output and block input as `ffn_inp`; RMS-normalize it; because
`il < n_layer_dense_lead`, call parallel `LLM_FFN_SILU`; then add the FFN output
to `ffn_inp` and expose `l_out`.  Pinned `src/llama-graph.cpp:1748-1905`
establishes the parallel FFN internals: project `up` and `gate` from the same
normalized input, compute `SiLU(gate) * up` through `ggml_swiglu_split`, and
apply `down`.

The accepted artifact fixes the relevant block-0 tensors:

| Tensor | Logical matrix dimensions | GGUF type |
|---|---:|---|
| `blk.0.ffn_norm.weight` | `[1536]` | F32 |
| `blk.0.ffn_gate.weight` | `[1536, 8960]` | Q4_K |
| `blk.0.ffn_up.weight` | `[1536, 8960]` | Q4_K |
| `blk.0.ffn_down.weight` | `[8960, 1536]` | Q6_K |

No bias is present in this dense cell.  The implementation must consume the
stored quantized rows with the CPU operator semantics selected by the pinned
type traits.  A dequantize-to-F32 shortcut may be retained as diagnostic code,
but it is not presumed equivalent and cannot replace the production path
without passing every frozen checkpoint.

## Frozen callback identities and payloads

The reference dump identifies every selected callback by `name`, per-name
ordinal, operation, data type, and logical shape.  Selection is by exact
post-operation occurrence.  An absent or ambiguous occurrence is VOID.  Both
implementations emit raw little-endian float32 logical payloads and a manifest
with payload path, identity fields, shape, byte count, and SHA-256.

| Required checkpoint | Frozen meaning | Expected logical leading dimension |
|---|---|---:|
| `ffn_norm-0` | FFN RMSNorm output | 1536 |
| `ffn_up-0` | Q4_K up projection of the normalized input | 8960 |
| `ffn_gate-0` | Q4_K gate projection of the same normalized input | 8960 |
| `ffn_swiglu-0` | `SiLU(ffn_gate) * ffn_up` | 8960 |
| `ffn_out-0` | Q6_K down projection, before residual addition | 1536 |
| `l_out-0` | `ffn_out + ffn_inp`; complete block-0 output | 1536 |

For `prefill8`, the trailing logical dimension is 8 for every checkpoint.  For
the final decode of `cached7p1`, it is 1.  Shape, finiteness, identity, and
payload-hash checks precede numerical adjudication.

## Frozen numerical and semantic gates

For C vector `c` and reference vector `r`:

`NRMSE = sqrt(sum((c-r)^2) / max(sum(r^2), 1e-30))`

`normalized_max = max(abs(c-r)) / max(max(abs(r)), 1e-6)`

| Gate | Frozen acceptance |
|---|---|
| General checkpoint parity | Every required checkpoint in each applicable arm: NRMSE <= `2e-3` and normalized max <= `1e-2` |
| Terminal block parity | `l_out-0` in each arm: NRMSE <= `1e-3` and normalized max <= `5e-3` |
| Schedule continuity | Within each implementation, token-7 `l_out-0` from `prefill8` vs `cached7p1`: NRMSE <= `2e-6` and normalized max <= `1e-5` |
| Completeness | Exact shapes, finite values, unambiguous callbacks, matching payload hashes, complete `CONFIG` |

Every gate must pass independently.  Tolerances may not be relaxed after a
run, and a numerical failure may not be repaired by repeating or selecting an
arm.

## Mandatory eight-item control card

1. **Nearest prior and canonical artifact:** measured
   `PASS_ENGINE_ATTENTION_VB_REPAIR`; canonical result linked above and raw
   adjudication SHA-256
   `e99f0b3e05286b02546a9d9c83f1a8745ba486d0c05087db313f23f9d87f0627`.
2. **Exact changed coordinate:** accepted `ffn_inp-0` -> dense block-0
   RMSNorm, parallel Q4_K gate/up, SwiGLU, Q6_K down, and residual `l_out-0`.
3. **Why prior work does not answer it:** Rung 2A stops before every operation
   under test; isolated Q4_K diagnostics do not compose SwiGLU or exercise the
   block-0 Q6_K down projection and residual.
4. **Frozen identity and paired controls:** artifact, hash, pinned reference,
   tokens, positions, two schedules, and exact `CONFIG` are frozen above.
5. **Separate gates:** callback parity and schedule continuity only; no MoE,
   full-model, quality, memory, or rate claim.
6. **Claim labels:** prior result is **MEASURED**; graph interpretation is
   **SOURCE-DERIVED**; this protocol and unexecuted implementation are
   **PROPOSED**; apparatus-invalid attempts are **VOID**.
7. **Void and stop rule:** preserve every void in a unique directory.  A narrow
   apparatus repair may receive a new directory, but no failed scientific cell
   may be tuned or repeated.  Stop after one non-VOID accepted-artifact C
   execution and adjudication.
8. **Raw and canonical records:** planned raw directory
   `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2b_20260921/`;
   planned canonical result
   `docs/research/donor_adaptation/probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2B_RESULT_20260921.md`.

## Apparatus negative controls

The test suite must establish causality before the accepted-artifact run:

| Control | Required behavior |
|---|---|
| Gate/up swap or independently perturbed gate input | Fail `ffn_swiglu` or a downstream checkpoint |
| Omitted SiLU | Fail `ffn_swiglu` or a downstream checkpoint |
| Mutated Q6_K down operator/input | Fail `ffn_out` or `l_out` |
| Omitted final residual | Fail `l_out` |
| Wrong artifact hash, token, position, callback, shape, or payload hash | Reject as apparatus invalid before scientific adjudication |

Controls operate on synthetic/copied fixtures and are not donor measurements.
Their outcomes must be recorded separately in the raw adjudication.

## Execution, adjudication, and next step

The wrapper must preserve build logs, execution logs, exit codes, executable
and source identity, complete `CONFIG`, manifests, payloads, control outcomes,
per-checkpoint metrics, and final label.  Reference and C payload hashes must
be verified before metrics are computed.

| Label | Rule |
|---|---|
| `PASS_ENGINE_RUNG2B` | Both schedules pass all required checkpoint, terminal, continuity, identity, and causal-control gates |
| `FAIL_ENGINE_RUNG2B` | Apparatus is valid but any frozen numerical or semantic gate fails |
| `VOID_ENGINE_RUNG2B` | Identity, configuration, reference selection, payload, shape, build, or required-output apparatus is invalid/incomplete |

After a pass only, the next engine-compatibility cell is separately frozen
Rung 2C: one block-1 routed-expert plus shared-expert MoE layer.  A Rung-2B
pass remains a one-dense-block semantic result and does not authorize a
full-model, generation, quality, RAM, or speed claim.

