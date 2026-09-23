# STRAT-01 GigaChat 3.1 engine Rung 2C protocol

**State:** frozen before implementation and execution.

**Question:** starting from the accepted block-0 terminal state, can the
standard C path reproduce one complete block-1 MLA-attention plus routed and
shared MoE layer from the accepted pretrained GigaChat 3.1 GGUF?

This is the first MoE engine-parity cell. It is not another block-0 repair,
not a carving experiment, and not a full-model, quality, RAM, or speed test.

## Frozen identity and predecessors

| item | frozen value |
|---|---|
| accepted GGUF | 6,474,702,976 bytes; SHA-256 `68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb` |
| pinned reference | llama.cpp `5b335f413e4f73b0809c4fe39af894efbcc6a0d2` |
| block-0 production adjudication | `58f661d9461f3dc17b2a52f830dcc9f36423621528941d475e23ee2aed8e0bf2` |
| block-0 production manifest | `7b8de95a3766e4f76bde37f4038ece2986b9067ad9ac7a0e4cc11855a3062497` |
| tokens | `[1,72,14,14129,14,2135,1512,2015]` |
| positions | `[0,1,2,3,4,5,6,7]` |
| schedules | `prefill8` and `cached7p1` |
| cache | per-layer K-only MLA rows, 576 F16 values `[latent512,rope64]`; no V allocation |

The new C run must reproduce the accepted production `l_out-0` full-payload
SHA-256 `258232509011378e8470ce6c03cffd51e927f28bfc1ea4f937af07bda9bf0f44`
in both schedules before any block-1 result is admissible. The new pinned
reference producer must independently reproduce its historical `l_out-0`
full-payload SHA-256
`385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa`
in both schedules. A mismatch is apparatus VOID, not a numerical Rung-2C
failure.

## Source-derived block-1 contract

The accepted GGUF and bound source configuration specify:

- hidden width 1,536 and 26 blocks;
- block 0 dense, every block from block 1 onward MoE;
- 64 routed experts, exactly four selected per token;
- one shared expert;
- routed and shared intermediate width 1,280;
- router weight `blk.1.ffn_gate_inp.weight`, shape `[1536,64]`, F32;
- expert-selection bias `blk.1.exp_probs_b.bias`, shape `[64]`, F32;
- routed gate/up tensors `[1536,1280,64]`, Q4_K;
- routed down tensor `[1280,1536,64]`, Q6_K;
- shared gate/up tensors `[1536,1280]`, Q4_K;
- shared down tensor `[1280,1536]`, Q6_K.

For each token, the exact router semantics are:

1. `logits = gate_inp^T x` in the pinned matrix orientation;
2. `probs = sigmoid(logits)`;
3. add `exp_probs_b` only to a separate selection-score copy;
4. select the ordered top four expert IDs from the biased scores;
5. gather the selected weights from the original unbiased sigmoid `probs`;
6. clamp their sum from below at `6.103515625e-5` and normalize the four weights;
7. compute each selected expert as `down(silu(gate) * up)` using the pinned
   Q4_K×Q8_K and Q6_K×Q8_K semantics;
8. weight each expert output, then sum the four slots in selected-slot order;
9. independently compute the shared expert with the same SwiGLU ordering;
10. add routed and shared outputs, then add the block-1 attention residual.

The artifact metadata fixes sigmoid gating, normalized weights, scale `1.0`,
one group and one used group. Therefore no group mask or extra routed scale is
active. Biasing the final expert weights, using softmax, omitting
renormalization, changing top-k order, or applying weights before the expert
FFN are different operators and are forbidden.

## Exact changed coordinate

The nearest accepted cell ends at dense block-0 `l_out-0`. Rung 2C adds:

- block-1 attention using the already accepted production MLA operators, now
  with layer-1 weights and a second compact cache row;
- a rank-3 selected-expert Q4_K/Q6_K path;
- discrete sigmoid-plus-bias top-4 routing;
- normalized unbiased route weights;
- shared-expert SwiGLU and routed/shared aggregation.

No preceding experiment evaluates this composition. STRAT-03 studied a
synthetic/local learned router geometry and cannot answer parity for this
pretrained donor. The older donor co-activation studies likewise do not prove
the executable GGUF operator semantics.

## Required logical checkpoints

Every checkpoint is selected by callback name, block index, ordinal,
operation, data type, and exact logical shape. The producer records all
callback occurrences; absent or ambiguous selection is VOID. Float payloads
are raw little-endian F32. Expert IDs are raw little-endian I32. The cached arm
is assembled prefix-7 then final-1 into the same token-major logical shapes as
prefill.

### Layer-1 attention and boundary

Require the layer-1 counterparts of the accepted Rung-2A set:

`attn_norm-1`, `q-1`, `kv_cmpr_pe-1`, `k_pe-1`, `kv_cmpr-1`, `q_pe-1`,
`q_nope_absorbed_perm-1`, `Qcur-1`, `Kcur-1`, `Vcur-1`, `kqv_out-1`, and
`ffn_inp-1`.

Their logical meanings, layouts, and dimensions are identical to the block-0
contract except for the layer index. `ffn_inp-1` is `[1536,8]` and is the
terminal attention boundary before the MoE.

### Layer-1 MoE

| logical checkpoint | type and logical shape | source-derived operation |
|---|---|---|
| `ffn_norm-1` | F32 `[1536,8]` | pinned double-accumulation RMSNorm |
| `ffn_moe_logits-1` | F32 `[64,8]` | F32 router matrix product |
| `ffn_moe_probs-1` | F32 `[64,8]` | sigmoid of unbiased logits |
| `ffn_moe_probs_biased-1` | F32 `[64,8]` | selection-only bias add |
| `ffn_moe_topk-1` | I32 `[4,8]` | ordered top-four expert IDs |
| `ffn_moe_weights-1` | F32 `[4,8]` | selected unbiased sigmoid weights |
| `ffn_moe_weights_norm-1` | F32 `[4,8]` | clamped-sum normalization |
| `ffn_moe_up-1` | F32 `[1280,4,8]` | selected Q4_K×Q8_K up projections |
| `ffn_moe_gate-1` | F32 `[1280,4,8]` | selected Q4_K×Q8_K gate projections |
| `ffn_moe_swiglu-1` | F32 `[1280,4,8]` | `silu(gate) * up` |
| `ffn_moe_down-1` | F32 `[1536,4,8]` | selected Q6_K×Q8_K down projections |
| `ffn_moe_weighted-1` | F32 `[1536,4,8]` | per-slot normalized route weighting |
| `ffn_moe_out-1` | F32 `[1536,8]` | ordered sum of four routed slots |
| `ffn_up-1` | F32 `[1280,8]` | shared Q4_K×Q8_K up projection |
| `ffn_gate-1` | F32 `[1280,8]` | shared Q4_K×Q8_K gate projection |
| `ffn_swiglu-1` | F32 `[1280,8]` | shared `silu(gate) * up` |
| `ffn_shexp-1` | F32 `[1536,8]` | shared Q6_K×Q8_K down projection |
| `ffn_out-1` | F32 `[1536,8]` | routed plus shared output |
| `l_out-1` | F32 `[1536,8]` | MoE output plus `ffn_inp-1` residual |

The reference manifest must preserve exact callback identity and any harmless
reshape/view needed to expose the logical shapes. It may not recompute the
router or experts in Python.

## Frozen gates

For every floating tensor, compute:

`NRMSE = sqrt(sum((c-r)^2) / max(sum(r^2), 1e-30))`

`normalized_max = max(abs(c-r)) / max(max(abs(r)), 1e-6)`

| gate | frozen acceptance |
|---|---|
| start-state identity | both implementations reproduce their respective frozen `l_out-0` hashes exactly in both schedules |
| router IDs | `ffn_moe_topk-1` is byte-identical to reference for every token and schedule, including selected-slot order |
| general floating parity | every required F32 checkpoint in each arm: NRMSE ≤ `2e-3` and normalized max ≤ `1e-2` |
| terminal attention parity | `ffn_inp-1` in each arm: NRMSE ≤ `1e-3` and normalized max ≤ `5e-3` |
| terminal block parity | `l_out-1` in each arm: NRMSE ≤ `1e-3` and normalized max ≤ `5e-3` |
| schedule continuity | within each implementation, token-7 prefill versus cached for every required checkpoint: NRMSE ≤ `2e-6` and normalized max ≤ `1e-5`; router IDs exact |
| cache invariants | exact layer/slot/position occupancy for layers 0 and 1; 576 F16 values per occupied row; no V allocation; dequantized layer-1 rows pass general parity |
| completeness | exact shape/type/count, finite F32 values, complete manifests, and no duplicate or ambiguous selection |

No threshold may be relaxed after observation. A router-ID mismatch is a
scientific FAIL when apparatus and inputs are valid, even if downstream
floating output happens to remain within tolerance.

## Mandatory apparatus and causal controls

Model-free tests must make each wrong implementation fail:

1. use biased rather than unbiased sigmoid values as route weights;
2. replace sigmoid plus top-k normalization with softmax;
3. omit selected-weight normalization;
4. permute selected expert IDs without applying the same permutation to every
   selected tensor and weight;
5. transpose or select the wrong expert slice in rank-3 Q4_K/Q6_K storage;
6. swap routed gate and up;
7. omit the shared expert;
8. omit routed/shared addition;
9. omit the final residual;
10. mutate the block-0 start hash or layer-1 cache index.

The accepted-artifact run is admissible only if all controls fire, all current
Rung-2A/Rung-2B/combined self-tests still pass, the 73,024-check legacy kernel
self-test passes, and source controls prove that production and diagnostics
share the accepted Q4_K/Q6_K/Q5_0 primitives rather than duplicate them.

## Execution and evidence contract

Before either scientific producer runs, an apparatus-only mode must compile,
run every model-free/self-test control, validate the artifact and predecessor
hashes, and record zero donor/reference graph executions.

After apparatus acceptance:

- run the pinned reference producer exactly once; that invocation emits both
  schedules and records two graph schedules;
- only after a valid reference record, run the standard C producer exactly
  once; that invocation emits both schedules and records two graph schedules;
- hash and validate every manifest and payload before adjudication;
- preserve compiler identity, commands, stdout/stderr, exit codes, source and
  binary hashes, complete `CONFIG`, callback records, caches, execution counts,
  controls, per-checkpoint metrics, and final status;
- use unique raw directories for every VOID; never overwrite evidence.

A reference apparatus failure does not authorize a C donor run. A valid
numerical FAIL is terminal for this coordinate and cannot be rerun after code
or threshold tuning. A narrow pre-scientific apparatus repair must keep the
same estimand and receive a new output directory.

Planned canonical raw directory:
`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_20260923/`.

## Adjudication

| label | rule |
|---|---|
| `PASS_ENGINE_RUNG2C` | all identity, exact-router, tensor, cache, continuity, completeness, and control gates pass in both schedules |
| `FAIL_ENGINE_RUNG2C` | apparatus is valid but any frozen numerical, router-ID, cache, continuity, or causal gate fails |
| `VOID_ENGINE_RUNG2C` | any artifact, predecessor, source, build, config, reference, callback-selection, payload, finiteness, completeness, control, or execution-accounting requirement is invalid |

## Non-claims and stop rule

This cell establishes at most blocks 0 and 1 on the standard C path. It does
not establish later MoE layers, tokenizer/logits, generation, language-model
quality through C, HumanEval, RAM fit, or accepted-token rate. Do not update
`SPEED_LEDGER.md`.

After any non-VOID result, document and index it before deciding the next
coordinate. A PASS does not authorize copying block 1 across all 25 MoE layers
without a separately frozen depth-propagation strategy; a FAIL must be
localized using preserved payloads without repeating the full donor graph.

## Pinned source references

Verified against llama.cpp
`5b335f413e4f73b0809c4fe39af894efbcc6a0d2`:

- `src/models/deepseek2.cpp:630-691` — block residual, MoE call, shared expert,
  routed/shared add, and terminal residual;
- `src/llama-graph.cpp:1993-2352` — sigmoid/bias selection, top-k, unbiased
  gathered weights, normalization, selected expert projections, weighting,
  and ordered aggregation;
- accepted GGUF inventory
  `benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung0_20260921/inventory.json`
  — exact block-1 tensor shapes, types, offsets, and byte spans;
- bound source configuration
  `benchmarks/donor_adaptation/density/results/strat01_gigachat_source_189fff27/config.json`
  — 64 routed experts, top-4, one shared expert, sigmoid/noaux routing, and
  normalized weights.
