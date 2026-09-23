# STRAT-01 GigaChat 3.1 Rung-2C layer-1-start cross-input protocol

**Frozen:** 2026-09-23, before implementation or adjudication

**Cell:** `STRAT-01-ENGINE-RUNG2C-LAYER1-START-CROSS-INPUT-DIAGNOSTIC`

**Purpose:** determine whether the accepted block-0 terminal residual is
sufficient for the layer-1 attention failure, or whether the layer-1
pre-attention builder introduces a sufficient residual even from the exact
reference block-0 output.

## Prior result and changed estimand

Rung 2C is closed as `FAIL_ENGINE_RUNG2C`. The subsequent zero-donor
[Q×KV result](STRAT_01_GIGACHAT31_ENGINE_RUNG2C_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md)
is `QUERY_AND_KV_RESIDUALS_INDEPENDENTLY_SUFFICIENT`: exact captured reference
Q/KV passes the C attention+V-B operator, while either captured C query or C
KV alone fails.

This cell moves the boundary back exactly one step. The existing C layer-1
pre-attention builder receives either captured reference `l_out-0` or captured
C `l_out-0`, then emits every registered checkpoint through `kqv_out-1`.
It does not execute block 0, a donor/reference graph, output projection,
residual, MoE, logits, tokenizer, or generation.

## Immutable inputs and provenance

All payloads come from:

`benchmarks/donor_adaptation/engine/results/strat01_gigachat_engine_rung2c_repair1_20260923/`

The source C prefill manifest is pinned by SHA-256
`bdaa49fb9493f891f66fa65c48dd703bae7636b35fd173f54c135326355923c3`;
the source reference prefill manifest by
`d7506adfd7cb20a54da2d406446c5c261452ecf0eecaa3663ef6a4a7acca1451`.
The recovered Rung-2C adjudication and run manifest remain pinned by
`3742bc5982dd47be36a8e422f7b79c9e6716dc628153e44e6f9e90553c9ba62c`
and `898f614a8baaf40716a7aab66e4e31ba51c92da84e133ff24a55e7ccafa60dd4`.

| input | shape | bytes | SHA-256 |
|---|---:|---:|---|
| reference `l_out-0.full` | `[1536,8]` | 49,152 | `385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa` |
| C `l_out-0` | `[1536,8]` | 49,152 | `258232509011378e8470ce6c03cffd51e927f28bfc1ea4f937af07bda9bf0f44` |

The accepted GGUF is 6,474,702,976 bytes with SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
Only the seven frozen layer-1 attention tensors registered by Rung 2C are
admitted. Their parsed names, types, shapes, offsets, file offsets, and spans
must be recorded and fail closed.

Only `prefill8` is read. Exact prefill-versus-7+1 continuity and an identical
failure surface were already established; repeating the second schedule would
duplicate evidence.

## Checkpoints and replay identities

The C-input arm must byte-replay the following immutable C payloads. The
reference-input arm is compared with the corresponding immutable reference
payloads.

| checkpoint | bytes | C SHA-256 | reference SHA-256 |
|---|---:|---|---|
| `attn_norm-1` | 49,152 | `dc668e6d6cc18c2282fc767f3cd57cbd38bb7d08ab496fc85c81f5663cb7e0fe` | `fc9d710b1124f79d7040b7d519930d6f0962f4d071fbb3e41fb7fa827de90606` |
| `q-1` | 196,608 | `95c189683f5b72af6d233bd23c8acb4801c0ae61bda5610e6a80d0100c49fe75` | `276626fc4b0f0339da95e66ac50be0ce3d958d1354ba584b72c63c9d8134b9a8` |
| `kv_cmpr_pe-1` | 18,432 | `eef0d629f07b043af506b1a3220972975d753965c5831d3586a527040821e957` | `448b912739c596ccd745936b67144aae07ccceb12f85ee04c0367bccd4a24ed9` |
| `k_pe-1` | 2,048 | `19ca83be23424f28d11f8a329804ac2965619f731842b6f37ff1d9bdf3b468ea` | `2ffddb8ade46a25cecab1387d295f7a5ad2f7479bd7e8b83a0aa64decc554b68` |
| `kv_cmpr-1` | 16,384 | `58fa280015b838d47df9b1a0ba9064d6cc8eb25000e454330d988ec48d537b57` | `3f3c00b3fe008aecda5e1164b5da2b295d349270b1e339831da1544a849a9387` |
| `q_pe-1` | 65,536 | `89926a0ec73eb8e021d319630a35348d84c33791bd7ee7f4ebe7fb5c0691ae08` | `f58bc7886200116d132787a0d8a997d3dd13ef3a2d7ccd93f66caf5ffdc4ce7b` |
| `q_nope_absorbed_perm-1` | 524,288 | `966dc1942bf4d05280d0d6842cc13eb67d63a070da677014dc588f58ccdd920e` | `18d963a382677e95fa675689121906a71faecdf2283943e3c5ab9be892872375` |
| `Qcur-1` | 589,824 | `3faa0a37833c9d9b9cf92d3bf985d13bf6f09b9c278f81e8813258e86e7d2b2f` | `9cb50fb41e0d33549eb53bd4ae19605705331562d36f975bb2ac82cc4c41cb16` |
| `Kcur-1` | 18,432 | `febfcaa2ca7192a71b7ffca1af26252c9e81b010068478928ed6d56fa78fb0d4` | `73febde321c2ea45a56a5d27a3605b820e08f8b038420a379afc4b9c3f0c72b4` |
| `Vcur-1` | 16,384 | `58fa280015b838d47df9b1a0ba9064d6cc8eb25000e454330d988ec48d537b57` | `3f3c00b3fe008aecda5e1164b5da2b295d349270b1e339831da1544a849a9387` |
| `kqv_out-1` | 196,608 | `7e6d44661f21dee62b449b44e1939b59d8ad0799c4ccf6779d77a01a48e3ee41` | `fa1006c4e2c365d3a5540baebbe4c88180f524d5f69c610fcc3b4205a856d489` |

## Implementation contract

Add a diagnostic-only command that:

1. hashes/parses the accepted GGUF and the two `l_out-0` inputs;
2. copies each input into a fresh `strat01_r2a_arm.embd` buffer;
3. calls the unchanged `strat01_r2c_build_attention_range` for all eight
   tokens and the unchanged prefill `strat01_r2a_run_schedule`;
4. writes the eleven checkpoints above for each input arm;
5. records zero donor graph executions and never self-adjudicates.

No helper may silently fork normalization, quantization, RoPE, K-B, cache,
attention, or V-B semantics. Compile with Clang C11, `-O3 -mavx2 -mfma`, no
fast-math, and the existing FP-contraction-off contract.

## External gates and controls

The C-input arm is an apparatus replay: every checkpoint must match its
captured C SHA-256 exactly. Failure makes the cell VOID.

For the reference-input arm, compare every checkpoint with the frozen
reference target using the unchanged general Rung-2C limits: NRMSE `<= 0.002`
and normalized maximum `<= 0.01`. Both limits must pass separately; no average
may rescue a checkpoint. Report per-token metrics descriptively.

Required controls:

- one-byte mutations of either `l_out-0` input are refused before execution;
- layer-0 tensors or any wrong layer-1 descriptor are refused;
- negating all 1,536 components of reference token 7 must fail at least one
  downstream target gate;
- swapping reference input rows 0 and 7 must fail at least one target gate;
- arm-label swapping must be detected by provenance validation;
- the C-input replay must reproduce the frozen C/reference metrics within
  `1e-12`, including the known `kqv_out-1` failure.

A model-free self-test must cover input decoding, fresh-arm reset, arm routing,
both mutations, descriptor refusal, checkpoint enumeration, and report paths.

## Decision rule

Apply in order:

1. **`VOID_RUNG2C_LAYER1_START_CROSS_INPUT`** for any identity, replay,
   control, source, or apparatus failure.
2. **`BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT`** if the reference-input arm passes
   all eleven checkpoints through `kqv_out-1`. The unchanged layer-1 builder
   is then adequate on exact block-0 output; tighten/localize block-0 terminal
   parity before any layer-1 repair.
3. **`LAYER1_PREATTENTION_RESIDUAL_SUFFICIENT_AT_<CHECKPOINT>`** if the
   reference-input arm first fails before `kqv_out-1`. The named earliest
   checkpoint becomes the only admissible next diagnostic boundary.
4. **`LAYER1_GENERATED_QKV_RESIDUAL_SUFFICIENT`** if every checkpoint through
   `Qcur-1`, `Kcur-1`, and `Vcur-1` passes but `kqv_out-1` fails. Attention/V-B
   remains closed on exact Q/KV; localize which residual generated by the
   layer-1 builder is sufficient using its newly emitted Q/KV, without a donor
   rerun.

Exactly one non-VOID invocation is allowed. Gates and ordering must not change
after values are observed.

## Non-claims and stop rule

This cell cannot repair or promote Rung 2C. It makes no claim about output
projection, MoE, later layers, tokenizer, logits, generation, quality, RAM,
or rate and cannot update `SPEED_LEDGER.md`. Document and index the result
before implementing any production change.
