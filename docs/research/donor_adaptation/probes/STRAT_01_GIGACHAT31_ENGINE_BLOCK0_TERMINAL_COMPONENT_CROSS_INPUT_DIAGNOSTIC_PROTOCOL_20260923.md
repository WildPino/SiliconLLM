# STRAT-01 GigaChat 3.1 block-0 terminal-component cross-input protocol

**Frozen:** 2026-09-23, before implementation or adjudication

**Cell:** `STRAT-01-ENGINE-BLOCK0-TERMINAL-COMPONENT-CROSS-INPUT-DIAGNOSTIC`

**Purpose:** determine whether the accepted block-0 attention-stream term,
the dense-FFN term, or only their joint residual is sufficient to make the
unchanged layer-1 pre-attention path fail.

## Prior result and changed estimand

The zero-donor
[layer-1-start result](STRAT_01_GIGACHAT31_ENGINE_RUNG2C_LAYER1_START_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md)
is `BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT`: exact reference `l_out-0` makes all
eleven layer-1 checkpoints pass, whereas the accepted C `l_out-0` byte-replays
the Rung-2C failure. Layer 1 is therefore closed and must not be modified.

This cell splits the already captured terminal addition

`l_out-0 = ffn_inp-0 + ffn_out-0`

into four offline F32 compositions. It then feeds each composed `l_out-0`
through the already qualified, unchanged C layer-1 builder. This is a new
additive-component intervention, not another block-0 or Rung-2C producer run.
It executes no donor/reference graph and does not recompute either component.

The hybrid arms identify sufficiency at the captured terminal-addend boundary.
They do not claim that a reference `ffn_out-0` is the natural FFN output of a C
`ffn_inp-0`, or vice versa.

## Immutable inputs and provenance

Only `prefill8` payloads are admitted. Exact prefill-versus-7+1 continuity and
an identical failure surface were already established; a second schedule
would duplicate evidence.

| component | shape | bytes | SHA-256 | frozen source |
|---|---:|---:|---|---|
| C `ffn_inp-0` | `[1536,8]` | 49,152 | `13721a425bf93e87aa313b9ea7718fdcc2d2ce25f9ec38403eee9fc6cce9847e` | combined RMSNorm + Q5/Q8 candidate |
| C `ffn_out-0` | `[1536,8]` | 49,152 | `8433164833fe4daecd24a840bb4cd70c41d390a15655abe479ef058248a6fb4f` | block-0 production integration |
| C `l_out-0` replay target | `[1536,8]` | 49,152 | `258232509011378e8470ce6c03cffd51e927f28bfc1ea4f937af07bda9bf0f44` | block-0 production integration |
| reference `ffn_inp-0` | `[1536,8]` | 49,152 | `baa389195bff42ff92593d92650ed9defdd1b1382469e5dde3516dfa82b76ec1` | pinned Rung-2B reference |
| reference `ffn_out-0` | `[1536,8]` | 49,152 | `f5514fe59d64b517a685ec70004a75152a5f4d511417148f34e4d13b3134bef4` | pinned Rung-2B reference |
| reference `l_out-0` replay target | `[1536,8]` | 49,152 | `385073c91f472dd9ffc1c86bcb63c6ed50256a6d5645e240d61ccdb613d814aa` | pinned Rung-2B reference |

The C `ffn_inp-0` source manifest is pinned by SHA-256
`4deff5b5999c9af9f862f2850dd7ff6b94ae67092ecbcba339ac10bf90cf0c09`.
Its raw source trace is the preserved candidate-source-hash VOID with
adjudication and run-manifest hashes
`8eca798bb38706b2d9f0a56ebebf5934f035200c5ae0e983ae4e801f56219e66`
and `4964efe45dbbde5b7fd80d957b25279c8099f267c6b40e2aac573858d232db5e`;
the canonical zero-graph offline repair that accepts and binds that payload is
`0a0aac6a58fe4bbf37d414d80d5681afdf357c7fda846596bea10685092c1b77`,
with run manifest
`8c2bc3f03d4755dbc1de8548211bdf0e8a9ac792d4905aa2232ffd1a113f7869`.
The block-0 production C manifest, adjudication, and run manifest are pinned by
`67b1c187ec129e4be32a37fa7792d7f6499843c105e9c73e535f9d95824118a3`,
`58f661d9461f3dc17b2a52f830dcc9f36423621528941d475e23ee2aed8e0bf2`,
and `7b8de95a3766e4f76bde37f4038ece2986b9067ad9ac7a0e4cc11855a3062497`.
The pinned reference prefill manifest is
`7873d9e59b3a089232dd6b15b0ba2fcfc4d37df2fb1f32d0488d6914d43882a9`.

The accepted GGUF is 6,474,702,976 bytes with SHA-256
`68a8732fb5cee04f83ebffd7924e15c534d4442c5a43d2ba9e2041fe310b8deb`.
The qualified layer-1-start adjudication and run manifest remain pinned by
`7c36f4df4d5b3b7e8947d77a3d9e83ce1f6b224d398870367a4888672d08f2da`
and `e71643986e2124341217ae8f7a27923e3a7454c71b72a084ac1c00b927d9a681`.

Before this protocol was frozen, an independent scalar binary32 addition
check established both replay identities with zero mismatched bytes:

- C `ffn_inp-0 + ffn_out-0` -> C `l_out-0` SHA-256 `25823250...bf0f44`;
- reference `ffn_inp-0 + ffn_out-0` -> reference `l_out-0` SHA-256
  `385073c9...d814aa`.

The apparatus must repeat these identities internally and fail closed; the
pre-freeze check is not a substitute for that control.

## Frozen arms

All sums use one correctly rounded IEEE-754 binary32 addition per element.

| arm | terminal composition | interpretation if it fails the reference gates |
|---|---|---|
| `REFERENCE_REFERENCE` | reference `ffn_inp-0` + reference `ffn_out-0` | apparatus VOID; must byte-reconstruct reference `l_out-0` |
| `C_C` | C `ffn_inp-0` + C `ffn_out-0` | replay control; must byte-reconstruct C `l_out-0` and the prior failure |
| `C_ATTENTION_REFERENCE_FFN` | C `ffn_inp-0` + reference `ffn_out-0` | captured attention-stream addend is sufficient |
| `REFERENCE_ATTENTION_C_FFN` | reference `ffn_inp-0` + C `ffn_out-0` | captured FFN addend is sufficient |

The two hybrid labels name the origin of each addend, not a naturally
recomputed mixed forward pass.

## Implementation contract

Add a diagnostic-only command that:

1. hashes and validates the accepted GGUF and all six immutable payloads;
2. performs all four F32 terminal additions in C with the existing
   FP-contraction-off contract;
3. refuses unless the two homogeneous sums byte-reconstruct their frozen
   `l_out-0` targets;
4. copies each sum into a fresh `strat01_r2a_arm.embd` buffer;
5. calls the unchanged `strat01_r2c_build_attention_range` for all eight
   tokens and the unchanged prefill `strat01_r2a_run_schedule`;
6. emits the same eleven layer-1 checkpoints used by the qualified
   layer-1-start diagnostic for every arm;
7. records zero donor and zero reference graph executions and never
   self-adjudicates.

Only the seven frozen layer-1 attention tensors registered by Rung 2C are
admitted. Their parsed names, types, shapes, offsets, file offsets, and spans
must be recorded and fail closed. No helper may silently fork normalization,
quantization, RoPE, K-B, cache, attention, or V-B semantics. Compile with
Clang C11, `-O3 -mavx2 -mfma`, no fast-math, and FP contraction disabled.

## External gates and controls

Every emitted checkpoint is compared with the immutable reference target
using the unchanged general Rung-2C limits: NRMSE `<= 0.002` and normalized
maximum `<= 0.01`. Both limits must pass separately; no average may rescue a
checkpoint. Report per-token metrics descriptively.

Required positive controls:

- `REFERENCE_REFERENCE` byte-reconstructs reference `l_out-0` and passes all
  eleven layer-1 gates;
- `C_C` byte-reconstructs C `l_out-0`, byte-replays all eleven captured C
  checkpoints, and reproduces the frozen metrics within `1e-12`.

Required negative and provenance controls:

- one-byte mutations of every admitted component are refused before addition;
- layer-0 tensors and any wrong layer-1 descriptor are refused;
- negating all 1,536 components of hybrid token 7 fails at least one
  downstream target gate;
- swapping hybrid rows 0 and 7 fails at least one target gate;
- arm-label or addend-origin swaps are detected by provenance validation;
- source inventories bind the shared layer-1 runner/header and this command.

A model-free self-test must cover F32 composition, homogeneous replay,
fresh-arm reset, four-arm routing, component mutations, descriptor refusal,
checkpoint enumeration, and report paths.

## Decision rule

Apply in order:

1. **`VOID_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT`** for any identity, replay,
   control, source, or apparatus failure.
2. **`ATTENTION_AND_FFN_RESIDUALS_INDEPENDENTLY_SUFFICIENT`** if both hybrid
   arms fail at least one of the eleven reference gates.
3. **`ATTENTION_STREAM_RESIDUAL_SUFFICIENT`** if only
   `C_ATTENTION_REFERENCE_FFN` fails.
4. **`FFN_RESIDUAL_SUFFICIENT`** if only `REFERENCE_ATTENTION_C_FFN` fails.
5. **`TERMINAL_RESIDUAL_JOINT_ONLY`** if both hybrid arms pass all eleven
   gates while the valid `C_C` replay fails.

Exactly one non-VOID invocation is allowed. Gates, ordering, and labels must
not change after values are observed.

## Non-claims and stop rule

This cell cannot repair or promote Rung 2C. It does not adjudicate the internal
source of either captured component, a naturally recomputed hybrid FFN,
output projection, MoE, later layers, tokenizer, logits, generation, quality,
RAM, or rate. It cannot update `SPEED_LEDGER.md`. Document and index the result
before implementing any production change or opening a finer component split.
