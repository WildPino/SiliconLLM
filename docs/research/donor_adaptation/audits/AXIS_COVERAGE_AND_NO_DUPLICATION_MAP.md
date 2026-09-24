# SiliconLLM research axes — coverage and no-duplication map

**Snapshot:** 2026-09-16. **Purpose:** answer “has this already been tried?” before opening a
brief, writing a runner, or spending CPU/GPU time. This is an index of project evidence, not a new
scientific result. The chronological source of truth for donor adaptation remains `../INDEX.md`; rate
arithmetic remains `../SPEED_LEDGER.md`; exact experiment status remains in the cited probe/result.
For donor selection, use `TARGET_DONOR_LOCAL_INVENTORY.md` for repository evidence and
`TARGET_DONOR_DECISION.md` plus its 2026-09-16 methodological erratum and
`TARGET_DONOR_FOLLOWUP_2026-09-16.md` for the frozen screening disposition.
The later [STRAT-01 screen](STRAT_01_SPARSE_HYBRID_METADATA_SCREEN.md) adds
Granite-4.0-H-Tiny-Base and LFM2.5-8B-A1B-Base at exact revisions with
per-organ active-weight arithmetic and engine-semantic gaps; it is metadata
only, not a quality or speed result.
The [corpus document-split audit](CORPUS_DOCUMENT_SPLIT_LIMITATION_2026-09-16.md)
shows that the legacy donor `calib`/`heldout` split is by shuffled 8 KiB chunk,
not guaranteed disjoint by source document; its sequence-level results must not
be promoted to a future document-bootstrap quality gate.
The still later [STRAT-03 result](../probes/STRAT_03_EXECUTABLE_SHARED_RESIDUAL_RESULT.md)
closes the frozen five-layer `k=3/256` local diagnostic with a byte-count-
matched SwiGLU shared path and `x`-only router: all three gates fail despite
valid controls. Do not repeat that geometry or infer a general impossibility
for other routers, jointly trained models or end-to-end quality.
The [STRAT-01 Rung-2B result](../probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2B_RESULT_20260921.md)
closes exact pretrained block-0 dense FFN parity in `engine.c` as FAIL: FFN
normalization passes, while both Q4_K gate/up projections are the first
failures in both schedules. This is not a repeat of STRAT-03 and does not test
carving, a learned router, shared-residual utility, or quality. Do not rerun
Rung 2B or advance to 2C; at that stage the next engine-fidelity continuation was the
separately scoped, no-full-donor [frozen-input Q4_K diagnostic](../probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2B_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260921.md).
That cell now closes positively for the exact reference input: up/gate NRMSE
is approximately `1.5e-7`, while byte-exact C-input replay reproduces the
Rung-2B FAIL and changes 46/48 Q8 blocks. Do not repeat it or rewrite the Q4_K
matrix path. The changed coordinate is upstream RMSNorm accumulation semantics.
The [frozen accumulator protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2B_RMSNORM_ACCUMULATOR_DIAGNOSTIC_PROTOCOL_20260921.md)
now closes as `DOUBLE_RMSNORM_INSUFFICIENT_FOR_PROJECTION_GATES`: double sum is
exact on the reference payload but leaves C up/gate NRMSE at
`0.00306126`/`0.00269141`. Do not repeat it or patch only FFN RMSNorm. The next
distinct coordinate is the same pinned accumulation semantic at the upstream
attention-input and compressed-KV RMSNorm sites, reusing frozen references.
That coordinate is frozen in the [upstream propagation protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_UPSTREAM_RMSNORM_PROPAGATION_DIAGNOSTIC_PROTOCOL_20260921.md).
It is not another FFN-only trial: it changes the two RMSNorm sites before the
accepted attention/Q8 path and adjudicates their propagated effect at both
Rung-2A boundaries and the unchanged FFN projection gates.
It now closes validly but insufficiently: terminal input improves by about
16.6%, while both FFN projections remain outside gate. Do not repeat any
RMSNorm cell. The next distinct coordinate is full-matrix Q5_0×Q8_0 K-B
runtime semantics; Rung 1 covered stored-row decode/dequantized matvec only.
The [frozen K-B protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_KB_Q5_0_Q8_0_DIAGNOSTIC_PROTOCOL_20260921.md)
therefore changes activation format and full-matrix operator semantics while
holding artifact, tensor, inputs, output target, layout, and model layer fixed.
Its [result](../probes/STRAT_01_GIGACHAT31_ENGINE_KB_Q5_0_Q8_0_DIAGNOSTIC_RESULT_20260921.md)
closes that coordinate as `Q5_0_Q8_0_SUFFICIENT_FOR_KB_BOUNDARY`: exact and
upstream-double inputs both reach NRMSE `4.80e-8`, and all three frozen inputs
have identical Q8_0 bytes. Do not repeat Q5_0 codec, K-B layout, or this
operator cell. Their composition is now measured in the
[combined result](../probes/STRAT_01_GIGACHAT31_ENGINE_COMBINED_RMS_Q5Q8_PROPAGATION_RESULT_20260922.md)
as `COMBINED_RMS_Q5Q8_CLOSES_PROJECTION_GATES`: every frozen attention, cache,
continuity, and dense-FFN gate passes, with up/gate NRMSE
`4.0201e-4`/`3.4697e-4`. Do not repeat any isolated or combined coordinate.
Their [production integration result](../probes/STRAT_01_GIGACHAT31_ENGINE_BLOCK0_PRODUCTION_INTEGRATION_RESULT_20260922.md)
now closes `PASS_ENGINE_BLOCK0_PRODUCTION_INTEGRATION`: the standard Rung-2B
path reproduces the three exact combined hashes, all twelve checkpoint gates
through `l_out-0` pass, all twelve continuity checks are exact, and all four
causal controls reject. One donor invocation ran and the immutable reference
was not rerun. Do not repeat production integration or any block-0 predecessor.
The next distinct engine-fidelity coordinate is the separately frozen [Rung
2C protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2C_PROTOCOL_20260923.md):
one block-1 routed-expert plus shared-expert MoE layer. It changes depth,
rank-3 selected-expert operators, discrete ordered top-4 routing, and
routed/shared aggregation; it does not repeat STRAT-03 or any block-0 cell.
Its initial [apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2C_APPARATUS_RESULT_20260923.md)
passed with zero producer invocations and zero graph schedules. The first
scientific attempt is [VOID 1](../probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2C_VOID1_20260923.md):
both reference schedules completed but cached top-k serialization refused on
an `i32`/`I32` apparatus mismatch; C did not run. Do not repeat that exact
source coordinate. Canonical apparatus `repair2` now passes the narrow
type/accounting repair with zero producer executions. The authorized rerun is
now closed as [`FAIL_ENGINE_RUNG2C`](../probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2C_RESULT_20260923.md):
both producers completed once, 34/64 checkpoints pass, and the first failure is
`kqv_out-1` while routing passes. Do not repeat Rung 2C. The next new evidence
is the frozen [zero-producer Q×KV cross-input attention-boundary diagnostic](../probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2C_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md).
Its [apparatus qualification](../probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2C_CROSS_INPUT_DIAGNOSTIC_APPARATUS_RESULT_20260923.md)
passed with zero donor executions. The single invocation is now closed by the
[`QUERY_AND_KV_RESIDUALS_INDEPENDENTLY_SUFFICIENT` result](../probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2C_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md):
reference/reference passes, while either C query or C KV alone fails. Do not
repeat or change attention/V-B; localize propagation from `l_out-0` next.
That axis is now closed as [`BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT`](../probes/STRAT_01_GIGACHAT31_ENGINE_RUNG2C_LAYER1_START_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md):
exact reference `l_out-0` makes all eleven layer-1 checkpoints pass. Do not
modify layer 1. The distinct next axis is now frozen in the
[block-0 terminal-component cross-input protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md):
captured C/reference `ffn_inp-0` and `ffn_out-0` are crossed in four F32 sums,
with byte-exact homogeneous replay and the unchanged layer-1 builder. This is
now apparatus-qualified with zero producer graphs; exactly one scientific
invocation is now closed by the
[`FFN_RESIDUAL_SUFFICIENT` result](../probes/STRAT_01_GIGACHAT31_ENGINE_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md):
the C attention-stream addend paired with reference FFN passes downstream,
whereas the C FFN addend paired with the reference attention stream fails and
nearly reproduces C/C. Do not repeat this split or modify layer 1. The next
distinct boundary is captured `ffn_swiglu-0` input versus the unchanged Q6_K
down operator, adjudicated after propagation through layer 1. That new axis is
now frozen in the
[FFN down-projection cross-input protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_FFN_DOWN_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md).
Its [apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_FFN_DOWN_CROSS_INPUT_DIAGNOSTIC_APPARATUS_RESULT_20260923.md)
is qualified with zero graph executions. One scientific invocation remains;
it now closes as
[`SWIGLU_INPUT_RESIDUAL_SUFFICIENT`](../probes/STRAT_01_GIGACHAT31_ENGINE_FFN_DOWN_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md):
the current Q6 operator passes on exact captured SwiGLU input at direct NRMSE
`8.35e-8` and across all downstream gates. Do not repeat or rewrite Q6. The
next distinct axis is captured gate versus up versus current SiLU/multiply
semantics, with propagation through the closed Q6 and layer-1 paths. That axis
is now frozen in the
[FFN SwiGLU cross-input protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_FFN_SWIGLU_CROSS_INPUT_DIAGNOSTIC_PROTOCOL_20260923.md)
and remains unmeasured. Its
[apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_FFN_SWIGLU_CROSS_INPUT_DIAGNOSTIC_APPARATUS_RESULT_20260923.md)
was qualified with zero model or producer executions, but the first invocation
is [VOID 1](../probes/STRAT_01_GIGACHAT31_ENGINE_FFN_SWIGLU_CROSS_INPUT_DIAGNOSTIC_VOID1_20260923.md):
an extra replay assertion compared against the wrong prior arm. Do not infer a
gate/up/operator verdict or inspect arm metrics. Only the registered one-line
repair, regression, fresh apparatus, and one repaired invocation are allowed.
The [repair-1 apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_FFN_SWIGLU_CROSS_INPUT_DIAGNOSTIC_APPARATUS_REPAIR1_RESULT_20260923.md)
passes with zero model access. Its repaired invocation is now closed as
[`GATE_AND_UP_RESIDUALS_INDEPENDENTLY_SUFFICIENT`](../probes/STRAT_01_GIGACHAT31_ENGINE_FFN_SWIGLU_CROSS_INPUT_DIAGNOSTIC_RESULT_20260923.md):
each C operand alone passes the direct SwiGLU/down gates but first fails after
propagation at `kqv_out-1`; reference/reference current SwiGLU passes. Do not
rewrite SwiGLU/Q6/layer 1. Compare the production `ffn_norm-0` identity with
the earlier Rung-2B cross-input lineage before opening a projection successor.
The identity audit closes that apparent opening without a run: reference
`ffn_norm-0` and the gate/up Q4 helper are unchanged, and production already
binds its new C normalized input to both measured outputs. The next genuinely
changed coordinate is the accepted generic Q4_K×Q8_K reduction versus the
pinned active x86 AVX2/FMA reduction order. Its
[frozen protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY_PROTOCOL_20260923.md)
requires exact block-0 Q/KV hashes. Its
[apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY_APPARATUS_RESULT_20260923.md)
is qualified with true pinned-x86 oracle identity, bit-exact one/multi-block
tests, live generic controls, 123 Python tests, all C self-tests, and zero model
reads. The committed
[result](../probes/STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY_RESULT_20260923.md)
is `FAIL_ENGINE_Q4K_Q8K_AVX2_REDUCTION_PARITY`: active x86 fails both exact
projection hashes and is slightly farther from reference than the historical
generic arm; that arm replays exactly. The reference producer compiled the
baseline `GGML_CPU_GENERIC` backend. Do not repeat AVX2 or model-free operator
work. A successor must change the coordinate to exact reference-build compiler
semantics and preregister it separately. The
[frozen successor](../probes/STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_PROTOCOL_20260923.md)
does exactly that and requires model-free bit identity before one full
projection. Do not substitute another AVX kernel, relax exactness, or rerun the
closed cell. Its
[apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_APPARATUS_RESULT_20260923.md)
is now qualified: the candidate is baseline-generic bit-exact at four lengths,
both historical controls fire, 127 Python tests and all C self-tests pass, and
no model was read. The only non-duplicate next action on this coordinate is the
single frozen full Q/KV adjudication; it cannot establish quality or rate.
That [adjudication](../probes/STRAT_01_GIGACHAT31_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY_RESULT_20260923.md)
is now `PASS_ENGINE_Q4K_Q8K_REFERENCE_GENERIC_COMPILE_PARITY`: candidate Q/KV
are byte-identical to reference and both diagnostic arms replay their distinct
historical hashes. Operator and full-projection parity are closed. Do not rerun
this cell, AVX2 parity, model-free Q4 tests, or gate/up cross-input. The next
non-duplicate coordinate is downstream propagation through immutable block-0
and Rung-2C checkpoints; it requires a new frozen protocol.
That [protocol is now frozen](../probes/STRAT_01_GIGACHAT31_ENGINE_REFERENCE_GENERIC_PROPAGATION_PROTOCOL_20260923.md).
It admits only the accepted helper as the changed coordinate, reuses all
reference tensors/caches, and allows one C producer for both schedules. Do not
substitute a new split, reference rerun, relaxed gate, or another Q4 kernel.
The [apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_REFERENCE_GENERIC_PROPAGATION_APPARATUS_RESULT_20260923.md)
is qualified with validated immutable references, 131 Python tests, all C
self-tests, and zero graph executions. Do not repeat apparatus or start a
parallel diagnostic; the single committed C-producer adjudication is next.
The [result](../probes/STRAT_01_GIGACHAT31_ENGINE_REFERENCE_GENERIC_PROPAGATION_RESULT_20260923.md)
is a valid FAIL and closes this cell. Q4 propagation raises checkpoint passes
from 34/64 to 52/64 and closes block-0 terminal, layer-1 attention, router,
up/gate, and SwiGLU. Routed/shared Q6 down first fail. Do not repeat global Q4
propagation or infer a generic Q6 defect; cross immutable layer-1
reference/current SwiGLU through those exact Q6 matrices next, with zero
producer graphs.
The [layer-1 Q6 cross-input protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_Q6_CROSS_INPUT_PROTOCOL_20260923.md)
is now frozen. It is not the closed block-0 Q6 cell: it binds width 1280,
token-dependent selected expert slices, and the shared expert's distinct Q6
tensor. It uses only `prefill8` because schedule continuity and metrics are
already identical, and it permits zero graph executions.
The [apparatus qualification](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_Q6_CROSS_INPUT_APPARATUS_RESULT_20260923.md)
passes with all immutable inputs validated, 135 Python tests, 17 C self-tests,
and zero graph executions. This closes apparatus construction, not the Q6
attribution. The sole open coordinate is the one committed direct diagnostic;
no parallel split, graph rerun, or Q6 edit is admissible first.
The [direct result](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_Q6_CROSS_INPUT_RESULT_20260923.md)
is `LAYER1_SWIGLU_RESIDUAL_SUFFICIENT`: reference routed/shared inputs pass at
NRMSE `7.11e-8`/`6.13e-8`, current inputs replay the canonical failures, and
all controls pass with zero graphs. Both branches localize the discrepancy to
tokens 4 and 5. Close the actual layer-1 Q6 operators; any successor must move
strictly upstream and must not duplicate Q4, SwiGLU, or Q6 attribution.
The [exact F16 vector-reduction propagation protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION_PROTOCOL_20260923.md)
is the next distinct coordinate. The 21 September F16 diagnostic closed
conversion and showed pinned vector reduction exact, but scalar reduction was
accepted only under the old local gate. The layer-1 Q6 result now proves that
the admitted residual is downstream-unsafe. Freeze exact model-free identity
before one C propagation; do not rerun a reference graph or reopen softmax,
Q4, SwiGLU, or Q6.
The [apparatus result](../probes/STRAT_01_GIGACHAT31_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION_APPARATUS_RESULT_20260923.md)
now qualifies that exact cell with zero graphs: five frozen payload hashes are
byte-exact, both mutations reject, and 141 Python plus 17 C self-tests pass.
Its pre-donor addendum records that the historical helper used the generic
F64 accumulator path rather than AVX/FMA. Apparatus construction and Stage A
are closed. The subsequent
[`FAIL_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION` result](../probes/STRAT_01_GIGACHAT31_ENGINE_F16_VECTOR_REDUCTION_PROPAGATION_RESULT_20260923.md)
is valid: both schedules and all controls pass, every tracked NRMSE improves,
but routed/shared Q6 down remain above gate at `0.003761`/`0.002372`, now
concentrated at token 6. Do not rerun the producer or reopen F16/Q4/SwiGLU/Q6.
The distinct next axis is the frozen
[post-F16 layer-1-start cross-input](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_LAYER1_START_CROSS_INPUT_PROTOCOL_20260923.md):
new current versus exact reference `l_out-0`, complete current layer 1,
byte-exact current replay, and zero graph executions. Its valid
[result](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_LAYER1_START_CROSS_INPUT_RESULT_20260923.md)
is `POST_F16_BLOCK0_TERMINAL_RESIDUAL_SUFFICIENT`: all 32 reference-start
checkpoints pass, terminal NRMSE is `1.575e-7`, and all current checkpoints
replay exactly. Close layer 1 and all prior graph-level Q4/F16 hypotheses. The
next changed coordinate was the frozen
[post-F16 block-0 terminal-component split](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_PROTOCOL_20260923.md).
Its valid [result](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_TERMINAL_COMPONENT_CROSS_INPUT_RESULT_20260923.md)
is `POST_F16_BLOCK0_FFN_OUT_RESIDUAL_SUFFICIENT`: current `ffn_inp-0` is
byte-identical to reference, and only current `ffn_out-0` reproduces the full
failure. That output is byte-identical to the earlier reference-gate/reference-
up through current expression+Q6 arm, so gate/up is already an unnecessary
coordinate and must not be repeated. The distinct axis was frozen in the
[post-F16 FFN operator-chain split](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_FFN_OPERATOR_CROSS_INPUT_PROTOCOL_20260923.md).
Its valid [result](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_FFN_OPERATOR_CROSS_INPUT_RESULT_20260923.md)
is `POST_F16_BLOCK0_SWIGLU_EXPRESSION_RESIDUAL_SUFFICIENT`: captured reference
SwiGLU through unchanged current Q6 passes every complete layer-1 gate, while
the production C expression on the same reference gate/up operands first
fails at routed `ffn_moe_down-1` and exactly replays the predecessor. One
diagnostic and zero graphs ran; all controls pass. Close Q6, gate/up, block 0,
and layer 1. The only next admissible axis is a frozen split of production
SwiGLU numerical semantics on the same immutable operands and downstream
amplifier. The qualified apparatus and its earlier fixture-bounds VOID remain
part of the record.
That successor is now frozen in the
[SwiGLU SSE2-semantics protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_SWIGLU_SSE2_SEMANTICS_PROTOCOL_20260924.md).
The exact new coordinate is the reference build's no-FMA four-lane SSE2
polynomial `ggml_v_expf` versus current scalar `expf`; 8960 is divisible by
four, so there is no scalar tail. Only model-free apparatus qualification is
next. Its [apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_SWIGLU_SSE2_SEMANTICS_APPARATUS_RESULT_20260924.md)
is now qualified after 165 Python tests, 21 C self-tests, eleven source
controls, and exact reference-source/build checks at zero diagnostic/model/
graph execution. Its sole valid
[result](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_BLOCK0_SWIGLU_SSE2_SEMANTICS_RESULT_20260924.md)
is `POST_F16_BLOCK0_SWIGLU_SSE2_EXACT_REPAIR`: pinned no-FMA four-lane
semantics are byte-exact to reference SwiGLU and pass all 32 complete layer-1
checkpoints (`l_out-1` NRMSE `9.31e-6`), while scalar `expf` first fails at
routed `ffn_moe_down-1` (`0.00376095`). One diagnostic and zero graphs ran;
all controls pass. Close this semantics split, AVX2/FMA sweeps, and every
predecessor. The only next distinct axis is separately frozen production
integration of the proven primitive, model-free qualification first, then
one accepted-GGUF execution through block 0 and complete layer 1.
That distinct axis is now frozen in the
[post-F16 SwiGLU production-integration protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION_PROTOCOL_20260924.md).
Only shared primitive ownership and the normal block-0 call may change;
layer 1 and all Q4/F16/Q6 coordinates remain fixed. The standard Rung-2C
command must then cover both schedules, all 32 checkpoints, and six caches.
Only model-free apparatus qualification is next; accepted GGUF access is
forbidden until exact sources are committed.
That [apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION_APPARATUS_RESULT_20260924.md)
is now qualified with one shared primitive, unchanged layer 1, 172 Python
tests, 21 C self-tests, eleven passing source controls, exact predecessor and
reference bindings, and zero GGUF/graph execution. Its sole
[production result](../probes/STRAT_01_GIGACHAT31_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION_RESULT_20260924.md)
is now valid `PASS_ENGINE_POST_F16_SWIGLU_PRODUCTION_INTEGRATION`: 64/64
checkpoints, 64/64 continuity comparisons, six exact caches, exact routes and
helper counts, and ten live causal controls pass across both schedules. One
production invocation and zero reference graphs ran. This closes all
engine-fidelity coordinates through complete layer 1. No prior axis reopens;
the next eligible coordinate must be a separately frozen depth extension.
That coordinate is now defined by the
[layer-2 static audit](STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_AUDIT_20260924.md)
and [frozen protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_PROTOCOL_20260924.md).
It adds only `blk.2` execution/evidence and leaves accepted layer 0/1
arithmetic unchanged. Model-free apparatus qualification is next; no donor
or reference graph is authorized before a committed passing apparatus.
The canonical [apparatus `repair1`](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_APPARATUS_RESULT_20260924.md)
is now qualified: the complete scientific adjudicator, 184 Python tests, 22 C
self-tests, the separate reference self-test, twelve source controls, and all
bindings pass with zero model access or graphs. It supersedes the earlier
pre-adjudicator apparatus. Commit the exact sources, then run each new
producer once; no closed axis reopens.

The sole layer-2 scientific execution is complete, so producer execution is
now permanently closed. Its raw `FAIL` status is not canonical because one
aggregate causal mutation did not reject; the frozen rule classifies that
record as VOID. The
[offline-recovery protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_OFFLINE_RECOVERY_PROTOCOL_20260924.md)
permits only a same-threshold branch-local repair over immutable outputs.
No model, reference, C producer, predecessor, or closed operator axis may be
rerun. `kqv_out-2` is only a provisional first-failure boundary until the
offline recovery is valid.

The canonical [offline-recovered result](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_DEPTH_EXTENSION_RESULT_20260924.md)
is `FAIL_ENGINE_LAYER2_DEPTH_EXTENSION_RECOVERED`. It closes layer-2 depth
execution with 52/64 checkpoints, 64/64 continuity comparisons, 9/9 caches,
8/8 predecessor witnesses, and all ten repaired controls valid. The first
failure is `kqv_out-2` in both schedules while every Q/K/V input passes.
Only an attention-output diagnostic over immutable payloads is non-duplicate;
new model/reference graphs require proof that the needed intermediate is
absent.

The next cell is now frozen in the
[layer-2 attention cross-input protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_ATTENTION_CROSS_INPUT_PROTOCOL_20260924.md).
It reuses the closed layer-1 diagnostic method but changes the actual depth,
inputs, target, and V-B weights. Prefill and cached payload hashes are exact,
so only one boundary replay is admissible. Qualify model-free first; the sole
scientific invocation may read the V-B matrix but executes zero model graphs.
The [apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_ATTENTION_CROSS_INPUT_APPARATUS_RESULT_20260924.md)
is qualified with every new/inherited test passing and zero model access or
graphs. Commit its exact sources, then execute the one permitted boundary
replay; do not add a cached duplicate.
The [result](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_ATTENTION_CROSS_INPUT_RESULT_20260924.md)
is `LAYER2_KV_RESIDUAL_SUFFICIENT`: exact-reference replay is byte-exact,
query-only passes, and KV-only fails. This closes the operator, query, second
schedule, and whole-KV repeat axes. The only non-duplicate successor splits
the immutable K row into its shared latent/value prefix 512 and positional
tail 64, holding reference query fixed.
The [KV-partition cross-input protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_PARTITION_CROSS_INPUT_PROTOCOL_20260924.md)
now freezes that successor. It is new only on the partition coordinate; the
query, attention/V-B operator, schedules, complete-KV replay, and all earlier
fidelity axes remain closed. Its
[apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_PARTITION_CROSS_INPUT_APPARATUS_RESULT_20260924.md)
is model-free qualified with all tests passing, zero artifact access, and zero
graphs. Commit the exact source set, then execute once.
Its [result](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_PARTITION_CROSS_INPUT_RESULT_20260924.md)
is `LAYER2_LATENT_VALUE_RESIDUAL_SUFFICIENT`: prefix-only fails and
positional-tail-only passes. This permanently closes the tail and all broader
query/KV/operator repeats. The sole new successor may split the already
captured pre-RMSNorm prefix from the KV RMSNorm operator that produces
`kv_cmpr-2`; it must not execute a new graph.
The [KV RMSNorm protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_RMSNORM_CROSS_INPUT_PROTOCOL_20260924.md)
now freezes exactly that successor. Its only changed coordinate is the
pre-RMSNorm projected prefix; all downstream and positional paths remain
closed. Its [apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_RMSNORM_CROSS_INPUT_APPARATUS_RESULT_20260924.md)
is model-free qualified; commit exact sources before the sole zero-graph
invocation.
The invocation completed all arms but is raw VOID due solely to a wrong
descriptive-prefix reshape. The frozen
[offline recovery](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_RMSNORM_OFFLINE_RECOVERY_PROTOCOL_20260924.md)
must reuse those bytes with zero model access and zero graphs. Producer rerun
is permanently closed.
The [canonical recovery result](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_RMSNORM_CROSS_INPUT_RESULT_20260924.md)
is `LAYER2_PROJECTED_KV_PREFIX_RESIDUAL_SUFFICIENT_RECOVERED`: both RMSNorm
replays are byte-exact, so the residual exists upstream in `kv_cmpr_pe-2`.
Close RMSNorm permanently. Only an immutable-input KV-A projection split is
new; no graph is justified.
That successor is frozen by the
[KV-A projection protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_A_PROJECTION_CROSS_INPUT_PROTOCOL_20260924.md).
Only the projection input changes; Q4 parity, RMSNorm, attention, and every
downstream axis remain closed. Its
[`LAYER2_ATTN_NORM_INPUT_RESIDUAL_SUFFICIENT` result](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_KV_A_PROJECTION_CROSS_INPUT_RESULT_20260924.md)
passes byte-exactly on reference input and replays every C stage exactly. Close
KV-A permanently. The only non-duplicate coordinate is immutable `l_out-1`
through production layer-2 attention RMSNorm, frozen in the
[attention RMSNorm protocol](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_ATTN_RMSNORM_CROSS_INPUT_PROTOCOL_20260924.md).
It reuses all closed downstream operators and authorizes no graph. Its
[repair-1 apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_ATTN_RMSNORM_CROSS_INPUT_APPARATUS_RESULT_20260924.md)
was model-free qualified after preserving the initial compile VOID. Its
[canonical result](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER2_ATTN_RMSNORM_CROSS_INPUT_RESULT_20260924.md)
is `LAYER1_TERMINAL_RESIDUAL_SUFFICIENT`: exact reference input passes every
downstream stage byte-exactly and C input replays the failure. Close attention
RMSNorm and every downstream axis. The only non-duplicate coordinate is the
frozen [layer-1 terminal-component split](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_TERMINAL_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md)
at `ffn_inp-1 + ffn_out-1`, using immutable payloads and zero graphs. Its
[model-free apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_TERMINAL_COMPONENT_CROSS_INPUT_APPARATUS_RESULT_20260924.md)
was qualified without artifact/payload access. Its
[canonical result](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_TERMINAL_COMPONENT_CROSS_INPUT_RESULT_20260924.md)
is `LAYER1_FFN_OUTPUT_RESIDUAL_SUFFICIENT`: C FFN output alone reproduces the
failing downstream hash and C FFN input alone passes. Close the terminal sum
and all layer-2 axes. The only non-duplicate coordinate is the frozen
[FFN-output component split](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_FFN_OUTPUT_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md)
between routed `ffn_moe_out-1` and shared `ffn_shexp-1`, with exact reference
residual and zero graphs. Its
[model-free apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_FFN_OUTPUT_COMPONENT_CROSS_INPUT_APPARATUS_RESULT_20260924.md)
qualified without artifact/payload access or diagnostic execution. Commit those
exact sources, then run the sole zero-graph diagnostic. Its canonical
[result](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_FFN_OUTPUT_COMPONENT_CROSS_INPUT_RESULT_20260924.md)
is `LAYER1_ROUTED_MOE_OUTPUT_RESIDUAL_SUFFICIENT`: C routed output replays the
failure and C shared output passes. Close the shared, terminal, and layer-2
axes. The only non-duplicate coordinate is the frozen
[routed-MoE component split](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_MOE_COMPONENT_CROSS_INPUT_PROTOCOL_20260924.md)
between captured expert-down outputs and normalized router weights, with
byte-identical top-4 IDs and zero graphs. Its
[model-free apparatus](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_MOE_COMPONENT_CROSS_INPUT_APPARATUS_RESULT_20260924.md)
qualified without artifact/payload access or diagnostic execution. Commit
exact sources, then run the sole zero-graph diagnostic. Its canonical
[result](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_MOE_COMPONENT_CROSS_INPUT_RESULT_20260924.md)
is `LAYER1_ROUTED_EXPERT_DOWN_OUTPUT_RESIDUAL_SUFFICIENT`; the normalized
router-weight arm passes. Close routing, weighting/sum, shared, terminal, and
layer 2. The only non-duplicate coordinate is the frozen
[routed-Q6 residual propagation](../probes/STRAT_01_GIGACHAT31_ENGINE_LAYER1_ROUTED_Q6_RESIDUAL_PROPAGATION_PROTOCOL_20260924.md):
reuse the immutable old Q6(reference-SwiGLU) output under the new downstream
gate; do not rerun Q6 or a graph. Qualify model-free first.

## 1. How to read this map

Evidence classes are intentionally different:

- **CLOSED:** the registered question has a deciding measurement. Reopen only by changing the
  estimand or a named scope coordinate (donor, model nature, scale, format, engine path, domain).
- **MEASURED-SCOPED:** a useful number exists, but only inside the stated scope. It is not a scale
  law or a result for another lineage.
- **OPEN / CONDITIONAL:** no deciding measurement exists, or the next action is armed only after a
  prior gate passes.
- **PARKED:** valid unfinished work outside the current donor-adaptation priority.
- **PRIOR-ART / DESIGN ONLY:** literature, arithmetic or a proposal; never cite it as a project
  measurement.
- **VOID:** apparatus or controls failed before the registered estimand was measured. A repaired run
  may be owed, but the void numbers are not evidence.

When status text disagrees across documents, use this precedence:

1. newest canonical probe/result and its audit;
2. `.planning/codebase/RESEARCH_CATALOG.md` and this map;
3. dedicated run records such as `docs/PHASE64_RUNG1_RUN_RECORD.md`;
4. `README.md` / `HANDOFF.md` summaries;
5. briefs, proposals, old plans and archived documents.

The lower item may describe a plan that a later item completed, voided, superseded or parked.

## 2. The four research lineages — do not merge their claims

| lineage | object actually studied | strongest surviving result | present status | canonical navigation |
|---|---|---|---|---|
| Frozen / reservoir lineage, phases 1–54 | random or fixed silicon-native substrate, learned readouts, local plasticity, O(1) superposition memory, on-policy/DAgger and n-gram assists | local/medium-range similarity is cheap; long-range relations and selection are not recoverable from the frozen substrate. Static per-step and O(1)-superposition routes closed; BPE-1024 produced recognizable but structurally weak English | **CLOSED as a route to the target LLM**; keep its engine and instrumentation lessons | `HANDOFF.md` §§Phase 47–54; `archive/docs/phases/` |
| Trained CPU-native lineage, phases 55–64 | selective SSM + local attention, dReLU/ternary MLP, granular MoE, thin retrieval tier, trained from scratch | sandbox engine and properties validated; Phase-64 MVE completed; S0 30M reached 0.9277 code-val BPB, but its registered offline property gates were never read | **PARKED** while donor adaptation consumes quota; no quality-at-scale claim | `README.md`; `docs/PHASE64_*`; `docs/PHASE64_RUNG1_RUN_RECORD.md` |
| Pretrained-donor adaptation, P/R/T/D/F/E/H series | Qwen2.5/Qwen2.5-Coder Transformer weights converted, factorized, carved and/or healed for `donor_engine.c` | one-byte 7B score is nearly fp32; post-hoc carve/rank remain the structural wall; H2I is terminal `SCORE-ONLY`, H4 recovers score/rank but stays generatively at floor, and H5 closes the frozen composition shortcut negatively | **ACTIVE**; chronology through E66/E63 Part B and terminal H1/H2I/H4/H4D/H5 | `../INDEX.md`; `../SPEED_LEDGER.md`; `.planning/codebase/RESEARCH_CATALOG.md` |
| External research / architecture exploration | SSM+retrieval, CPU bandwidth, linear attention, donor methods, CPU-native alternatives | candidate mechanisms, counterevidence and source trails | **PRIOR-ART / DESIGN ONLY** unless a row above cites a local probe | `docs/research/*Research*`; `docs/research/ATTENTION_LINEARISATION_PRIOR_ART.md`; `../prior_art/`; `docs/research/LLM_Reimagined_CPU_Native_Exploration_20260626/` |

The most dangerous false equivalence is “ternary works” versus “ternary fails.” It works when a
small native model is **trained into** its ternary MLP; naive/post-hoc ternarization of a pretrained
Transformer destroys ranking. Those are different objects and both measurements stand.

## 3. Coverage matrix by attackable axis

| axis | variants actually attacked | decisive evidence and scope | status now | do not repeat / legitimate reopening |
|---|---|---|---|---|
| **Model nature / recurrence** | frozen reservoir; trained selective SSM; dense Transformer donor; SSM+retrieval and hybrid alternatives in literature | phases 47–54 close frozen-substrate long-range semantics; phases 55–64 validate a trained SSM only at sandbox/S0 scale; donor E-series executes real Qwen Transformer weights | **CLOSED** frozen route; **MEASURED-SCOPED** native SSM and donor Transformer; hybrids mostly **PRIOR-ART** | Do not propose another readout/local-memory trick on the same frozen state. A new recurrent/hybrid architecture needs a new trained object and cannot inherit donor or sandbox quality |
| **Family and domain** | TinyStories/prose; code; logs; Qwen2.5 general and Coder donors | native properties transfer to code; logs are a measured LZ-favouring boundary. Donor quality uses pinned slices and Qwen scales; E15/E66 establish the real Coder-7B endpoints | **MEASURED-SCOPED** | Change family/domain only with a reason and new controls. Never compare BPB across tokenizers/slices as if paired; Phase-64 §4.2 explicitly forbids two tempting comparisons |
| **Parameter scale** | native ~8–22M, MVE/S0 ~30M; donors 0.5/1.5/3/7B; synthetic/constructed ~10B speed cells | E60/E62/E66 measure one-byte fidelity through 7B but E62 is non-monotone. E36/E39/E40 measure 10B **shape economics**, not pretrained quality. E63 Part B measures the exact A10B mixed-format path at 49.37 tok/s, CI [49.18, 50.49], `G-E63d=DESK-MODEL-HELD` | **OPEN at the joint 10B pretrained quality+rate claim** | Do not fit a scale exponent from 0.5/1.5/3B or promote synthetic 10B weights. E63 is a synthetic mixed-format rate, not a full-int8 stream or a donor-rate transfer; target quality still requires target-scale pretrained/trained weights |
| **Width, depth, heads, KV and rank shape** | depth reduction/redistribution; D/F/L shape budgets; GQA/KV/context; low-rank q/o/head; R128 attention; residual-layer rank | E24/E35 price depth; E34 prices attention+head floor; E21/E25/E29/E39/E40 price low-rank shape and kernels; E65 measures r/D=1/32 quality cost at D=1536 | mostly **CLOSED per registered shape**, no width law | Do not extrapolate E65 to D=4096 or call R128 free. Reopen with an explicit target shape and a target-quality measurement, not parameter-count equivalence alone |
| **Vocabulary, tokenizer, embedding and head** | BPE-1024/2048 native; Qwen vocab; tied/untied and fp32/ternary head; vocabulary-size speed/token-worth | Phase 50 makes unit choice load-bearing; E17 shows the donor head is not the sole ranking mechanism; E18 organ ladder; E43 measures token units but its speed half is permanently void | **MEASURED-SCOPED** | Do not rerun E43 as the same speed question. A smaller-vocab claim must report bytes/text as well as tok/s and re-establish quality; changing tokenizer breaks direct BPB/rank comparability |
| **Context and explicit memory** | SSM state; O(1) superposition; local attention; IVF-PQ/two-stage recall; decode context slopes | O(1) superposition/addressing closed negative; Phase 56 validates learned recall representation and two-stage cost. 128K query **cost** is measured, while direct recall quality is at shorter footing plus proxy/argument. E46/E48 price Transformer context slope | cost **MEASURED**; fused target-length benefit **OPEN** | Do not quote “128K recall validated” without separating cost from quality. Reopen only with target-length generated-rollout recall or an integrated model benefit gate |
| **Weight precision / representation** | fp32; naive and activation-aware ternary; 4-bit packed ternary storage; 1 byte/weight int8/R8; mixed tagged-v2 | T1/T2/T2b/E12/E15/E16/E20 close post-hoc ternary rule search for this donor; E60/E62/E66 show the one-byte rung is faithful through 7B; E63 proves the exact mixed A10B container/path and rate | ternary PTQ **CLOSED**; one-byte fidelity **CLOSED through 7B**; mixed 10B quality **OPEN** | No more ternary-rule sweep absent a genuinely new estimand. E63 has one-byte carved FFN only (11.4% of charged weights); do not call it a whole-token one-byte or full-int8 result. Change donor, quantizer or trained format explicitly to reopen E66 |
| **Activation precision** | fp32 activations; int8 block/LUT activation quantization; trained fast-LUT path | E14/E32 price BPB cost; E59 gains ~1.25–1.33× but changes 42–74% of trained tokens | **CLOSED for blind enablement** | Never enable `--lutblk` from speed alone. Any successor must pair same-format quality/rank with speed and must identify activation quantization separately from weight format |
| **Folding, scales and normalization** | layer RMSNorm fold; final/head fold; row/group scales; magnitude matching at upcycle | E2 confirms layer fold; `--fold all` hurts when final gain is folded into ternary head. T3 isolates the useful fold from harmful rotation. Phase-64 ε-identity/magnitude matching records a residual D→E shock | **CLOSED per construction**; upcycle transition mechanism **MEASURED-SCOPED** | Reuse the adopted layer fold. Do not assume a ternary-specific fold helps int8 (E66 says its contribution is tiny/opposite in that construction) or hide an architecture switch behind “same checkpoint” |
| **Rotation, basis change and permutation** | dense orthogonal/Hadamard rotation; weight sparsifying basis; coactivation permutation; layout transpose | D2 says affordable orthogonal bases do not create useful donor sparsity; T3 rotations hurt ternarization; D0/D0c show finer coactivation granularity helps but random null helps more; E13's tile-major transpose is a kernel layout, not a model transformation | model-transform route **CLOSED / negative**; layout **ADOPTED** | Do not rerun generic Hadamard/orthogonal rotation. A new transform needs a different mechanism and must keep model-space transforms separate from bit-exact storage/layout transforms |
| **Low-rank factorization** | SVD/weight-only; activation-weighted q/o and head; residual layers; engine factored matvec; rank moved across organs | R2A/E21 find activation-weighted q/o geometry; E25 proves engine path/parity; E29 narrows which layers rank; E39/E40 buy speed; E65 shows the fast rank fraction consumes 51.66% of dense→chance gap at D=1536 | speed path **CLOSED/measured**; post-hoc quality at target width **OPEN but constrained** | Do not repeat rank solely because it speeds the shape. A new rank scheme must beat E65 on quality at a declared fraction and preserve ranking, or be trained rather than post-hoc |
| **Sparsity, pruning and carve granularity** | magnitude/block pruning; coactivation/null; neuron/expert granularity; static/mass-ranked/output-greedy carve; shared low-rank residual plus mass top-3; 1.17% FFN; coarse locality | D1/D0/D0c/E19/E27/E33/E37/E38 measure post-hoc sparsity; E38's **intermediate-mass** selector is above chance at `k=3` but is not a universal oracle. E67's genuinely output-aware greedy selector gives only **4.23%** aggregate local SSE reduction, `NO_LOCAL_SIGNAL`. E68's rank-64 linear shared residual changes the operator: aggregate local SSE ratio is **0.46674069016**, `PARTIAL_LOCAL_SIGNAL`, but layer 27 dominates 90.72% of mass SSE and layer 1 worsens (ratio 1.7711406384). E64 shows carve is dearer on int8 | mass-ranked and E67 output-greedy post-hoc routes **CLOSED at tested scope**; E68 shared residual **MEASURED-SCOPED `PARTIAL_LOCAL_SIGNAL`**; exact loss-aware selection remains **UNMEASURED** | Do not repeat E38/E67 or claim either bounds every selector. E68 is local FFN error only, uses a dense-`z` selection oracle, and is not BPB/rank/rate. A successor must change the residual/geometry and pass exact byte, quality and engine gates |
| **Routing and selection** | learned gate predictors; static/mass-ranked/output-greedy selectors; post-hoc shared linear residual over mass selection; trained routers; granular MoE; frozen H4+H2I cross-composition | F1 characterizes predictability; E23/E37 price real router; E38 rejects mass-based selection at `k=3`; E67's output-greedy arm remains only 4.23% better in local SSE. E68's rank-64 combined local residual reduces aggregate SSE by 53.33% but fails complementarity because its benefit is layer-concentrated and layer 1 worsens; shared-only is worse than mass-only. H1 S3 closes ternary 8-layer joint trainability; H2I v4 learns held-out score and beats STATIC routing but fails strict combined rank; H5 frozen composition is adverse | mass/output-greedy post-hoc **CLOSED at tested scope**; E68 shared linear residual **MEASURED-SCOPED `PARTIAL_LOCAL_SIGNAL`**; loss-aware selector and jointly trained nonlinear shared path **OPEN/UNMEASURED**; trained one-byte score+rank **CLOSED `SCORE-ONLY`**; frozen H4+H2I transfer **CLOSED negative** | Do not treat local output-error wins as BPB/rate or teacher-forced behavior as free-running success. E68 does not price a cheap shared component or replace the dense-`z` oracle; H5 does not disprove fresh joint training |
| **Organ allocation / composition** | FFN, q/k/v/o, head, norms; per-organ ternary/fp32; cheap components combined; parameter mass moved from FFN to attention; frozen H4 rank48 q/o + H2I R8 FFN | T2b/E18/E21/E22/E34/E36/E39/E40 price organs and composition. Cheap isolated changes can compose super-additively; attention+head set a floor; R128 shape can be fast with live synthetic FFN. H5 reproduces H4 then worsens BPB, rank and free agreement when independently trained H2I FFNs are installed | economics **CLOSED for current engine/shape family**; frozen H4+H2I composition **CLOSED negative** | Do not add isolated deltas to predict a mixed artifact or treat H5 as a test of fresh joint training. A reopening needs a jointly specified geometry, its own step-zero control, and exact-composition quality/rate measurement |
| **Training regime** | readout-only, on-policy/DAgger, CE, cross-tokenizer span-KD, STE/QAT, alpha-QAT, MoE upcycle, reverse-KL, resume/DDP | native MVE: CE beats off-domain span-KD; QAT shock recovers; A→F and resume pass. H0/H1 show donor format/carve can train. H2I v4 shows trained one-byte score/routing but misses its combined rank gate. Phase-64 S0 final number lacks its offline gate verdict | donor H2I **CLOSED `SCORE-ONLY`**; native S0 **PARKED** | Do not launch old H2T or rerun H2I as written. Do not generalize off-domain KD failure to on-domain KD. A later checkpoint is not automatically better: use the predeclared adjudication, no checkpoint selection |
| **Training order / inversion** | QAT before/after structural switch; magnitude-matched dense→MoE surgery; reverse-KL final stage; CE versus KD | Phase-64 records an arm-independent D→E discontinuity and the ε-identity law; S0 applied alpha-QAT then upcycle and gained most in F, but no causal arm isolates why | **MEASURED, mechanism OPEN/PARKED** | Do not infer that reverse-KL caused the S0 gain or that ordering is solved. Reopen only as a one-variable order A/B with transition-BPB continuity |
| **Conversion and export** | PyTorch↔engine parity; >2 GB loading; RMSNorm fold; packed/int8/tagged-v2; per-matrix/per-layer kinds | E1/E2 establish parity; E63 adds exact mixed container mechanics and its A10B rate; E66 runs a real 7B one-byte model. Tagged-v2 can represent H2I, but current exporter cannot emit the exact mixed trained object | base path **ADOPTED**; H2I seam **CLOSED (combined gate failed)** | Do not implement H2I export: v4 did not pass score+rank. Do not rerun the fired E63 A10B gate or alter its pinned runner; E61c/3B ancillary cells remain separately unmeasured |
| **Kernel, packing and memory layout** | nibble/trit packing; streamed vs gathered rows; tile-major LUT; factored/int8 matvec; FMA chains; cache/L3/DRAM regimes | P1/P2/E9–E13/E25/E26/E28/E30/E31/E58/E60/E61 quantify exact paths. E63 Part B measures a mixed A10B path: 106,168,320/928,251,904 charged weights are int8, the remainder stays packed, and charged bytes rise 464.13→517.21 MB/token. Gathered bytes are not streamed bytes; one host has format-specific bandwidths; FMA chain is not the binder | major paths **CLOSED/measured** | Do not transfer a bandwidth denominator, E63's 45.83 G charged-weights/s descriptive rate, or its 49.37 tok/s to another format/layout or a full-int8 donor. A kernel change requires parity plus same-format trained-model fidelity, not only synthetic GEMV speed |
| **Attention, softmax and context kernel** | serial reduction; independent accumulators/AVX4; GQA/KV traffic; real-exp/libm/errno; full attention lever stack | E4 and E48–E54 isolate and fix the earlier disabled/serial path; E40 explicitly exhausts the registered attention levers | **CLOSED for “next easy attention lever”** | Read E40 before proposing another accumulator/KV/softmax pass. Reopen only for a changed algorithm, hardware or correctness contract, and preserve `CONFIG` provenance |
| **Threads, occupancy and measurement environment** | output parallelism; OpenMP region/wake policy; clean/busy boxes; page-cache/occupancy guards; paired order reversal | native Phase 63 adopts bit-identical threading. E44/E52 establish interval/occupancy discipline. E63 Part B passed 45/45 preflight and 10/10 timing cells, yielding A10B's admissible CI | protocol **ADOPTED**; A10B absolute rate **MEASURED-SCOPED** | Keep occupancy and interval discipline. Do not treat the earlier refusals as measurements or rerun the fired A10B gate; any ancillary repair or changed environment is a separately scoped result |
| **Tokens per weight pass / decoding** | block verify; state look-ahead; speculative/self-drafting MTP in prior art; vocabulary token-worth | native block-verify is exact but does not pay at 8.3M; state carries ~1 reliable future step. External MTP gains are prior art, not a local target-scale result | local sandbox question **CLOSED**; target-scale composition **UNMEASURED** | Do not quote literature MTP multipliers as SiliconLLM speed. Reopen only when weight streaming dominates enough to amortize verification and include acceptance-rate measurement |
| **Metrics and adjudication** | BPB; chance distance; free-running and teacher-forced rank; full trajectories; parity/logits; tok/s intervals; context/token value | E16–E20 prove BPB and greedy rank can disagree or invert; E66 passes score but misses 150/160 rank; H2I freezes both rank modes and score. E43 proves a token is not a fixed amount of text | protocol **ADOPTED** | No quality promotion from BPB alone; no speed promotion from kernel rate alone; no free-running rescue by teacher-forced rank. Score gate runs before H2I rank, exactly once |

## 4. Shape and scale ledger — what “large” currently means

| object | real trained/pretrained weights? | quality measured? | engine arithmetic/parity? | rate measured? | licensed statement |
|---|---:|---:|---:|---:|---|
| Native ~8–22M sandbox | yes, trained from scratch | yes | yes | yes, cache-resident | validates architecture properties and engine kernels, not billion-scale streaming |
| Phase-64 MVE/S0 ~30M | yes, trained from scratch | MVE gates yes; S0 BPB yes but property gates absent | not exact S0 export | GPU training throughput, not target CPU decode | machinery works; S0's 0.9277 is a measurement without its registered verdict |
| Qwen 0.5/1.5/3B donor | yes, pretrained | yes across multiple formats | yes | selected format rates, scoped | anchors format/rule trends; E62 forbids a smooth scale law |
| Qwen2.5-Coder-7B donor | yes, pretrained | fp32, ternary and one-byte score/rank | yes | no valid one-byte target-rate result | one-byte fidelity survives; strict rank bar does not |
| Qwen2.5-Coder-14B candidate | no local weights; official metadata plus derived arithmetic only | no | family-level format compatibility only; no artifact parity | no | 2.958 tok/s is a conditional quotient at E40's observed packed effective rate, not a physical bound or donor prediction; do not download before a separately gated head+body transformation |
| Qwen3-30B-A3B-Base candidate | no local weights; official metadata plus derived arithmetic only | no | current carved path is not exact Qwen3 MoE | no | 13.607 tok/s full-stream and 33.657 tok/s no-expert conditional quotients use E40's observed rate, not impossibility proofs; defer exact support until head+attention also change |
| E36/E39/E40 ~10B shapes | no: synthetic/noise weights | no donor-quality result | yes for their engine paths | yes, bands/controls as documented | the CPU can execute a suitable 10B **shape** at/above the target rate |
| E63 A10B mixed ~10B artifact | constructed exact-format artifact, not a pretrained 10B quality object | no | yes | `G-E63d=DESK-MODEL-HELD`: 49.37 tok/s, CI [49.18, 50.49]; `G-E63e` ratio 0.9952 | synthetic mixed-format rate gate closed: only 106,168,320/928,251,904 charged weights (11.4%) are int8; no full-int8 rate, trained 10B quality, or strict 50-tok/s pass; E61c/3B ancillary sweeps incomplete |
| H2I 8-layer mixed donor object | donor weights plus trained factors/router/carved layers | Phase A and Phase B score measured; strict combined rank failed | container capable, exporter gap audited but closed by verdict | no | trained one-byte carve improves held-out BPB and routing, but is terminal `SCORE-ONLY`; no export/scale claim |
| H4 rank48 ternary q/o, 1.5B donor | yes, pretrained body plus trained low-rank attention factors | CPU fp32 terminal 1.1537375 BPB; 66/160 teacher-forced, 6/160 free; `TRAINING_HELPS`/`BEATS_QO96`/`BEATS_QO192` true, `GENERATOR_PARTIAL` false | no engine export | no | training recovers score/rank beyond QO192 post-hoc but generation stays AT-FLOOR; any Stage B requires a new brief, with no 10B/format/rate transfer |
| H4D same trained masters, fp32 A/B inference | yes, same pretrained 1.5B body and H4 terminal checkpoint | 1.6104023 BPB, 54/160 teacher-forced, 1/160 free; worse than H4 ternary | no engine export | no | quantizer-bypass diagnostic already measured; do not repeat or claim separately trained fp32 would fail |
| H5 frozen H4 rank48 q/o + H2I R8 FFN | yes, same pretrained 1.5B donor with independently trained components | BPB 1.399168, 55/160 teacher-forced, 1/160 free; all transfer clauses fail against H4 | no engine export | no | terminal negative frozen-cross-composition result; it neither tests nor disproves fresh joint training, and carries no 10B/rate claim |

## 5. Apparent contradictions and measured inversions

These are not invitations to average the two sides; they are boundaries that future work must carry.

| apparent contradiction / inversion | correct resolution |
|---|---|
| Native ternary costs little, while donor ternary collapses | QAT/from-scratch ternary MLP and post-hoc Transformer ternarization are different estimands. Training into the format survives; post-hoc conversion does not preserve donor rank |
| One-byte conversion is nearly lossless, but one-byte carve is worse than ternary carve | E66 isolates the uncarved format; E64 isolates post-hoc structure on that format. The byte conversion is not the remaining quality wall; selection/carving is |
| E40 exceeds 100 tok/s while E63 A10B is near 50, but no working 10B is demonstrated | Both are synthetic, format/shape-specific observations: E40 is not a physical `41.389` Gweights/s cap, and E63 is a mixed artifact rather than a full-int8 stream. Neither establishes pretrained quality; the joint claim needs one exact useful artifact measured for both quality and rate |
| Lower BPB can have floor-level greedy agreement | E16–E20 measure discontinuous rank and exposure drift. BPB, teacher-forced rank, free rank and trajectories are separate gates |
| Fewer active weights should be faster, yet gather can lose | E26/E31 show gathered traffic and per-expert overhead; the cost depends on layout/granularity, not byte count alone |
| Attention was the bottleneck, then attention levers were exhausted | earlier runs used or priced different kernels. E49/E50 correct disabled fast-attention provenance; E40 is the terminal registered sweep |
| Recall is “validated at 128K,” yet target-length recall is open | 128K **CPU query cost** is measured. Direct quality is established at shorter context plus proxies/structure; integrated target-length benefit remains unmeasured |
| Phase-64 S0 improves to 0.9277 but “has no verdict” | the scalar is valid, but the pre-registered matched-dense/property gates were never run. A number without its comparator cannot decide the rung |
| Cross-tokenizer KD failed, yet KD is not globally dead | the measured MVE teacher/domain was off-domain and CE won. The recipe correctly flips to CE-primary; only on-domain KD remains a legitimate challenger |
| Format quality varies non-monotonically with scale | E62 measures exactly that. Do not smooth, fit an exponent or extrapolate to 10B |
| H4 recovers score/rank, H4D is worse in fp32, and H5 composition is adverse | H4 remains generatively AT-FLOOR; H4D only bypasses ternarization on that same STE checkpoint; H5 tests frozen independently trained components. None proves a rank-capacity law or rules out fresh joint training |

## 6. Unmeasured cells that are real — and their priority

| priority | cell | why still open | action rule |
|---:|---|---|---|
| — | no inherited immediate experiment | E63 A10B is complete and H4/H4D/H5 are terminal; E61c/3B are ancillary cells that the runner did not reach | do not rerun fired gates or infer an automatic next launch; a new experiment needs a changed coordinate and its own brief |
| parked | Phase-64 S0 offline property gates | checkpoints exist; GPU training is complete, but dense/sparsity/router/recall/QAT gates were not read | CPU-only if the native branch resumes; do not spend donor quota or treat 0.9277 as the missing verdict |
| prior-art gap | direct integrated target-length recall benefit | cost and representation pieces exist separately | requires a new fused-model quality experiment, not another ANN microbench |
| future target | joint pretrained ~10B quality + >=50/100 tok/s | evidence still lacks one useful target-scale artifact measured for both metrics; E63 supplies only synthetic mixed-format rate, and H5 closes the frozen composition shortcut | define and train an exact deployable geometry, then preregister its quality, engine parity and rate gates |

H1 closes its ternary 8-layer cell, H2I v4 closes the corresponding one-byte score+rank cell as
`SCORE-ONLY`, and H5 closes the frozen H4+H2I composition shortcut negatively. Score/routing can
learn, but the relevant rank/generation gates do not promote a deployment. None licenses an extra
GPU launch, post-hoc carve rerun, one-byte/rank promotion, H2I export, or an automatic next action.

## 7. Mandatory no-duplication checklist for every new brief

Before assigning a new ID, write all of these into the brief:

1. **Nearest prior cell:** experiment ID and exact cited artifact.
2. **Changed coordinate:** donor/family, model nature, scale/shape, format, training regime, domain,
   engine path or estimand. “Try another value” is insufficient when an oracle or terminal sweep
   already closed the mechanism.
3. **Why the prior result does not answer it:** one falsifiable sentence.
4. **Paired controls:** same binary, same slice, same ordering policy, explicit engine `CONFIG`.
5. **Metric separation:** quality, rank, trajectory, arithmetic parity and rate are not proxies for
   one another.
6. **Scale discipline:** mark every number as measured, derived or literature; never promote a
   synthetic shape or a non-monotone ladder into a scale law.
7. **Void rule:** failed controls/apparatus produce no scientific cell. Repair narrowly and preserve
   the void record.
8. **Stop rule:** state what failure prevents the next export, rank run, training stage or scale-up.

If the changed coordinate cannot be named, the experiment is a duplicate and should not run.
