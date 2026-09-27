# Pretrained-to-native conditional-capacity method (work in progress)

**Status: research procedure, not a validated conversion recipe.** This file
describes what can be reproduced now, what has failed, and what still needs an
experiment. The target is one identifiable artifact derived from a pretrained
LLM, retaining useful held-out, generative and task quality against that donor,
running in the project C engine at ≥50 accepted batch-1 tokens/s on a declared
CPU and context. The same artifact must pass quality and rate. A second donor
family or scale must test which steps transfer. More stored parameters alone
are not evidence of transferred capability.

## 1. Starting models and route selection

The initial case is **GigaChat 3.1 Lightning 10B-A1.8B**, source revision
`189fff27a1dee68473960c3d5bca53e0e07a3191`. Its base has 11.480B
distinct BF16 elements, 26 layers, 64 routed experts/top-4, one shared expert,
MLA, and a dense first FFN. Source shards, BF16 GGUF and Q4_K_M GGUF are
locally bound in the [source-binding record](../donor_adaptation/probes/STRAT_01_GIGACHAT31_SOURCE_BINDING_PROTOCOL_20260919.md).
The Q4 base-only artifact passed [fresh paired BPB](../donor_adaptation/probes/STRAT_01_GIGACHAT31_FRESH_BPB_PROTOCOL_20260919.md),
[PIQA](../donor_adaptation/probes/STRAT_01_GIGACHAT31_PIQA_RESULT_20260920.md),
and [document rollout](../donor_adaptation/probes/STRAT_01_GIGACHAT31_DOCUMENT_ROLLOUT_RESULT_20260921.md).
Those are source/format quality baselines, **not** proof of a low-cost native
target. A uniform ideal W4 would use 814 MB/token for its large matrices, but
the [actual mixed Q4_K_M descriptor ledger](METH_01_GIGACHAT_Q4_ACTIVE_LEDGER_20260925.md)
charges **1,016 MB of active compressed payload/token** including Q6_K,
Q5_0, F32 router, head and controls. At 50 tok/s that is 50.8 GB/s for
weight payload alone. These are calculations, not a measured C rate or DRAM
trace. The old C donor-port line is paused; its 54/64 passing
layer checkpoints and first normalization residual are reusable fidelity
evidence, not an automatic instruction to continue that port.

An alternative sparse/recurrent family is **Granite 4.0 H Tiny base**. The
[metadata screen](METH_02_GRANITE_H_TINY_METADATA_SCREEN_20260925.md) binds an
official revision with 6.939B total and 1.465B analyzer-active weights/token,
36 Mamba2 and four attention layers, 64 routed experts/top-6 and a shared
path. Its ideal W4 active payload already exceeds the 14 ms design
allotment at 40 GB/s. The [official Q4 GGUF header
ledger](METH_03_GRANITE_Q4_ACTIVE_LEDGER_20260925.md) charges 913 MB/token;
only the header prefix was fetched. Full local weights, a paired quality
baseline and a Mamba2 operator bridge are missing. It is a candidate, not
evidence that the method transfers.

Another locally bound sparse donor is **allenai/StdMoE_1b14b_1T_Preanneal**,
revision `d2a4949c9d4ad6cf47fbac131f7e020077332b21`: 16 layers,
128 experts/top-8 and one shared expert, with all 11 fp32 shards present
([acquisition](../donor_adaptation/probes/STRAT_02_STAGE0_ACQUISITION.md)).
Its existing [W4 paired gate](../donor_adaptation/probes/STRAT_02_W4_BF16_QUALITY_RESULT.md)
failed: +0.026025 BPB, upper one-sided CI95 +0.027178 against the
+0.02 bar. Restoring only the fp32 router improved the point to
+0.019532, but the [heldout upper CI](../donor_adaptation/probes/STRAT_02F_W4_ROUTER_F32_HELDOUT_RESULT.md)
was +0.020395 and still failed. That arm prices 676,397,056 active
bytes/token before native overhead: a 16.91 ms/token payload floor at
40 GB/s, leaving only 3.09 ms of a 20 ms total budget. Thus this real
E128 source is a **different-family transfer candidate**, not a
demonstrated quality-and-rate path; repeating its
already failed nominal W4 gate would not advance the method.

The dense-source transfer challenge is **Qwen2.5-1.5B** at revision
`8faed761d45a263340a0528343f099c05c9a4323`, whose local snapshot
contains `config.json`, tokenizer and safetensors, now hash-verified with
the H1 checkpoint in the [local readiness record](METH_10_QWEN_LOCAL_READINESS_20260926.md).
It is dense and has a
different mixer, so it tests a different branch of the procedure. Existing
training on 8/28 FFN layers improved a hard carve from 1.096636 to 0.962593
BPB at H1 S2 and **0.923490** at the later H1 S3 checkpoint, while the
intact donor was 0.767595; that is an adaptation signal, not
quality retention ([H1](../donor_adaptation/probes/H1_THE_CARVE_TRAINED.md)).
The frozen H4+H2I composition worsened BPB by 0.245430 and free generation;
do not reuse it as a final model or re-run the same cell
([H5](../donor_adaptation/probes/H5_CROSS_COMPOSITION_RESULT.md)).

Candidate selection currently uses these reproducible checks:

| Check | Input / tool | Output and decision |
|---|---|---|
| Identity and rights | Exact source revision, license, tokenizer, shard hashes; existing source-binding manifests | Reject an unknown or mismatched payload before scoring; record any terms that restrict use |
| Operator inventory | `benchmarks/donor_adaptation/donor_inventory.py analyze` for covered configs, plus the donor-specific tensor-header/organ ledger | Classify dense versus routed/shared experts; list mixer, router, head, tokenizer and unsupported C operators |
| Conditional cost | Count distinct stored weights, active weights per token, router and head separately; price bytes with measured hardware rates | If the active path cannot plausibly fit a 20 ms total budget, require a stated transformation before porting |
| Baseline quality | Paired donor/format held-out scorer, task and rollout instruments with pinned IDs and document split | Keep the source donor as the reference; reject a broken input or tokenizer mapping |

The current selection rule is **provisional**: use same-geometry
Qwen2.5-0.5B-Instruct as the next treatable dense donor for a new
zero-output residual-expert adaptation, after its [METH-42](METH_42_INSTRUCT_DONOR_PILOT_RESULT_20260927.md)
chat-format pilot passed the repetition screen. Its broad task quality
and packed/native conversion remain unverified. GigaChat remains the
locally bound sparse ~10B capacity case; retain StdMoE E128 as a
second sparse family only if a transformation addresses both its
measured W4 fidelity and active-byte failures. Do not claim one
procedure handles these donors until their full pipelines pass.
For another family, rerun the inventory and choose a variant from its actual
operator/traffic map; do not infer compatibility from model names or parameter
counts. The metadata-only Granite/LFM candidates are not local quality or
rate results.

## 2. Available operations and missing transformations

| Stage | Operation and implementation | Type | Current evidence / gap |
|---|---|---|---|
| A. Bind | Verify source revision, shard/tensor identity, tokenizer IDs and reference runtime; GigaChat `strat01_gigachat_source_binding.py` and saved reports | Exact identity check | GigaChat base source↔BF16 GGUF tensor bytes bound. Text→ID tokenizer parity in `engine.c` remains open. Qwen revision is pinned, but a new run must bind its exact local weight hash. |
| B. Establish donor | Score fresh held-out documents, task and greedy rollouts with the intact source and candidate under identical token IDs | Measurement | GigaChat Q4 versus BF16 passes scoped quality gates. Qwen0.5B donor and factor-0.50 adapter are paired on [METH-19](METH_19_HALF_RESIDUAL_INDEPENDENT_RESULT_20260926.md) separate documents. The stored R8 composition passes [METH-28](METH_28_R8_PIQA_COMPOSITION_RESULT_20260926.md) PIQA task retention on all 1,838 items but fails [METH-27](METH_27_R8_FRESH_GENERATION_RESULT_20260926.md) generation; broad task or code usefulness is not proven. |
| C. Decompose | List core/mixer/head/router/shared/routed tensors; calculate active and stored bytes, per-token selected experts and expected training exposure | Exact shape arithmetic plus measured kernel anchors | [METH-00/01/03](METH_03_GRANITE_Q4_ACTIVE_LEDGER_20260925.md) price sparse-donor organs. [METH-22](METH_22_QWEN05B_ACTIVE_LEDGER_RESULT_20260926.md) counts actual Qwen0.5B organs: ideal one-byte body plus fp32 tied head and E128 adapter still addresses 919.166 MB/token, 22.979 ms at 40 GB/s before compute; head precision must change under that design. [NES-02](NES_02_CPU_EXPERT_COUNT_STRESS_20260925.md) measures synthetic 10× native expert-count stress and prices dense routing at 10B/100B; it is no learned-quality result. Cache residency and throughput are not assumed. |
| D1. Sparse-source variant | Preserve donor structure initially, then apply an organ-selective low-bit representation from BF16, with source BF16/Q4 paired controls; adapt or change structure only if step-zero quality/cost requires it | **Measured approximation failure; new design open** | GigaChat Q4 retains quality but charges 1,016 MB/token. [METH-04](METH_04_GIGACHAT_LOWBITS_PREFLIGHT_RESULT_20260925.md) supplied a 414-tensor type map at 534.025 MB/token. [METH-05/06](METH_06_CYRILLIC_ROUTE_RESULT_20260926.md) repaired rare expert calibration and made an actual 534.025 MB/token GGUF, but it lost +0.239465 BPB. [METH-07](METH_07_ORGAN_PRECISION_ABLATION_RESULT_20260926.md) identified expert sensitivity; [METH-08](METH_08_Q2_REALLOCATION_RESULT_20260926.md) passed the active-byte gate at 533.140 MB/token but lost +0.275153 BPB; [METH-09](METH_09_Q2_EXPERT_ISOLATION_RESULT_20260926.md) found only a 0.023740 BPB benefit from Q2_K experts before the harmful head/dense payment. No quality-valid compact donor or C rate exists. |
| D2. Dense-source hard carve | Route a sparse subset of pretrained FFN channels, with an optional compact shared approximation | **Post-hoc full-layer geometries rejected** | H1 S3 improves an eight-layer trained carve but remains +0.155895 BPB behind donor. [METH-11](METH_11_SHARED_RESIDUAL_RESULT_20260926.md) kept 25% of Qwen FFN groups shared and routed 16 of 192 residual groups: +0.706052 BPB. [METH-12](METH_12_FITTED_SHARED_RESULT_20260926.md) fitted a rank-256 shared residual and routed 16/256 groups: local SSE improved 5.23%, but full pilot lost +0.943873 BPB. [METH-13/14](METH_14_QWEN05B_ORACLE_ROUTE_RESULT_20260926.md) show Qwen0.5B E128/top-32 remains +0.775726 BPB behind donor even with a non-deployable local-mass oracle. H4/H2I and STRAT-03 also fail in their scopes. |
| D3. Exact-core residual upcycle | Freeze the pretrained donor as shared core; add zero-output conditional residual experts; jointly train their factors and router; bound their output scale before a quality-preserving low-bit core | **Stored compact-core document/PIQA gates pass; generation fails; native rate open** | [METH-15/16](METH_16_RESIDUAL_EXPERT_CONTINUATION_RESULT_20260926.md) train E128. [METH-17/18](METH_18_RESIDUAL_SCALE_ROUTE_DIAGNOSTIC_RESULT_20260926.md) reject full amplitude and identify amplitude sensitivity. [METH-19](METH_19_HALF_RESIDUAL_INDEPENDENT_RESULT_20260926.md) fixes factor 0.50. [METH-24/25](METH_25_FRESH_R8_ARTIFACT_RESULT_20260926.md) export a 496 MB R8 core and pass new-document/ranking screens from its stored bytes. [METH-28](METH_28_R8_PIQA_COMPOSITION_RESULT_20260926.md) passes full PIQA retention. [METH-27](METH_27_R8_FRESH_GENERATION_RESULT_20260926.md) fails relative and absolute generation gates (16/24 loops); native promotion is held. The exhaustive router cannot scale with E, and no C accepted-token rate exists. |
| E. Export | Emit versioned C weights/metadata, tokenizer, precision map and golden intermediate/logit traces; run the exact timed C path | Exact serialization plus approximate kernels | Native E32 export/parity exists. GigaChat C fidelity is partial. No converted pretrained conditional target has completed this stage. |
| F. Validate | Paired donor→target held-out BPB with uncertainty, generation/task checks, routing utility, RAM/bytes/latency breakdown and ≥50 accepted tok/s on the same exported target | Measurement | **Open for every converted target.** Pilot/synthetic rate, scalar BPB, and partial port parity cannot be combined into a pass. |

[METH-13](METH_13_QWEN05B_JOINT_UPCYCLE_RESULT_20260926.md) adds a
Qwen2.5-0.5B dense-source CPU preflight. A fitted input-only router
selecting 32 of 128 donor-channel groups recovered 0.7917 of oracle
top-32 activation mass, but the full 24-layer hard carve lost
+1.360474 BPB. Its frozen +0.40 BPB stop rejected this geometry before
joint training. The groups are not independently trained experts. This
narrows the next D2 attempt to a genuinely new architecture with
donor-preserving warm start, explicit per-expert learning, and a
fresh heldout gate; route utility alone does not license GPU work.
[METH-14](METH_14_QWEN05B_ORACLE_ROUTE_RESULT_20260926.md) holds that
geometry fixed and selects top-32 groups using their actual post-SwiGLU
mass. It recovers 0.584748 BPB versus the fitted input-only router but
remains +0.775726 BPB behind donor, above its frozen +0.40 limit.
This non-deployable local oracle isolates a material fitted-route gap;
the residual loss could reflect truncation, the oracle objective, or
their interaction. It is not a formal upper bound for every learned router.
[METH-15/16](METH_16_RESIDUAL_EXPERT_CONTINUATION_RESULT_20260926.md)
change the mechanism: 128 separately trainable rank-8 residual experts
per layer start at zero over the intact frozen donor. Step-zero logits
are identical. After 1,024 RTX 3060 updates, paired internal BPB
improves by 0.016383; a router-row permutation raises loss by 0.042131.
This is a concrete pretrained-to-conditional training path, but its
large dense shared core and exhaustive router do not yet meet the
native 50 tok/s or 10B/100B scaling requirement. Both arms repeat
frequently in the fixed greedy prompt set, and fresh documents/tasks
are still owed.
[METH-17](METH_17_FRESH_DOCUMENT_TRANSFER_RESULT_20260926.md) supplies
the next quality check: 60 selected external/repository documents,
245,700 bytes, with the tested 256-byte fragments absent from both
earlier Qwen corpus files. The same checkpoint loses +0.013215 pooled BPB, above its
prospective +0.01 point limit; every selected code document loses.
The one-sided CI95 upper +0.015317 does clear +0.02, so the failure is
specific to the stricter joint gate, not a gross quality collapse.
This rejects native export of the current adapter and makes
donor-preserving generalization across domains the next D3 step.
[METH-18/19](METH_19_HALF_RESIDUAL_INDEPENDENT_RESULT_20260926.md)
make the next step concrete: a 0.50 output-factor transform of the
same trained E128 bank is exact in fp32 factor space and was selected
after a diagnostic on METH-17. It passes a second, disjoint 56-document
gate at −0.000737 pooled BPB with a +0.000133 one-sided upper bound
and +0.007235 pooled trained-route utility. The adapter-only
`safetensors` export is 187,177,472 bytes; the donor remains BF16 and
the route still scans all 128 rows. Most code documents still worsen,
and no useful generation/task or C-rate evidence follows from this
document result.
[METH-20/21](METH_21_HALF_ADAPTER_PIQA_RESULT_20260926.md) test the
exported adapter itself. Relative triple-8gram repetition passes
against donor (16/24 versus 17/24), but the absolute 16/24 and
code 7/8 loop rates leave useful free generation unproven. The full
1,838-item PIQA test retains donor accuracy within its prospective
gate (69.967% versus 70.620%, paired −0.653 points; lower CI95
−1.360 points). That is one task-specific relative result, not
broad capability transfer. [METH-22](METH_22_QWEN05B_ACTIVE_LEDGER_RESULT_20260926.md)
prices the full donor core: one-byte FFN/attention matrices with
the existing fp32 tied head and adapter still address 919.166
MB/token, or 22.979 ms at a 40 GB/s streaming yardstick before
compute. A compressed head and body must retain these quality
properties in composition before the C rate gate can be attempted.

Transformations must record a quality/cost delta at each switch **and** after
composition. The tested H5 failure shows why independent passes cannot be
added together. Preserve source weights and any distinct learned expert
weights separately; count copied/tied weights once. The native E32→E128
[NES-01](E128_EQUAL_TOKEN_PROTOCOL_20260925.md) tests whether the target
geometry can gain useful capacity at fixed active expert work. It transfers
**no pretrained knowledge** and cannot satisfy stages D–F. Its
[result](NES_01_E128_RESULT_20260925.md) improves BPB and passes pilot C
latency, but fails the frozen free-generation gate. Target expert-count
scaling therefore still needs a quality remedy; the BPB gain alone cannot
license E256 or 100B-scale claims. The intended capacity dial permits `E` to
grow with RAM, so future variants must price dense-router work, LUT expert
reads and free-generation quality as `E` grows, even at fixed top-k. The
[NES-02 CPU stress](NES_02_CPU_EXPERT_COUNT_STRESS_20260925.md) now quantifies
the 10× effect: E128→synthetic E1280 makes dense router+selection 10.002×
slower on the tested CPU, while selected-expert LUT time rises 1.285×. Its
parallel row-scoring candidate misses frozen speed gates. At this geometry,
an exhaustive fp32 router for 100B stored parameters would address about
1.039 GB/token, a 25.97 ms payload floor at 40 GB/s, already beyond the
20 ms/token target. A large-E method therefore needs bounded candidate
routing with measured route recall and quality, and a compact packed-only
expert export. Synthetic experts establish no learned capacity.
The [NES-03 int8 router](NES_03_INT8_ROUTER_SHORTLIST_RESULT_20260925.md)
now supplies one executable approximation: an int8 full-row sketch with a
32-candidate fp32 rescore. It kept the exact top-8 on the tested E128 and
synthetic E1280 C trajectories, preserved E128 BPB/top-1/greedy output, and
cut the E1280 router median from 536.1 to 116.8 µs/token with six threads.
It is an optional pilot component, **not** a validated large-E or donor
conversion method. The prototype still keeps full fp32 router and expert
reference weights in RAM; the E128 generation failure remains.

## 3. Verification contract and resource accounting

Each candidate run must pin donor revision/hash, tokenizer, source/target
artifacts, data document IDs/split, C revision/build flags, CPU, context,
batch, and exact command. Compare target to its own donor, not to an unrelated
model. Use document-level paired BPB and confidence interval, fixed
generation/task checks, and C-path parity before timing. The strategic
[roadmap](../STRATEGIC_10B_20260916/ROADMAP.md) gives a preferred
ΔBPB ≤+0.01 and mandatory one-sided CI95 upper bound ≤+0.02 for promotion;
subsequent tasks and rollouts still decide usefulness. The final rate gate is
lower CI95 ≥50 accepted tok/s, with prefill/TTFT and 100 tok/s reported
separately. A new experiment must freeze its exact criteria before observing
its result.

Record total conversion/training GPU-hours, tokens and per-expert exposure,
peak RAM/optimizer state, output size, resident versus DRAM-read bytes/token,
router scoring/selection cost, expert and core times, and total latency.
Top-k fixed does not fix dense-router cost; equal training tokens do not fix
per-expert exposure. Mark arithmetic, synthetic probes, measurements,
hypotheses and invalid runs distinctly.

## 4. Current decision boundary

[METH-01](METH_01_GIGACHAT_Q4_ACTIVE_LEDGER_20260925.md) shows that the
actual GigaChat Q4 baseline charges more active payload than the earlier
ideal W4 screen. The [Granite screen](METH_02_GRANITE_H_TINY_METADATA_SCREEN_20260925.md)
identifies a recurrence-bearing alternative. The subsequent
[official Granite header ledger](METH_03_GRANITE_Q4_ACTIVE_LEDGER_20260925.md)
prices 913.314 MB/token, also over the 560 MB streaming design allotment at
the stated hardware yardstick. [METH-04](METH_04_GIGACHAT_LOWBITS_PREFLIGHT_RESULT_20260925.md)
specified a GigaChat BF16→mixed low-bit map at 534.025 MB/token, leaving
25.975 MB beneath that allotment and only 5.975 MB beneath the stricter
540 MB actual-header gate. The 13.351 ms/token addressed-payload floor
would leave only 6.649 ms of a 20 ms total budget if quality held.
[METH-05](METH_05_SOURCE_ID_IMATRIX_RESULT_20260926.md) implemented
source-tokenizer calibration but found rare experts below its frozen
64-observation floor. [METH-06](METH_06_CYRILLIC_ROUTE_RESULT_20260926.md)
added distinct Cyrillic documents; the merged BF16 matrix passes that
coverage gate with a minimum of 94. An actual GGUF matches all 414 planned
types and 534.025 MB/token, but its nine-document donor-relative loss is
+0.239465 BPB, with each category over the frozen +0.20 gross-failure
stop. This rejects the current IQ2 map before full quality and C timing.
More calibration cannot be assumed to remedy the measured precision loss.
[METH-07](METH_07_ORGAN_PRECISION_ABLATION_RESULT_20260926.md) shows that
restoring Q4_K to experts or MLA separately recovers roughly half of that
loss, but costs 735.625 or 696.864 MB/token. The cost-qualified
[METH-08](METH_08_Q2_REALLOCATION_RESULT_20260926.md) swap raises expert
precision to Q2_K while lowering head/first FFN precision; its actual
533.140 MB/token GGUF loses +0.275153 BPB on the same pilot. The isolated
[METH-09](METH_09_Q2_EXPERT_ISOLATION_RESULT_20260926.md) Q2_K expert
arm gains just 0.023740 BPB over IQ2, while its head/dense payment
adds 0.059428 BPB of loss. These interventions do not establish a
quality/cost-valid representation. They narrow the next mechanism:
an explicit trained correction or a changed conditional geometry with
priced native export, rather than another unsupported precision swap.
The Qwen0.5B hard E128/top-32 carve fails. METH-15/16's additive
residual architecture passes its internal joint-training gate, but
METH-17 rejects its full-amplitude adapter on separate documents.
METH-19's bound half-amplitude adapter passes a new document gate,
while most code documents still worsen slightly. METH-20 passes a
relative repetition screen but exposes poor absolute generation;
METH-21 passes one full PIQA relative-retention gate. METH-22
identified tied-head precision as a cost-binding conversion variable.
METH-23 diagnosed R8 head/body compatibility on reused data;
METH-24 exported its exact rule as a packed-only 496.122 MB core;
METH-25 loaded the stored codes and scales and passed a new
48-document/top-1 quality screen with the bound E128 adapter. This
validates an artifact-level *PyTorch BF16-reconstruction* step for
Qwen2.5-0.5B, not an int8 execution step. The runnable tools, hashes,
limits and commands are in the corresponding [METH-24](METH_24_R8_CORE_EXPORT_RESULT_20260926.md)
and [METH-25](METH_25_FRESH_R8_ARTIFACT_RESULT_20260926.md) records.
[METH-27](METH_27_R8_FRESH_GENERATION_RESULT_20260926.md) then
tests generation from the stored bytes on 24 fixed prompts: pooled
loops remain 16/24 and technical loops worsen 2/8→4/8 relative to
R8 donor, failing both prospective relative and absolute gates.
[METH-28](METH_28_R8_PIQA_COMPOSITION_RESULT_20260926.md) scores
all 1,838 PIQA items from those same stored bytes; adapter accuracy
is 69.695%, −0.925 points versus BF16 donor, within its frozen task
retention gate. These results demonstrate why document BPB, ranking,
task and free generation must be separate checks on the *same*
composition. The current pair is held before native promotion. A
generative repair must use new data and prompts outside these audits,
then pass joint quality before C parity and accepted-token timing.
More stored expert capacity requires distinct trained E expansion and
sublinear CPU routing. [METH-26](METH_26_LARGE_E_ROUTER_LEDGER_20260926.md)
shows exhaustive int8 routing at E273,547 would read 5.882 GB/token
in this geometry, before selected experts and the core. The current
NES-03 shortlist still scans all E rows, so it cannot settle this
constraint by itself. [METH-29](METH_29_BALANCED_ROUTER_INDEX_RESULT_20260926.md)
then tests one post-hoc balanced index on actual trained E128 inputs:
32/64 candidate rows include only 62.484%/85.848% of the exact
top-4 IDs, missing its prospective route-fidelity gate. That closes
this centroid rule, not sublinear routing generally. A coarse route
must be learned or otherwise indexed with held-out recall before
CPU timing and larger-E claims.
[METH-30](METH_30_LEARNED_COARSE_ROUTER_RESULT_20260926.md) fits a
per-layer nonlinear coarse gate on separate source windows while
keeping METH-29's groups and E128 fine router fixed. On 147,456
reused external input-layer cases, 64 candidates include 91.609%
of the exact top-4 IDs and match 71.523% of full sets; internal
validation reaches 93.562% ID inclusion. Both arms fail the
prospective 99.9%/99% route gate, and the 64-candidate pilot already
costs about 129.14 D896 row-dot equivalents versus 128 exhaustive.
This closes the fixed-group distillation recipe before C lookup,
not a jointly trained hierarchy or another bounded index.
[METH-31](METH_31_RANK8_LUT_POOL_RESULT_20260926.md) measures the
current `engine.c` ternary LUT kernel in the Qwen L24/D896/rank8/top4
selected-factor shape. Its synthetic E1,280→E12,800 pool step grows
from 0.5505 to 5.505 GB, while six-thread selected-path latency
grows 0.4440→0.5046 ms/token (1.136×), passing the fixed 1.25×
and 5 ms component gates. The one-thread mode is faster, so this
projection shape should not use a per-expert six-worker launch.
The codes are synthetic, and neither real-factor ternary quality
nor bounded-router CPU cost nor end-to-end rate follows. The
next conversion decision needs a measured packed-factor quality
gate on a real stored artifact; the next route decision needs a
fidelity-passing hierarchy or index before C integration.
[METH-32](METH_32_TERNARY_FACTOR_RESULT_20260926.md) now gives that
first real-factor check: row-optimal ternary A/B factors are saved
in the exact LUT tile/code order, with fp32 scales and unchanged
router, then decoded from stored bytes for the R8 quality audit.
The 77.18 MB artifact improves pooled BPB by 0.000596 against
intact factors on reused METH-25 documents, but top-1 agreement
is only 96.012% versus its prospective 99% gate; changed hidden
states also alter the fine top-4 set in 13.668% of input-layer
cases. This **rejects the all-ternary A/B candidate** before
native promotion. This motivated a controlled A/B mixed-precision
diagnosis; new documents, generation and task checks must follow
any diagnostic pass. No actual-factor C rate is known.
[METH-33](METH_33_FACTOR_PRECISION_ABLATION_RESULT_20260926.md)
completes that diagnosis: A-ternary/B-fp32 and A-fp32/B-ternary
read 4.132 and 3.441 MB selected factor bytes/token, and both
pass reused-document BPB screens, but match only 97.021% and
97.282% of intact top-1 IDs against the fixed 99% gate. Neither
is selected for storage or fresh promotion. This led to the
higher-fidelity representation tested next.
[METH-34](METH_34_I4_LUT_FACTOR_RESULT_20260926.md) tests that
higher-fidelity representation: 15-level LUT-addressable codes
for both rank-8 factors, with fp32 row scales and original router.
The stored 132.23 MB artifact reloads exactly and selected
factor codes+scales address 3.788 MB/token. Reused-document
packed-minus-intact BPB is +0.000069, but only 6,041/6,144
top-1 IDs agree with intact factors (98.324%) against the
prospective 99% gate. It is rejected before fresh promotion.
The audit decodes saved weights in PyTorch; activation quantization,
C LUT parity and actual-factor CPU speed were not tested. Further
format tuning on these reused texts is paused until the parent
R8+E128 candidate passes generation and bounded routing is viable.
[METH-35](METH_35_LOW_RANK_ROUTER_RESULT_20260926.md) replaces the
failed fixed-group lookup with a frozen-router SVD index. On
reused route inputs, its fp32 rank-64 sketch plus 64 exact
candidate rows passes ≥99.9% top-4 ID inclusion and ≥99%
complete-set/category gates; five cheaper prespecified arms fail.
At E128 this arm performs 1.071× exhaustive float multiply terms.
[METH-36](METH_36_INT8_SKETCH_RESULT_20260926.md) stores fp32
bases, int8 row-scaled sketches and a disjoint 24-document
manifest. Decoded from its 5.720 MB stored artifact, the
rank-64/64-candidate index includes 99.939% of exact selected
IDs and matches 99.757% of full top-4 sets on different inputs.
This is a reproducible route-fidelity step, not a route-replaced
language-quality or native-speed step. Its one-byte sketch
would still read 420.17 MB/token at the hypothetical E273,547
point before scales, exact candidates, experts and core; the
corresponding CPU time and large-E learned routing remain unknown.
[METH-37](METH_37_ROUTE_REPLACEMENT_RESULT_20260926.md) applies
the stored C64 route to the actual R8+E128 model on 24
different METH-19 documents. Paired BPB, donor-relative BPB
and relative greedy-repetition screens pass, but only 98.356%
of next-token top-1 IDs match the exact-route model against
the prospective 99% gate. Route inclusion on a fixed original
trajectory is therefore insufficient to select this candidate.
[METH-38](METH_38_CANDIDATE_CAUSE_RESULT_20260926.md) increases
the exact shortlist to C96/C128 on reused prompts. C96 misses
only 9/589,824 exact IDs and reaches 99.349% top-1, but C128
still changes 21 outputs with all route sets identical.
[METH-39](METH_39_RESCORE_NUMERICS_RESULT_20260926.md) confirms
that gathering full `F.linear` scores restores exact top-1,
whereas candidate-only batched rescore changes more outputs
than elementwise summation. The full-score control is not
scalable; these reused-prompt diagnostics did not promote C96.
[METH-40](METH_40_ROUTER_SCAN_CPU_RESULT_20260926.md) measures
the saved rank-64 int8 sketch at a synthetic ~10B/~100B expert
ladder. At E27,355/E273,547, six-thread projection + exhaustive
scan/selection + merge takes 3.714/18.453 ms/token. The larger
point passes the predeclared 20 ms component ceiling but leaves
only 1.547 ms for exact rescore, selected experts, the core and
other inference work. Expanded rows are synthetic and do not
test quality with more independently learned experts.
[METH-41](METH_41_C96_INDEPENDENT_ROUTE_RESULT_20260927.md)
then freezes the C96 candidate-only elementwise rescore and
tests it on 24 source-disjoint code, prose and technical
documents. It passes the prospective E128 route-replacement
BPB, top-1, relative generation and donor-relative document
gates; top-1 is 6,099/6,144 and pooled loss is +0.000059 BPB
versus exact. The parent still loops on 15/24 C96 continuations,
so C96 is a quality-valid route component, not a useful full
artifact. A practical large-E design should avoid a full scan
or reduce its payload and prove route fidelity on distinct
trained experts. Rank, shortlist, score arithmetic and route
trajectory still need binding in a native executable export.
[METH-42](METH_42_INSTRUCT_DONOR_PILOT_RESULT_20260927.md)
tests the Qwen2.5-0.5B-Instruct revision with the same L24/D896
geometry and tokenizer as the base Qwen donor. On 24 common
chat-format summary inputs, BF16 base/Instruct/Instruct with the
old base-trained E128 adapter repeat on 7/0/0 prompts. Instruct
terminates all 24 with EOS, passing the prospective generation
screen, but manual inspection finds factual mistakes in some
summaries. The old adapter graft leaves pooled document BPB
within +0.002721 of Instruct but matches only 84.277% of its
prompt-position top-1 IDs versus the ≥95% gate. It is rejected:
matching tensor geometry does not make base-trained conditional
weights transferable across donor fine-tunes. New Instruct experts
must initialize at exact donor logits; the direct graft is rejected.
[METH-43](METH_43_INSTRUCT_ZERO_EXPERT_RESULT_20260927.md) implements
that exact-donor initialization with E128/top-4/rank-8 additive experts,
but a trailing-window chat loss fails donor prompt top-1 retention
(89.716%) after 16 updates. The failure shows that raw BPB and active
gradients alone do not select an instruction-quality parent.
[METH-44](METH_44_FULL_CHAT_RETENTION_RESULT_20260927.md) changes the
training objective: assistant-only CE plus donor KL on every position
of full chat prompt/response sequences, with two raw and two chat
microbatches per update and a lower LR. A fresh 24-prompt same-corpus
development screen passes at 95.981% top-1 after 16 updates, while
raw BPB is −0.001843 versus the BF16 Instruct donor. This is an
implemented transfer recipe at 0.5B/E128, not a finished conversion.
[METH-45](METH_45_INSTRUCT_CONTINUATION_RESULT_20260927.md) resumes
its bound optimizer/RNG checkpoint under a fixed 1024-update plan.
It stops at update 512 when chat top-1 drops to 93.985% below its
95% gate despite raw BPB −0.028953. The source-disjoint 12-document
chat manifest and PIQA task audit remain unopened because the
terminal training gate failed. This continuation recipe is rejected;
the failure does not refute the E128 conditional geometry.
[METH-46](METH_46_ROUTE_UTILITY_DIAGNOSTIC_RESULT_20260927.md)
tests that distinction: independently permuting learned router rows
raises raw BPB by +0.020330 at update 256, so matching router rows
to factors has measurable value on the viewed slice. The worst raw
expert load is 24.56× layer mean, and at least 102/128 experts are
selected in every layer. This supports a new retention objective on
the same conditional geometry while making load balance and
sublinear CPU routing explicit requirements for large E. The
permutation is a sensitivity control, not independent large-E
quality evidence. No T4 run follows from these screens.
[METH-47](METH_47_STRONG_KL_RETENTION_RESULT_20260927.md) resumes the
exact METH-44 donor/expert checkpoint with stronger donor KL on raw
(2.0) and full chat (4.0) positions and smaller factor/router LRs
(1e-4/1e-5). On a new disjoint 24-prompt same-corpus development set,
chat top-1 rises from 96.670% at resume to 98.270% at update 512;
raw BPB improves −0.015638. The previously frozen external audit then
passes its automatic gates: 12-document BPB −0.005033, prompt top-1
96.078%, 12/12 greedy EOS without loops, full PIQA 1290/1838 versus
donor 1291/1838, and route permutation penalty +0.013181 BPB. This
is the strongest currently implemented 0.5B/E128 transfer candidate.
Manual inspection of the saved summaries nevertheless finds specific
unsupported student claims. The automatic gates therefore do not yet
prove semantic conservation; neither packed export nor native C parity,
rate, nor independently trained large-E capacity has been validated.
[METH-48](METH_48_INT8_FACTOR_BANK_RESULT_20260927.md) and
[METH-49](METH_49_INT8_ROW_FACTOR_BANK_RESULT_20260927.md) establish
reproducible int8 export/reload tools for the actual METH-47 rank-8
factors, but reject both factor formats under the frozen ≥99% prompt
top-1 gate. Per-expert and per-row scales produce 95.839% and 95.744%
agreement with the unchanged-router checkpoint on the reused external
set even though pooled document BPB is almost unchanged. The exact
row-scale tensor payload is 430,848 bytes per expert across L24;
E27,355 and E273,547 project to 11.786 and 117.857 GB of factor
payload. This realizes the *storage arithmetic* of 10× more experts
with roughly 10× user RAM, while adding no evidence that the expanded
experts are trained, useful, reachable by a bounded router or fast in C.
No current one-byte factor representation is part of the validated
conversion method. A BF16-effective exact factor anchor and a diagnosis
of route/rank sensitivity are the next reproducible representation steps;
joint quantization-aware adaptation may be needed after a new frozen
quality set, rather than selecting a scale from viewed outcomes.
[METH-50](METH_50_ROUTE_AMPLIFICATION_RESULT_20260927.md) makes this
constraint more specific. On the same viewed prompts, row-int8 factors
change 6.58% of exact top-4 sets across the 24-layer route stream.
Replaying the original IDs with row-int8 factors restores only 11 net
top-1 positions, from 95.744% to 96.270%, still below the fixed 99%
gate. Replaying original IDs with original factors gives exact logits.
Factor error and downstream route churn are therefore both measured;
an improved router alone does not validate the tested int8 bank.
METH-47 load also remains concentrated on the viewed prompts (worst
single expert 18.30× uniform mean, despite at least 115/128 selected
per layer). Training a much larger expert set needs explicit use/load
telemetry as well as route fidelity and donor-relative quality.
NES-01 has concluded with a failed joint gate. Its trained quality and C
pilot results remain useful target-geometry evidence. NES-02 closes the first
10× CPU cost probe and rejects parallel dense-row scoring as a sufficient
large-E optimization. NES-03 provides a numerical/cost-passing int8 router
option at pilot scale, with a fragile E128 latency margin. Independent larger
experts remain untrained, and no pretrained donor has been converted into a
quality-valid, ≥50 accepted tok/s native artifact. The next method decision
is whether METH-47's donor-relative semantic quality survives a broader
frozen source-grounded generation audit. In parallel, a high-fidelity
packed export and bounded CPU router can be developed as diagnostics;
promotion needs semantic quality, C parity/rate on the same artifact,
and a distinctly trained expert-count ladder.

Operational experiment history, running processes and exact resumption point
live in [INDEX.md](INDEX.md); this file changes when a method step is actually
validated, rejected, or made executable.
