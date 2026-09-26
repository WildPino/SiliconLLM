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

The current selection rule is **provisional**: start with the locally bound
pretrained sparse GigaChat for a conditional-capacity case; retain StdMoE
E128 as a second sparse family only if a transformation addresses both its
measured W4 fidelity and active-byte failures; use dense Qwen to test
whether a different route can create conditional structure. Do not claim
one procedure handles these donors until their full pipelines pass.
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
No T4 run is authorized by these screens.
NES-01 has concluded with a failed joint gate. Its trained quality and C
pilot results remain useful target-geometry evidence. NES-02 closes the first
10× CPU cost probe and rejects parallel dense-row scoring as a sufficient
large-E optimization. NES-03 provides a numerical/cost-passing int8 router
option at pilot scale, with a fragile E128 latency margin. Independent larger
experts remain untrained, and no pretrained donor has been converted into a
quality-valid, ≥50 accepted tok/s native artifact. The next method decision
is a concrete donor-to-target transformation and paired quality/cost gate,
not another router-only optimization.

Operational experiment history, running processes and exact resumption point
live in [INDEX.md](INDEX.md); this file changes when a method step is actually
validated, rejected, or made executable.
