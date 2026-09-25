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

The dense-source transfer challenge is **Qwen2.5-1.5B** at revision
`8faed761d45a263340a0528343f099c05c9a4323`, whose local snapshot
contains `config.json`, tokenizer and safetensors. It is dense and has a
different mixer, so it tests a different branch of the procedure. Existing
training on 8/28 FFN layers improved a hard carve from 1.096636 to 0.962593
BPB, while the intact donor was 0.767595; that is an adaptation signal, not
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
pretrained sparse GigaChat for a conditional-capacity case; use the dense Qwen
case to test whether a different route can create the conditional structure.
Do not claim one procedure handles both until their full pipelines pass.
For another family, rerun the inventory and choose a variant from its actual
operator/traffic map; do not infer compatibility from model names or parameter
counts. The metadata-only Granite/LFM candidates are not local quality or
rate results.

## 2. Available operations and missing transformations

| Stage | Operation and implementation | Type | Current evidence / gap |
|---|---|---|---|
| A. Bind | Verify source revision, shard/tensor identity, tokenizer IDs and reference runtime; GigaChat `strat01_gigachat_source_binding.py` and saved reports | Exact identity check | GigaChat base source↔BF16 GGUF tensor bytes bound. Text→ID tokenizer parity in `engine.c` remains open. Qwen revision is pinned, but a new run must bind its exact local weight hash. |
| B. Establish donor | Score fresh held-out documents, task and greedy rollouts with the intact source and candidate under identical token IDs | Measurement | GigaChat Q4 versus BF16 passes scoped quality gates. Qwen intact anchor and limited-layer arms exist on a small frozen slice; a final fresh split is missing. |
| C. Decompose | List core/mixer/head/router/shared/routed tensors; calculate active and stored bytes, per-token selected experts and expected training exposure | Exact shape arithmetic plus measured kernel anchors | [METH-00](METH_00_GIGACHAT_COST_PREFLIGHT_20260925.md) gives an ideal W4 preflight; [METH-01](METH_01_GIGACHAT_Q4_ACTIVE_LEDGER_20260925.md) prices local GigaChat mixed-GGUF and [METH-03](METH_03_GRANITE_Q4_ACTIVE_LEDGER_20260925.md) the official Granite header. [NES-02](NES_02_CPU_EXPERT_COUNT_STRESS_20260925.md) measures a synthetic 10× native expert-count stress and prices dense routing at 10B/100B; it is no quality result. Cache residency and effective expert throughput are not assumed. |
| D1. Sparse-source variant | Preserve pretrained routed/shared functions initially, then selectively reduce expensive organs or alter routing/representation while keeping a compact reusable core; retain the original as a paired control | **Proposed approximation / adaptation** | GigaChat Q4 retains quality, but its actual mixed-format active payload is ~1,016 MB/token. Even ideal 2-bit MLA plus routed experts remains above the 14 ms design allotment at 40 GB/s. No transformed target with both quality and C speed exists. Full donor C port alone does not close this stage. |
| D2. Dense-source variant | Create a shared path plus residual experts with an economical input-only router; jointly adapt affected projections, router and experts, with continuous transitions and donor supervision | **Proposed training** | H1 shows training helps one carve; H4/H2I frozen composition and STRAT-03's tested local geometry fail. A new jointly specified geometry and step-zero control are required, not frozen assembly. |
| E. Export | Emit versioned C weights/metadata, tokenizer, precision map and golden intermediate/logit traces; run the exact timed C path | Exact serialization plus approximate kernels | Native E32 export/parity exists. GigaChat C fidelity is partial. No converted pretrained conditional target has completed this stage. |
| F. Validate | Paired donor→target held-out BPB with uncertainty, generation/task checks, routing utility, RAM/bytes/latency breakdown and ≥50 accepted tok/s on the same exported target | Measurement | **Open for every converted target.** Pilot/synthetic rate, scalar BPB, and partial port parity cannot be combined into a pass. |

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
the stated hardware yardstick. A candidate now requires a specified structural
or precision transformation, with a quantitative byte margin and paired
donor-relative quality gate; keep other organs fixed in each local sensitivity
cell, then verify the composed result. Reuse GigaChat's source binding and
quality results rather than repeating them. The next selection must compare
such a route against a tractable dense-source joint-training case, with an
explicit step-zero and resource estimate. These screens cannot establish
actual C speed or quality and authorize no T4 run.
NES-01 has concluded with a failed joint gate. Its trained quality and C
pilot results remain useful target-geometry evidence. NES-02 closes the first
10× CPU cost probe and rejects parallel dense-row scoring as a sufficient
large-E optimization; independent larger experts remain untrained.

Operational experiment history, running processes and exact resumption point
live in [INDEX.md](INDEX.md); this file changes when a method step is actually
validated, rejected, or made executable.
