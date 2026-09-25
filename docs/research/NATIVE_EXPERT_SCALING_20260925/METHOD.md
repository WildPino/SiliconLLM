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
target. The base Q4 large-matrix payload is calculated at 814 MB/token; 50
tok/s would require 40.7 GB/s for that payload alone, before routing, MLA,
state, scales, cache effects or sampling. This is arithmetic from the
[organ ledger](../donor_adaptation/audits/STRAT_01_GIGACHAT31_10B_METADATA_SCREEN_20260918.md),
not a measured C rate. The old C donor-port line is paused; its 54/64 passing
layer checkpoints and first normalization residual are reusable fidelity
evidence, not an automatic instruction to continue that port.

The transfer challenge is **Qwen2.5-1.5B** at revision
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
| C. Decompose | List core/mixer/head/router/shared/routed tensors; calculate active and stored bytes, per-token selected experts and expected training exposure | Exact shape arithmetic plus measured kernel anchors | [METH-00](METH_00_GIGACHAT_COST_PREFLIGHT_20260925.md) gives the GigaChat base traffic preflight. Native E32 ledger exists. Cache residency and effective expert throughput are not assumed. |
| D1. Sparse-source variant | Preserve pretrained routed/shared functions initially, then selectively reduce expensive organs or alter routing/representation while keeping a compact reusable core; retain the original as a paired control | **Proposed approximation / adaptation** | GigaChat Q4 retains quality, but its ~814 MB/token W4 payload gives insufficient margin by arithmetic. No transformed GigaChat target with both quality and C speed exists. Full donor C port alone does not close this stage. |
| D2. Dense-source variant | Create a shared path plus residual experts with an economical input-only router; jointly adapt affected projections, router and experts, with continuous transitions and donor supervision | **Proposed training** | H1 shows training helps one carve; H4/H2I frozen composition and STRAT-03's tested local geometry fail. A new jointly specified geometry and step-zero control are required, not frozen assembly. |
| E. Export | Emit versioned C weights/metadata, tokenizer, precision map and golden intermediate/logit traces; run the exact timed C path | Exact serialization plus approximate kernels | Native E32 export/parity exists. GigaChat C fidelity is partial. No converted pretrained conditional target has completed this stage. |
| F. Validate | Paired donor→target held-out BPB with uncertainty, generation/task checks, routing utility, RAM/bytes/latency breakdown and ≥50 accepted tok/s on the same exported target | Measurement | **Open for every converted target.** Pilot/synthetic rate, scalar BPB, and partial port parity cannot be combined into a pass. |

Transformations must record a quality/cost delta at each switch **and** after
composition. The tested H5 failure shows why independent passes cannot be
added together. Preserve source weights and any distinct learned expert
weights separately; count copied/tied weights once. The native E32→E128
[NES-01](E128_EQUAL_TOKEN_PROTOCOL_20260925.md) tests whether the target
geometry can gain useful capacity at fixed active expert work. It transfers
**no pretrained knowledge** and cannot satisfy stages D–F.

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

The bounded [METH-00 preflight](METH_00_GIGACHAT_COST_PREFLIGHT_20260925.md)
binds the already local GigaChat Q4/BF16 baseline to an organ traffic target.
The next donor-specific experiment must select one organ treatment, freeze
its paired donor-relative quality and byte gates, and keep other organs fixed.
Compose treatments only after checking each part, then test the integrated C
path. Reuse source binding and quality results; do not repeat them. If no
candidate transformation has a plausible byte/latency path, compare the
dense-source joint-training hypothesis with an explicit step-zero and resource
estimate. No T4 run is authorized by this preflight.
NES-01 continues independently as target-geometry evidence; no C timing is
run concurrently with its GPU training.

Operational experiment history, running processes and exact resumption point
live in [INDEX.md](INDEX.md); this file changes when a method step is actually
validated, rejected, or made executable.
