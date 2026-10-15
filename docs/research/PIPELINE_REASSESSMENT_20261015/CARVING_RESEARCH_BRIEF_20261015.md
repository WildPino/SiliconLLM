# Carving and transformed conditional execution: research brief

15 October 2026. Decision brief for a fresh architectural review. The pipeline
goal is incomplete. This document launches no numerical experiment and admits
no model. Result links are the authority for scope, controls and receipts.

## The question

How can pretrained computation be reorganized so that each token touches far
fewer weights while retaining useful access to the donor's distinct capacity?
Consider carving, changes of representation and amortized execution, including
methods that cross the original operator boundaries. Do not assume that keeping
the donor's dense core intact is mandatory, or that another small model alone
solves scaling. Separate an exact algebraic decomposition from a useful sparse
approximation and from a cheap executable selector.

The owner explicitly accepts a native MoE donor and one family that includes
very large models. Dense-to-MoE conversion and multiple unrelated families are
not prerequisites. See the [scope clarification](../NATIVE_EXPERT_SCALING_20260925/SCOPE_CLARIFICATION_20261015.md)
and [preserved objective](../NATIVE_EXPERT_SCALING_20260925/HANDOFF_SNAPSHOT_20261014/goal-objective.md).
The same useful chatbot artifact must reach at least 50 useful accepted batch-one
CPU tokens/s in the C engine on declared contexts. The intended separation is
large distinct stored capacity versus small active work, not merely fewer
stored bits or duplicated expert names. Quality need not mean bitwise identity;
the degree and limits of preservation must be demonstrated.

Hardware: Ryzen 5 3600X, six physical cores, 80 GiB RAM; RTX 3060 12 GiB for
bounded conversion/learning, not final inference. Several existing cost screens
use only three physical cores. Their failures are not universal six-core bounds.
Conversion resources and the cost of obtaining labels matter.

Historical component gates are evidence about their prescribed recipes, not
universal necessary conditions for useful chatbot quality. A new representation
may trade local approximation errors differently. If a proposed route cannot
meet those old local gates, explain how a fresh, prospectively fixed whole-model
quality test would decide usefulness; do not silently reclassify old failures.

## Evidence map

### 1. Expert subdivision is exact when all children execute

The [packed child-layout test](../NATIVE_EXPERT_SCALING_20260925/GRANITE_CHILD256_LAYOUT_RESULT_20261015.md)
copies a synthetic 512-wide SwiGLU expert into two 256-wide children without
requantization. All 6,144 Q4_K blocks are preserved. Both children with the
inherited parent mass reconstruct the full contribution at relative L2
5.58e-8. Parent payload is 884,736 bytes; one child addresses 442,368 bytes.
Q4_K's 256-value block boundary matters: naive 128-wide children waste padding.
Parent bias must be counted once; contribution masses must remain meaningful.

This establishes a real C format/arithmetic primitive, **not** quality after
omitting a child. Choosing source neurons or children cheaply is still open.
The fixture does not contain actual donor weights.

The [complete synthetic projection-bundle screen](../NATIVE_EXPERT_SCALING_20260925/GRANITE_CHILD256_COST_RESULT_20261015.md)
charges Q4 experts, I8 projections/head and actual router/child IDs and masses
for L24, D1024, E32, top8. One child halves expert addressed payload from
169.87 to 84.93 MB, but the I8 core/head still contributes about 125.83 MB,
before router/selector overhead. One-child three-process medians are
14.365/15.209/11.995 ms resident and 15.434/19.609/20.338 ms evicted.
All fixed groups fail the prospective 14 ms/stability allowance. Causal
attention, RoPE, KV and generation are absent. No quality pilot follows from
the fastest sample. This is a warning about the coupled cost, not a proof
that subdivision or all layouts cannot work.

### 2. Falcon FFN: conditional coverage exists, cheap selection is unresolved

The [exact remainder oracle](../NATIVE_EXPERT_SCALING_20260925/FFN_REMAINDER_ORACLE_RESULT_20261013.md)
uses authentic own-history pre-FFN inputs, all 2,048 output dimensions, sites
0/12/23, 64 FIT centers and all 4,608 source neurons. It approximates a source
FFN by an affine cell plus selected exact nonlinear remainder contributions.
Selection scores require **all source neuron functions** first. This is a
specified energy heuristic, not the optimum sparse subset.

| Site | FIT-chosen active width | DEV mean relative FFN error | Fixed-cell DEV error at same width |
| --- | ---: | ---: | ---: |
| 0 | 512 | 0.062173 | 0.400908 |
| 12 | 2,048 | 0.048711 | 0.404537 |
| 23 | 32 | 0.072805 | 0.113219 |

The local gates are mean <=0.10 and worst <=0.20; these rows pass the diagnostic
gates. They do not establish whole-model quality. In particular, site12 needs
2,048/4,608 neurons under this rule: claiming that 512 works everywhere would
be false. At512 its DEV mean is0.225681. Any affine shared operator and its
storage must also be charged; selected-row savings alone are incomplete.

The [PCA selector](../NATIVE_EXPERT_SCALING_20260925/FFN_SKETCH_ROUTER_RESULT_20261014.md)
and [source-derivative metric selector](../NATIVE_EXPERT_SCALING_20260925/FFN_ACTIVE_ROUTER_RESULT_20261014.md)
choose IDs before full reference evaluation. Neither admits a FIT-selected
candidate across the three sites. For the latter, site0 rank512 DEV mean
is0.137284; site12 rank512 fails even FIT; site23 FIT-selected rank64 DEV
mean is0.105607. The better geometry helps but does not solve selection.
This rejects those linear projected feature recipes, not every nonlinear,
hierarchical or jointly learned router and not every transformed neuron bank.

Potentially useful distinction: source-space sparsity and predictable source-space
sparsity are different questions. A transformation might change which functions
are selectable instead of merely predicting original neuron magnitudes better.

### 3. Falcon recurrent path: preserving depth leaves expensive context maps

The full-width source recurrence bridge has useful known-corpus behavior
(14/16 tasks; DEV KL about0.02156, disagreement4.8847%), but source attention,
FFN/head work and dense contextual generators remain outside that success.
The [full23 generator codec test](../NATIVE_EXPERT_SCALING_20260925/CONTEXTUAL_GENERATOR_CODEC_RESULT_20261014.md)
reduces coefficient/scales payload from631.57 MB BF16 to167.76 MB I4.
The qualified passive three-core I4 medians are22.806/23.840 ms resident/evicted,
before recurrence/readout/FFN/attention/head. Later joined generator/readout
measurements are also above20 ms. The often repeated13.8 ms is not a qualified
full-pipeline result or a demonstrated23-generator median in this receipt.
I4 worst source-field relative error is15.28%; arithmetic correctness against
decoded weights does not establish donor-quality preservation.

Exact activation-LUT recurrence around16.8 ms includes an internal recurrent
readout; that readout is **not the vocabulary head**. Replacing a vocabulary
head cannot remove it. Exact byte/trit LUT experiments did not provide a rescue.
Token/F0/causal-convolution compilation can eliminate a truly token-only part,
but a substantial context-dependent residual remains. Prior common linear
residual fits require large ranks within their tested family.

Unadapted identity skipping at3/6/12 blocks collapses quality. A compact
D512/L12/E72/top8 student after512 fixed updates passes FIT but fails both
old and new DEV badly (new DEV KL7.11/disagreement81.49%). These results reject
those schedules/recipes, not all depth changes after adequate learning.
Repeating the same fit longer without a new hypothesis is unsupported.

### 4. Actual GPT-OSS-120B core: two cheap representations now rejected

The [protected-core physical screen](PROTECTED_CORE_120B_CPU_RESULT_20261015.md)
charges source-shaped Q/K/V/O and F32 routers for36 layers. Synthetic Q4_K
weights occupy557.38 MB plus53.08 MB routers. Head, experts and causal attention
are artificially free. Three physical-worker process medians are34.39/50.95/
53.88 ms. The tested unchanged core/kernel already exceeds20 ms; this is not
a universal hardware or transformed-representation lower bound.

We then acquired only the genuine first source block and selected embedding
rows: [source receipt](FIRST_ATTENTION_SOURCE_RESULT_20261015.md). There are
24 FIT cases from8 domains and12 DEV cases from4 disjoint domains, at most256
tokens each; common62-token prefix excluded,4,498 variable queries.

**Uniform Q/O rank256 recipe:** [result](FIRST_ATTENTION_RANK_RESULT_20261015.md).
Source K/V, norm, biases, sinks, YaRN and window stay fixed. FIT-only
activation-weighted initialization plus512 fixed Adam updates gives branch L2
0.21176 FIT /0.29268 DEV against a passing full-F32 control0.00664/0.00659.
Branch gates are1% pooled /5% sequence p95; probability TV also fails.
Independent CPU replay verifies the numerical rejection. The producer completed
numerics but failed GPU-release cleanup; this fault remains recorded, not PASS.
Only that finite rank/fit recipe is rejected, not all rank256 optima.

**One source head-group with ideal mass:** [result](FIRST_ATTENTION_SOURCE_GROUP_RESULT_20261015.md).
64 query heads are split into16 contiguous groups of4. For every token an oracle
sees the true target and chooses the best group and **any signed real scalar**.
All-group reconstruction matches the F32 source within6e-7 relative L2.
Best one-group branch error is0.51810 FIT /0.49974 DEV, far above the same
1%/5% branch gates. Even an ideal selector cannot repair this fixed class.
The result does not constrain newly transformed banks or cross-head
reorganization. Do not propose merely training a router for these source groups.

The favorable36-layer one-group projection/router ledger was147.64 MB before
head/experts/attention/selector. Source routed experts still activate about
3.583B coefficients/token, or1.792 GB at ideal4bit without scales. Therefore
solving only attention, only experts or only the head is insufficient here.
No full120B checkpoint was acquired and no whole-model quality claim exists.

### 5. Historical useful successes and their limits

- [METH253](../NATIVE_EXPERT_SCALING_20260925/METH_253_SPLIT_PRIVATE_FEATURE_RESULT_20261001.md):
  Qwen0.5B actual24-layer private FFN native cost9.318 ms on six threads.
  This is a component, not the complete decoder. METH259 retains source-width
  operators and a1.321 GB archive;30,556 composite functions do not create
  new100B semantic capacity. METH266 PIQA70.18% versus70.24% donor differs by
  one question; METH265's72-generation health is good. METH267's blind panel
  has16 severe candidate errors versus15 donor, failing its frozen gate.
  These are valuable controls, not a completed large-scale conversion.
- [METH510](../NATIVE_EXPERT_SCALING_20260925/METH_510_NATIVE_HEAD_RESULT_20261006.md):
  full-vocabulary I8 screening plus top8 original-F32 refinement costs0.995 ms
  on Switch V32128/D768. Original I8 head alone is0.847 ms; refinement improves
  fidelity and adds cost. It is **not an ANN index**, does not avoid scanning
  all I8 rows and does not imply sub-ms Falcon/GPT-OSS heads.
- [METH511](../NATIVE_EXPERT_SCALING_20260925/METH_511_WHOLE_HYBRID_HEAD_RESULT_20261007.md):
  Switch-base1287.4B has warm89.98 accepted ID/s and52.56 prose ID/s with
  all15 fixed quality gates passing on bounded infilling; top1 agreement
  98.7--98.9%, generated edit0.0344. One economic all-book criterion fails,
  so the frozen recipe is ineligible. This is encoder-decoder infilling,
  not useful general chat. Switch-base256 prose34.78 ID/s does not pass50.
- Earlier exact greedy speculative verification and activation replay already
  share a weight stream across positions. Observed expert unions remain near
  independent routing: amortization factors about0.96/0.86 forK4/K8; cold
  accepted gain0.679x. Reuse needs changed route coherence or another mechanism,
  not an assumption that batching is free. Charge draft, union, per-position
  math/head, commit and rejected work per accepted token.

### 6. Earlier changed representations also need to be considered

The [catalog](../NATIVE_EXPERT_SCALING_20260925/EXPERIMENT_CATALOG_20261014.md)
indexes609 result-report files as of14 October; it does not claim every body
was deeply reviewed. The [primary readback](../NATIVE_EXPERT_SCALING_20260925/DECISIONS_FROM_CATALOG_20261014.md)
identifies additional traps:

- METH177/178/179 enlarged trained child pools but did not establish useful
  diversity: within-parent route-weighted B variation was only2.318%, and
  rotating child IDs did not robustly worsen BPB. Different hashes or slots
  are insufficient evidence of different useful conditional functions.
- [METH185](../NATIVE_EXPERT_SCALING_20260925/METH_185_PRETRAINED_FFN_CHANNEL_SPARSITY_RESULT_20260930.md)
  directly drops Qwen source neurons using exact activations. Even retaining
  2,048/4,864 yields16.49% median /20.85% p95 output error, failing1%/5%.
  It has no shared affine correction and differs from the Falcon remainder
  oracle. Neither result is an optimum-subset lower bound.
- [METH222](../NATIVE_EXPERT_SCALING_20260925/METH_222_CONDITIONAL_FUNCTION_RESULT_20261001.md)
  uses an affine common, shared whitened PCA64 and hierarchical affine cells
  on actual FFN input/output. E160 validation SSE/teacher energy0.4694 is worse
  than E16's0.4392; rotating leaves raises it to0.9635. Functional alignment
  exists without useful capacity scaling in this recipe.
- [METH225](../NATIVE_EXPERT_SCALING_20260925/METH_225_NONLINEAR_COMMON_DISTILLATION_RESULT_20261001.md)
  nonlinear common improves validation SSE to0.373 but fails its0.10 gate.
  [METH230](../NATIVE_EXPERT_SCALING_20260925/METH_230_SOURCE_ANCHORED_NONLINEAR_RESULT_20261001.md)
  source-anchored nonlinear specialists reach low FIT error but stop on a
  degenerate child with one unique input and no slope information. This does
  not establish validation failure or impossibility of nonlinear specialists.
- [Conditional output banks](../NATIVE_EXPERT_SCALING_20260925/CONDITIONAL_OUTPUT_RESULT_20261013.md)
  test actual Falcon recurrent readout inputs/outputs: tree16/r128 and
  tree64/r32 reduce addressed coefficient counts but have roughly48--90% DEV
  error. Exhaustive target-aware selection across all frozen affine images
  still leaves45--85% DEV error. Router improvement cannot fix those images;
  this is not a bound on nonlinear banks or a redesigned shared path.

## New literature and metadata leads, not local experimental results

The current bounded primary-source review is in
[ACTIVE_WORK_REVIEW_20261015](ACTIVE_WORK_REVIEW_20261015/run_manifest.json), with
source/evidence JSONL. Its metadata intake downloaded1.109 MB, no tensors;
config/code/index URLs are pinned to immutable revisions. Literature is a
lead to inspect, not evidence that the local pipeline works.

- LongCat-Flash-Lite has31.407B learned n-gram table coefficients. Twelve
  token/history-selected256-wide BF16 rows require only6 KiB row payload per
  token, followed by learned projections. This proves a different capacity
  access pattern exists in a public pretrained model. But its14 dual blocks
  still execute28 MLA and28 dense FFNs. The reproduced
  [favorable source-work ledger](ACTIVE_WORK_REVIEW_20261015/active_work_ledger.json)
  counts1.523 GB ideal payload/token,76.13 GB/s at50, before experts/cache/norms/
  padding/scales. Shapes are derived from pinned config/code and index names,
  not acquired tensor headers. Extracting its input-memory function exactly
  does not preserve decoder behavior in a compact recipient automatically.
- [Memory Grafting](https://arxiv.org/html/2605.20948v1) extracts frozen donor
  hidden vectors for frequent recipient n-grams offline. A trained recipient
  fuses them through learned projections/gates; misses use learned hash memory.
  Vocabulary mismatch can be bridged offline. Published recipient training is
  50--100B tokens, not an established cheap post-training transfer.
- [Engram](https://arxiv.org/html/2601.07372v1),
  [STEM](https://arxiv.org/abs/2601.10639),
  [STAPLE](https://arxiv.org/html/2610.12599v1) and
  [MoLE](https://github.com/JieShibo/MoLE) offer conditional-memory mechanisms.
  Distinguish token-static functions by construction from approximating an
  arbitrary contextual source FFN; some retain the dense FFN and most require
  substantial training. None supplies a demonstrated local cheap conversion.
- JetMoE and SwitchHead demonstrate conditional attention at smaller scales;
  PEER gives very fine product-key expert retrieval. No verified very-large
  checkpoint with a sufficiently cheap mandatory core emerged from the limited
  dated search. This is not an exhaustive absence proof. Qwen3-Next80B source
  header ledger METH397 still has2.054B nonrouted active coefficients; a hybrid
  recurrent label does not remove projections or required state updates.
- AQLM/QTIP/QuIP#/SpQR/BiLLM/T-MAC/Vec-LUT can change coding or kernel costs.
  No checked source establishes a near-lossless sub-bit Zen2 batch-one rescue
  of the complete source120B active path. GPU results, newly pretrained ternary
  models and parallel-token throughput cannot be substituted for this goal.

## Data already available for a decisive bounded test

- Actual first GPT-OSS attention block: ten BF16 tensors53.106 MB plus selected
  embedding rows8.646 MB;24 FIT/12 DEV first-user cases, saved BF16/F32 traces,
  saved selected head-group branches/masses and all16 optimal residuals, plus
  rank factors/optimizer traces. Full16 contribution tensors were not retained;
  they can be reconstructed from the sealed source and F32 probabilities.
  See source/rank/oracle bindings for exact locations and hashes. This supports
  local representation diagnostics, not a whole-model donor comparison.
- Actual Falcon FFN operands at the qualified sites, own-history captures,
  source-only compilers and audited oracle/selector outputs. Full source
  coverage capture has432 cases (360 FIT/72 DEV),76,512 labels, retained source
  scores and411 raw tensors. The [capture result](../NATIVE_EXPERT_SCALING_20260925/CHATBOT_SOURCE_COVERAGE_RESULT_20261015.md)
  and its binding govern use.
  Public corpus messages may be retokenized for another donor; Falcon logits
  are not labels for another model. Reserved cohorts remain sealed.
- Packed Q4_K child copying, scalar/AVX2 controls, CPU routing/mass kernels,
  complete C attention/KV and hybrid-head code exist. Component availability
  does not establish a quality-qualified composed target.
- The metadata review has pinned public configs/code/index but no additional
  donor tensors. No full large-model download or lengthy training is running.

## Requested research output

Give an independent answer to the carving question, not another audit of the
recorded runs. Inspect repository and primary literature as useful. Challenge
the framing where necessary. Consider changes of basis, shared-plus-conditional
decomposition, nonlinear function banks, neuron/expert repartitioning, token-static
extraction, compressed contextual corrections, reuse and cross-operator fusion;
these are prompts, not a required menu or preferred answer.

Choose at most two or three credible hypotheses and rank them. For each state:

1. What mathematical function is preserved, approximated or newly learned;
   what distinct donor capacity remains accessible, and what is lost.
2. How IDs **and masses** are obtained before expensive work; include search,
   correction, table construction/storage and any unavoidable source scan.
3. A complete per-accepted-token budget: shared maps, selected weights, format
   overhead, router, state/KV, head and frontend. Separate addressed bytes,
   measured DRAM, arithmetic and cache assumptions. Identify the required
   reduction factor and the conversion/training bottleneck honestly.
4. Why the proposal escapes the specific failed representations above; identify
   which failures still apply. Do not infer impossibility from a local gate.
5. One bounded decisive falsifier using retained real data where possible,
   with a favorable oracle before expensive learning when justified. State
   what PASS would and would not establish, stop conditions and cost limits.
   Freeze new data/gates before a later numerical run; consumed DEV is not a
   fresh confirmation set.

If no credible path survives, identify the smallest missing scientific premise
and an experiment that could establish it. Avoid selecting a smaller donor
without a concrete mechanism for very-large-family scaling. No new numerical
run, source acquisition or large training is authorized by this review brief.
