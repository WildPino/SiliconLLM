# Pretrained chatbot conversion into the original engine architecture

8 October 2026. Direction correction following the human's explicit concern.
Goal ACTIVE/INCOMPLETE. This record supersedes the order-stability fit as the
operational next step. All existing scientific results and closures remain.

## Executive Summary

The concern is justified. The latest Qwen converter replaces dense FFNs with
conditional SwiGLU functions but preserves all 24 Transformer attention blocks,
the original wide residual representation and the full large vocabulary head.
Its prepared native profile uses BF16/F32 arithmetic and direct routing. Calling
that profile through `engine.c` does not supply the architectural advantage of
the original selective SSM/SWA and ternary LUT backend. Its useful result is a
whole-output transfer experiment, whose fixed final recipe failed admission;
it is now retained as a diagnostic reference rather than the selected deployment
route. [1][3][4]

The original engine remains the starting target. Its small recurrent core,
bounded attention window and selected ternary functions offer a real mechanism
for separating stored capacity from active work. However, the original router
scans every expert, and its loader retains reference and unpacked copies. The
selected expert kernels can have constant work as the pool grows; the complete
original program has not established unlimited parameter-independent cost or
preserved large-donor quality. Both limitations have concrete remedies to
investigate: structured routing and packed-only deployment storage. [1][2]

The research priority changes to donor selection and staged architectural
adaptation. Published work supports Transformer-to-recurrent transfer,
pretrained FFN partition/upcycling and pretrained-to-ternary adaptation in
separate scopes. Their combination into our compact chatbot engine remains an
open experiment. A month or more of offline T4 work is an allowed possibility;
the next step is to determine a feasible target and a measured training budget,
then run one staged conversion pilot. Fresh chatbot quality and accepted >=50
batch1 IDs/s on the same artifact remain the joint destination. [6][8][9][11]

## Introduction: scope and method

This is a focused architectural decision, using the goal objective, original
default backend, retained project measurements and primary papers/model cards.
The repository remains the record of the method; this memo is accompanied by
`engine_target_review_20261008/{sources,evidence,claims}.jsonl` and its manifest.
No old measurement was replayed. New observations are code reads and public
model metadata, with no tensor values, inference, fitting or native timing.

The human permits lengthy adaptation and engine extensions serving the original
capacity-versus-active-cost thesis. This does not require byte-for-byte retention
of D256/L6/V1024 or every activation. It does require a complete costed target
before further training, followed by a real transformation from a pretrained
chatbot. The old 10 tok/s floor and 16GB footprint cap are historical, not the
current goal. A small fast student or a large synthetic bank alone is insufficient.

## Main Analysis

### 1. What the original engine actually establishes

The default E4M1 backend has D256, five selective SSM blocks, one SWA block with
window128, six gated-dReLU MoE FFNs, expert width128 and top8. It retains F32
control/projection/readout organs and uses ternary weight codes with quantized
activations for expert matrix products. E is read from the artifact header.
The dimensions and code paths are concrete; changing the total expert count
does not change those selected expert shapes. [1]

The retained Phase64 budget prices integrated expert traffic at about4.2GB/s
and the isolated synthetic t6 expert kernel at17GB/s. Its E128/E256 equal-speed
739–1309 tok/s rows are modeled. They are not trained large-pool quality results.
The useful E1280 Switch factor result also has a bounded scope: route plus
selected factors measured2.510ms versus2.017ms for its E128 control, without
LUT-coded arithmetic or accepted whole-model throughput. These results support
the mechanism and identify engineering costs; they do not settle conversion. [2][5]

For gated experts, ignoring padding and per-row scales,

\[
P_{\rm expert}=3LDhn,\qquad
P_{\rm active}=3LDhk,\qquad
B_{\rm selected}=b\,3LDhk.
\]

Here n is experts **per layer**, k is selected experts per layer, h is their
width and b is physical code bytes per coefficient. Total stored experts are
Ln. With the default byte code for two ternary coefficients, b=1/2; the optional
nibble layout stores four coefficients per byte, b=1/4 before padding/scales.
The former can use a byte to hold a four-bit pair code: representation entropy
and actual allocation are different. Keep that distinction in cost reports. [1]

These formulas establish a possible capacity dial. They do not establish that
arbitrary donor knowledge can be addressed at fixed k,h,D, or that an arbitrary
router can find its best experts at constant cost. The hypothesis to demonstrate
is useful conditional capacity at fixed active geometry, with all costs charged.

### 2. The present Qwen route misses part of that target

The latest E16 geometry still requires246,935,552 matrix products and494,017,536
logical coefficient bytes per decode step. Its head alone has136,134,656
products. All source attention remains. These are dimension deductions, not
physical DRAM or throughput measurements. Its 1280-update whole fit improved
outputs but failed all six absolute/category transfer gates. An order control
could diagnose that recipe, but would retain the same architectural mismatch. [3][4]

The decision is therefore to defer the proposed interleaving fit. Keep the
teacher corpus, canonical interaction tools, whole-output learner, numerical
checks and unexecuted native reference profile. They are useful apparatus for
later comparison. Do not promote the rejected checkpoint or treat an additional
fit of this fixed expensive core as the default route to the goal.

The complete deployment cost must instead be written as

\[
t_{\rm token}=t_{\rm core}+t_{\rm state/SWA}+t_{\rm route}(n)
+t_{\rm experts}(k,h)+t_{\rm head}(V,D)+t_{\rm glue}.
\]

SSM removes dependence on growing attention history from its state update; it
does not automatically make large dense projections cheap. Preserving donor
width/depth while replacing attention can still leave excessive active work.
Core compression, recurrent transfer, expert sparsification and precision are
distinct transformations, whose errors must ultimately compose.

### 3. The missing conversion has credible research precedents

*The Mamba in the Llama* initializes recurrent operators from pretrained
attention projections and reports hybrid chatbot transfer. Its example
distillation budget is less than five days on8x80GB A100, not a T4 measurement.
It mostly preserves shapes and transfers FFNs; it does not prove our smaller
core or ternary conditional conversion. Nevertheless, architecture change with
pretrained initialization and subsequent adaptation is a concrete route. [6]

The controlled Mamba comparison reports pure-SSM weaknesses in copying,
in-context learning and long-context reasoning, with stronger hybrid results.
This motivates retaining an attention capability. It does not establish that
an arbitrary finite SWA window preserves the donor's long-context behavior. [7]

MoEfication partitions pretrained FFNs and learns selection, reporting10–30%
FFN parameter use with over95% of original downstream performance in its tested
scope. Sparse Upcycling transfers dense checkpoints into trained sparse models.
These motivate source-informed initialization and adaptation, while neither
establishes our fixed small expert width or large chatbot preservation. [8][9]

BitDistill adapts full-precision pretrained models including Qwen to ternary
weights for specific downstream tasks, using distillation and a continual
pretraining warmup. ReLU Strikes Back supplies separate activation-sparsity
evidence. Native BitNet's producer card instead reports training from scratch
with ternary quantization; it is a useful format/control donor, not evidence
that post-training rounding alone preserves a pretrained chatbot. [10][11][12]

**Inference for this project:** initialization plus substantial staged adaptation
deserves priority over another local exact interpolant. No cited result validates
the conjunction compact recurrent core + useful small ternary experts + cheap
large-n routing + our hardware rate. That conjunction is the experiment.

### 4. Four conversion problems and their candidate solutions

**Reusable core and memory.** Construct a compact recurrent representation that
retains task-relevant history, with bounded SWA where necessary. For a Transformer
donor, initialize from compatible projections then distill recurrent transitions
and reduce representation/depth under an explicit cost budget. For a recurrent
or hybrid donor, reuse its state structure where compatible. Mamba1, Mamba2 and
other recurrences need explicit operator mappings; their labels do not establish
parity with the original scan.

**Conditional functions and redundancy.** Partition or extract source FFN atoms,
allow overlapping/redundant atoms, and train a compact shared function plus
selected private functions. Exact permutation/packing is distinguished from
omission, folding and retraining. More RAM can preserve additional branches and
redundancy; it cannot automatically recover missing information in a consulted
branch. Use ablation/removal/permutation to demonstrate useful stored capacity.
The old failed local constructions remain failures of their specified classes.

**Ternary/LUT execution.** Train with the actual exported ternary coefficients,
scales and activation quantization in the forward path, followed by C arithmetic
checks. Keep sensitive organs F32 initially, following the project precision
evidence. Gated-dReLU is the existing backend. A small gated-SiLU expert variant
is allowed if its ternary products still execute through the same LUT mechanism
and its complete nonlinear cost is measured. Changing activation is not required
merely for resemblance; ReLU zero-skip identities cannot be applied to SiLU.

**Addressing and readout.** Learn a structured/hierarchical selector whose
shortlist and normalization costs are included, checking both selected IDs and
mass. The original full scan remains an explicit reference. Preserve the first
donor's token IDs/chat/EOS contract and charge the whole head. A compressed
readout or changed tokenizer is a separate approximation with text-level and
behavioral comparisons, not a free speed improvement. Tokens above65535 also
require a suitable native ID format; the old u16 corpus is not a chat interface.

These solutions are proposals. They are connected through whole output and
own-history behavior: a faithful local response is useful for diagnosis, not an
absolute requirement that forbids a different internally adapted representation.

### 5. Donor families should be selected for target affinity

The first screening group comprises the locally available Qwen instruction
donor, a small pretrained recurrent/hybrid instruction donor, and a native
ternary instruction donor. Giga's actual pretrained weights and frozen branch
evidence remain available for a later scale/family variant; resume only a scoped
part justified by the new conversion, rather than its historical next step.

Falcon-H1-Tiny-90M-Instruct is a concrete hybrid candidate. Its producer describes
English instruction capability. A public configuration read at revision
`e6389502a0b12cd8da894b395ba5bf7436873b16` records D512/L24/FFN768/V32768,
SiLU, SSM768/state64,8 query heads/2 KV heads, no sliding window and null
attention-layer indices. These are metadata, not observed quality or a compatible
original-engine artifact. Its presence avoids assuming that all usable donors
must begin as dense Transformers. It does not warrant porting all its operators
unchanged or equating its capacity to a 10B donor. [13][14]

Qwen supplies reusable local supervision and interaction tools. Native BitNet
supplies a contrasting ternary starting point but retains Transformer attention
and squared ReLU. Select the first actual pilot only after comparing the complete
active cost, state/operator mapping, chat contract and conversion memory. Do not
download large weights or add a runtime solely because its model card is appealing.

### 6. Long training changes the strategy, not the destination

The human's month-plus T4 allowance is meaningful: staged recurrent adaptation,
quantization-aware recovery and whole chatbot alignment need not be restricted
to a tiny cached corpus or a one-hour fit. The new training data should cover
language, tasks, memory and evolving student histories. Existing cached labels
remain useful diagnostics; they are not sufficient evidence of that coverage.

Long wall time is not a substitute for memory feasibility. Keeping a 100B donor
frozen on CPU/storage, acquiring bounded teacher supervision and training a
compact core plus active experts are different from allocating dense Adam states
for all100B coefficients on a T4. Sparse routing alone does not make dense
optimizer storage sparse. Measure optimizer/activation/teacher/transfer costs
for the actual proposed recipe before extrapolating tokens or GPU hours.

Use a local prototype, then a short measured T4 feasibility pilot only after
communicating its reason, budget and stops. Extend toward weeks/months when
there is a functioning target-shaped learner, sufficient data and measured
quality recovery per compute. A flat pilot closes or changes that recipe; it
does not justify an automatic month of identical training. Published A100
budgets are research precedents rather than our resource forecast. [6]

## Synthesis: the revised target contract

The machine-readable companion `chatbot_engine_target_contract_v1.json` fixes
the architectural requirements and accounting categories. The original dimensions
are an anchor, not an admitted universal student shape. An extension must make
its contribution to bounded active cost reviewable, including readout, input,
state, normalization and dispatch. Total pool size and active work are recorded
separately. Large-n runtime is tested with actual varied accesses; useful capacity
is tested with pretrained information and intervention, not duplicate tensors.

One concrete deployment prerequisite follows directly from the loader: a packed
expert artifact/loader must avoid retaining full F32 and int8 reference copies.
Another follows from routing: a constant selected k does not eliminate the
linear E score scan. Both are extensions serving the original engine architecture,
and must be priced alongside conversion quality. [1]

## Limitations and Counterevidence Register

There is no admitted compact chatbot, no qualified combined conversion, and no
new measured speed in this review. The latest Qwen recipe is closed; individual
paper successes are not independent confirmations of the combined proposal.
Model cards/configurations cannot establish generation quality, conversion
feasibility or T4 throughput. The original tiny trained sandbox and large
synthetic pools have different evidential scopes. Full source-relative dialogue
and accepted rate are still required on the final same artifact. [2][4][5]

SSM alone does not compress the dense core, native ternary alone does not make
capacity selective, and upcycling alone does not reduce active expert width.
Changing vocabulary granularity changes tokens/s interpretation. These are
specific remaining problems rather than reasons to abandon the original target.

## Recommendations and exact next action

1. Implement a reusable **target-aware donor triage/accounting stage**, starting
   with existing Qwen census and pinned small hybrid/ternary metadata. Produce
   operator mappings, complete target coefficient/state/head/router/storage
   budgets and trainable/optimizer memory for one pilot. Metadata-only, local
   CPU target <=60s and <=5MiB report; no old model observation replay. Freeze
   the implementation and candidate inputs before execution. If a file/field
   is missing, record the gap rather than silently assume compatibility.
2. Select and preregister **one target-shaped staged learner**: recurrent/core
   adaptation, source-informed conditional initialization, actual ternary/LUT
   arithmetic, then whole chatbot recovery. Intermediate high-precision models
   are scaffolding; deployability is assessed against the final target. Determine
   order from measured error/cost, without requiring all four changes at once.
3. Export only an eligible candidate into the original backend or a justified
   extension. Check canonical input/EOS/state/operator behavior, fresh excluded
   own-history tasks and >=50 accepted batch1 IDs/s on the same complete artifact.
4. Expand useful distinct n with fixed active geometry and measured CPU routing
   IDs/mass/physical DRAM, then another actual family and larger donor scales.
   Do not declare the 10B/100B path from a tiny prototype or metadata alone.

The pending order-stability control and Transformer native profile are retained
as optional diagnostics. They are **not** the next selected experiment. No T4
allocation, heavy fit, C launch or new endpoint is initiated by this review.

## Claims-Evidence Table

| Decision-bearing claim | Evidence | Status |
|---|---|---|
| Selected expert work can be bounded while original routing/storage still grow |[1][2]|Code deduction plus scoped measurements |
| Latest Qwen route retains costly Transformer core and failed transfer admission |[3][4]|Dimension deduction and qualified experiment |
| Individual recurrent/conditional/ternary transfer routes have precedents |[6][8][9][11]|Published separate scopes; composition open |
| Hybrid donors merit target-aware triage |[7][13][14]|Inference and metadata, not pilot admission |
| Long offline adaptation is a viable research direction to budget |Human steering;[6][11]|Authorized direction; actual cost unmeasured |

## Bibliography

All sources read 8 October2026. Local paths resolve against this document.

[1] SiliconLLM (2026). "engine.c default E1M1/E4M1 backend". [Code](../../../benchmarks/phase60/engine.c). SHA in source registry.

[2] SiliconLLM (2026). "Phase64.1 Budget Document". [Retained measurements and model](../../PHASE64_BUDGET.md). SHA in source registry.

[3] SiliconLLM (2026). "Finite joint SwiGLU geometry". [Result](CHATBOT_COMPACT_GEOMETRY_RESULT_20261007.md). SHA in source registry.

[4] SiliconLLM (2026). "Actual whole-output adaptation". [Result](CHATBOT_WHOLE_OUTPUT_RESULT_20261008.md). SHA in source registry.

[5] SiliconLLM (2026). "Quality-valid centered E1280 factor access". [Result](METH_125_VARIED_CENTERED_FACTOR_CPU_RESULT_20260928.md).

[6] Wang et al. (2024). "The Mamba in the Llama: Distilling and Accelerating Hybrid Models". NeurIPS; v4 revised2025. https://arxiv.org/html/2408.15237v4

[7] Waleffe et al. (2024). "An Empirical Study of Mamba-based Language Models". arXiv. https://arxiv.org/abs/2406.07887v1

[8] Zhang et al. (2022). "MoEfication: Transformer Feed-forward Layers are Mixtures of Experts". Findings of ACL. https://aclanthology.org/2022.findings-acl.71/

[9] Komatsuzaki et al. (2022). "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints". arXiv; v2 revised2023. https://arxiv.org/abs/2212.05055v2

[10] Mirzadeh et al. (2023). "ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models". arXiv. https://arxiv.org/abs/2310.04564v1

[11] Wu et al. (2025). "BitNet Distillation". arXiv. https://arxiv.org/abs/2510.13998v1

[12] Microsoft (2025). "BitNet b1.58 2B4T BF16 model card". Producer documentation. https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16

[13] TII (2026). "Falcon-H1-Tiny-90M-Instruct model card". Producer documentation. https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct

[14] TII (2026). "Falcon-H1-Tiny-90M-Instruct pinned configuration". Producer metadata. https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct/resolve/e6389502a0b12cd8da894b395ba5bf7436873b16/config.json

## Methodology Appendix

Scope was set by the human's engine.c correction and unchanged goal objective.
Local original code and retained budgets were compared with the latest target;
independent search angles were recurrent transfer, FFN partition/upcycling,
ternary adaptation, activation sparsity and candidate hybrid metadata. Web claims
use primary papers or producer documentation; local claims bind retained results.
Falcon config was fetched through public HF API/resolve metadata, after browser
raw/blob access returned errors; revision is recorded and weights were not fetched.
No search-cli was available. Source clusters are recorded to avoid counting
multiple URLs from one study as independent replication. This is not an exhaustive
literature census. Claim verification distinguishes direct support, deductions
and proposals, with structural/citation checks recorded separately.
