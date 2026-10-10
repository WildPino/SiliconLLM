# Literature follow-up: preserving a chatbot while changing its operators

10 October 2026. Focused primary-paper investigation requested during the stored
audit. Goal INCOMPLETE. This document adds algebraic conditions and research
options; it reports no new learner, source, GPU or native-model experiment.

## Executive Summary

The literature supports three separate parts of our objective: conversion of
pretrained sequence mixers, expansion into sparsely consulted functions, and
structured retrieval from very large parameter pools. It does not establish
their conjunction in our original CPU LUT/ternary/SSM engine. The distinction
matters because preserving a donor's learned function is a stronger starting
point than merely transferring its parameter values into different operations.

The most actionable addition is granular upcycling: it explicitly addresses
the information lost when a dense FFN is split, then sparsely selected. Its
initialization couples shard coverage with output scaling. [4] The user's idea
of storing redundant leaves has a precise interpretation as a coverage and
coefficient problem. Replication can preserve information, but selection must
still reconstruct the original contribution on the inputs that matter.

PEER supplies a second concrete mechanism: factorized product keys permit exact
selection within a constrained score family over more than a million tiny
experts. [5] It is relevant to the large-n router requirement, with explicit
limits on transfer from our current unconstrained router and on CPU bandwidth.

The conversion papers preserve much more donor structure than our current
student. [1][2] Width compression and ternarization have additional conditions
and recovery costs. [8][10] The next practical choice remains the already
proposed stored target/readout attribution, followed by a representation-aware
conversion diagnosis. The newly identified coverage and structured-routing
paths deserve bounded protocols after that diagnosis; paper results alone
justify neither a long unchanged training campaign nor a useful-speed claim.

## Introduction

The project target is a useful pretrained chatbot converted into the original
engine, with small active work and many useful RAM-resident functions. The
retained local experiment now contains 24 additional updates in each of two
matched arms. Their native DEV case KL is approximately 6.52, and both answer
zero of sixteen canonical tasks correctly. These are worker observations;
their independent stored adjudication is running separately. The useful donor
answered fourteen. See the [local result](ORIGINAL_JOINT_HISTORY_RECOVERY_RESULT_20261010.md).

This follow-up asks what published work actually preserves during conversion,
what redundant partitioning can preserve algebraically, and what scaling a
router entails. Earlier [strategic review](STRATEGIC_REVIEW_20261004.md) and
[chatbot reassessment](CHATBOT_ENGINE_TARGET_REASSESSMENT_20261008.md) already
introduced several papers. The new emphasis is their hypotheses, especially
virtual shard grouping and exact product-key selection, rather than treating
the references as independent endorsements of our complete pipeline.

## Main Analysis

### 1. Sequence conversion transfers a compatible surrounding computation

MOHAWK aligns sequence mixing matrices, then individual block outputs, then
end-to-end predictions. It transfers embeddings, norms, FFNs and the output
head while retaining the teacher's surrounding architecture. Its reported
budgets are 3B tokens for Phi-Mamba and 5B for the hybrid. [1] Mamba in the Llama
reuses attention projections and transfers FFNs, with chatbot evaluation;
one training example costs less than five days on eight 80GB A100s. [2]

**Project inference:** neither result validates simultaneously reducing residual
width 2048 to 256, collapsing twenty-four blocks into six, changing the FFN
activation, introducing normalized top-eight selection, and ternarizing weights.
Our current six-boundary loss is also different from MOHAWK's independent
block training on common teacher inputs. A failed simultaneous auxiliary loss
does not refute staged operator alignment. Conversely, those papers do not
prove that alignment alone will recover our smaller student.

An error decomposition should name each changed map: coordinate/width map,
sequence update, layer composition, FFN selection, activation/quantization and
decoder. Their interaction cannot be credited to a single scalar MSE. For maps
in compatible coordinates, with valid local error bounds epsilon_i and
downstream Lipschitz bounds L_j on the encountered states, telescoping gives

    ||F_student(x)-F_teacher(x)|| <= sum_i epsilon_i * product_(j>i) L_j.

This is our analytic template, not a numerical bound established for the
chatbot. Calibration-average MSE is not a uniform error bound. Local fits
measured only on teacher inputs may underestimate errors
after earlier student outputs shift the input distribution. We therefore
need both common-input diagnostics and whole student-history validation.

### 2. Redundant leaves require coverage and coefficient preservation

Net2WiderNet copies incoming neuron weights and divides outgoing weights by
the replication count, preserving the function under its elementwise-network
hypotheses. [6] Sparse Upcycling instead copies complete pretrained FFNs and
continues training. [3] Granular Upcycling partitions FFNs, replicates shards,
and initializes selection to cover the original shards with scale correction;
it warns that naive granular initialization loses the original function. [4]

**Our algebra.** Write a same-operator donor FFN as a sum of shard functions
f(x)=sum_j f_j(x). Let candidate expert r contain coefficients A_rj and compute
e_r(x)=sum_j A_rj f_j(x). Selecting S(x), with normalized weights g_r(x), gives

    f_MoE(x) = sum_j b_j(x) f_j(x),
    b_j(x)   = sum_(r in S(x)) g_r(x) A_rj.

The exact requirement is sum_j (b_j-1) f_j(x)=0. Componentwise b_j=1 is
sufficient; it is necessary when the shard values are linearly independent at
that input. Cancellation or distribution-specific redundancy can relax it.
This is why a universal claim based only on the number of selected shards
would be unjustified.

Copying a small shard into more experts does not by itself make absent shard
information appear in an active call. With normalized mass, copies of the
same complete function reproduce that function; copies of separate fragments
produce an average unless their output scale compensates. Overlapping experts
can help cover important contributions. Their benefit must be measured against
active width, memory reads and the reconstruction residual.

For the original down projection, the exact shard sum follows from linearity.
It survives replication with coefficient correction while keeping inputs and
elementwise activations identical. Our source SiLU to engine dReLU/AQ/ternary
transition breaks that premise, as does replacing a composition of four donor
layers with a sum of their extracted fragments. No coverage matrix repairs
those operator changes by identity alone.

Granular Upcycling's cubic weight scale is derived for squared ReLU. Its
SwiGLU use is empirical, so that formula must not be imported as an exact
SiLU/dReLU equivalence. [4] LLaMA-MoE separately reports an initial quality
decline after splitting and new routing, with recovery using 200B tokens. [13]
This is useful counterevidence against calling arbitrary sharding lossless.

For overlap design, the retained shard Gram matrix supplies a diagnostic:

    ||f_MoE(x)-f(x)||^2 = (b-1)^T G(x) (b-1),
    G_ij(x) = <f_i(x),f_j(x)>.

This equation describes vector-output error on that input, not task quality.
It suggests selecting redundant combinations by directional contribution
rather than channel count. Data-specific near-dependence must be tested on
held-out inputs; copied parameters themselves add no new linear directions.

### 3. Product keys solve a constrained selection problem at large n

PEER uses tiny single-neuron experts and product-key retrieval; its
experiments include 1024 squared experts, eight retrieval heads, and sixteen
experts per head. These are language-model pretraining comparisons, not a
pretrained chatbot conversion into our engine. [5] The foundational product-key
memory paper and Memory Layers at Scale provide related evidence for large
learned sparse capacity; the latter reaches 128B memory parameters. [11][12]

**Mechanism and our inference.** Represent the expert ID by a pair (i,j),
and require scores s_ij(x)=a_i(x)+b_j(x). Compute each side's top-k, then the
top-k among their k squared sums. A pair excluded from either marginal top-k
cannot exceed the kth combined score: holding its other coordinate fixed,
there are already k pairs with better marginal scores. Ties require compatible
lexicographic pair-ID ordering. This finds the exact top-k of the product-score family.
The router score work is roughly O(sqrt(n)*d + k^2), plus selection overhead.

It does not recover the exact top-k of an arbitrary existing flat router. The
factorization restricts score geometry. Therefore transferring our router
requires learning/reindexing the bank or demonstrating that the relevant
scores fit this form. Multiplying n by ten increases the marginal key count
by roughly sqrt(10), even at fixed active k. It does not produce constant
latency, and selected weights still travel through the memory hierarchy.

The local full-V normalized mass requirement can fit selected-logit softmax,
but mass equality must be checked together with IDs, ties and score precision.
CPU work also includes the query projection and memory layout. No paper
here proves physical DRAM latency on the Ryzen target. Memory Layers at Scale
explicitly identifies bandwidth dependence and unfinished efficiency work. [12]

### 4. Width, shared information and ternary conversion are independent costs

SliceGPT proves full-dimensional orthogonal coordinate invariance with
coordinated weight and RMSNorm transformations. Layer-dependent rotations
also require consistent residual transformations; deleting dimensions is a
subsequent approximation. [8] **Our inference:** a fixed rectangular projection
P is not such an invertible change of basis. The discarded directions and
normalization/decoder coupling must be assessed rather than assuming the
orthogonal identity extends to width deletion.

The softmax-bottleneck paper formalizes a rank restriction on logits and
log probabilities modulo per-context constants. [9] For our fixed learned
head W of shape V by 256, all pre-softmax vectors remain in its column space,
whatever the complexity of the expert bank. More expert choices can improve
the nonlinear history-to-state mapping; they do not enlarge this fixed output
space. This does not prove a quality ceiling: the donor distributions on the
relevant manifold might admit an adequate low-dimensional approximation.
Softmax probability matrices themselves are not constrained to rank 256.

DeepSeekMoE offers always-active shared experts alongside routed experts,
intended to consolidate common information. [7] **Project option:** decompose
f(x) into a small shared computation plus selected residual functions. This
could reduce redundant storage or simplify selection, but adds active cost
and has no measured donor-relative benefit here. It is an alternative to
overlap, with the same need to measure errors and actual bytes read.

BitDistill adapts full-precision models to ternary weights using SubLN,
continual pretraining and attention distillation for downstream tasks. Its
reported warm-up uses 10B tokens. [10] That supports staged adaptation; it
does not show that rounding, recurrent compression and general chat retention
can be combined without additional loss. Our per-row scales and original
operators also differ from its producer's quantization and runtime.

## Synthesis

Three recurrent design patterns have primary precedents: preserving
compatible donor structure while replacing a mixer [1][2][8], separating
replication from sparse coverage [3][4][6], and structuring selection to avoid
a full scan [5][11][12]. These are converging mechanisms, not three independent
replications of the same performance claim. Individual numerical results
remain specific to each paper's model, tasks and compute budget.

Our present pipeline changes all three patterns at once. The better research
sequence is to identify the failing map, recover a compatible donor function,
then progressively reduce active work and precision with measured losses.
The user's redundancy idea remains credible as a conversion tool, provided
the stored expansion buys a smaller measured reconstruction residual for
the same active budget. It is not a theorem that more copies improve routing.

## Counterevidence Register

LLaMA-MoE's immediate post-split quality decline [13], substantial conversion
training in MOHAWK and Mamba in the Llama [1][2], and BitDistill's task-specific
scope [10] limit optimistic transfer claims. The memory paper's bandwidth and
production-efficiency limitations [12] limit constant-speed claims. The local
0/16 behavior limits deployment claims even where KL has improved.

## Claims-Evidence Table

| Decision-bearing claim | Primary support | Remaining project evidence |
|---|---|---|
| Staged compatible transfer is a plausible direction | [1][2][8] | Original-engine compact donor-history recovery |
| Redundant expansion needs a preservation identity | [3][4][6], algebra above | Same-operator shard coverage, then ternary/operator residual |
| Large-n selection has concrete structured methods | [5][11][12] | Useful IDs and normalized mass; CPU/DRAM timing |
| Small fixed head and precision need separate diagnosis | [8][9][10] | Attribution, measured distribution approximation and chat tasks |

## Limitations

This is a focused addendum, not an exhaustive 2026 literature census. It uses
primary texts and abstracts, with no independent paper reproduction or
cross-paper benchmark normalization. Some references were already known;
their repeated citation does not add a new local experiment. The evidence
ledger labels each paper-specific claim and its locator.

No cited work establishes our complete same-artifact contract: useful chatbot,
original CPU LUT/ternary/SSM/SWA, large useful n, normalized structured routing,
physical DRAM, and at least fifty accepted IDs per second. The audit and
proposed attribution have distinct status; literature research consumes none
of the reserved model queries and cannot substitute for their results.

## Recommendations

Complete stored adjudication, retaining the parent's resource failure. Next,
use the already [proposed target/readout attribution](ORIGINAL_LATENT_READOUT_ATTRIBUTION_NEXT_20261010.md)
to separate target/decoder consistency from internal-history recovery. Its
interpretation must remain prospective.

Then prepare a same-operator FFN coverage diagnosis, using saved inputs if
available, before any overlap/duplication training. Compare the exact complete
shard sum, selected fragments, corrected coverage and an overlap design at
matched active width. Keep operator and precision changes separately visible;
the algebra above supplies conditions and error measures, not empirical
thresholds to invent after seeing results.

For useful large-n capacity, product-key selection is the strongest concrete
candidate identified here. A future bounded protocol should compare its
complete Cartesian oracle with its structured implementation, testing IDs,
mass and deterministic ties, then learned transfer quality and DRAM cost.
Copied synthetic banks can validate mechanics but cannot qualify usefulness.
No new timed benchmark is authorized by this note while the stored audit runs.

## Methodology

Date verified with local PowerShell on 10 October 2026. Batched web searches
covered mixer conversion, granular upcycling, function-preserving expansion,
product-key retrieval, output rank, width compression and ternary adaptation.
Search CLI was unavailable; the built-in web tool was used. Secondary search
hits were excluded from support. Thirteen primary papers were inspected;
version-specific URLs, short evidence spans and claim boundaries are persisted
in [sources](literature_followup_20261010/sources.jsonl),
[evidence](literature_followup_20261010/evidence.jsonl),
[claims](literature_followup_20261010/claims.jsonl) and the
[manifest](literature_followup_20261010/run_manifest.json).

Per the user's limited research request, the repository note and provenance
ledger are the deliverable; no general survey or standalone PDF was produced.
The algebra and project recommendations are explicitly our deductions. Paper
performance is attributed to its authors and never retroactively credited to
our local candidate. Structure and citation validators are run separately;
automated URL accessibility is not independent scientific verification.

## Bibliography

[1] Bick et al. (2024). "Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models". arXiv, v2 revised 2025. https://arxiv.org/html/2408.10189v2 (Retrieved: 2026-10-10).

[2] Wang et al. (2024). "The Mamba in the Llama: Distilling and Accelerating Hybrid Models". arXiv, v4 revised 2025. https://arxiv.org/html/2408.15237v4 (Retrieved: 2026-10-10).

[3] Komatsuzaki et al. (2022). "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints". arXiv, v2 revised 2023. https://arxiv.org/html/2212.05055v2 (Retrieved: 2026-10-10).

[4] He et al. (2024). "Upcycling Large Language Models into Mixture of Experts". arXiv, v2 revised 2025. https://arxiv.org/html/2410.07524v2 (Retrieved: 2026-10-10).

[5] Xu Owen He (2024). "Mixture of A Million Experts". arXiv v1. https://arxiv.org/html/2407.04153v1 (Retrieved: 2026-10-10).

[6] Chen, Goodfellow and Shlens (2015). "Net2Net: Accelerating Learning via Knowledge Transfer". arXiv, v4 revised 2016. https://arxiv.org/html/1511.05641v4 (Retrieved: 2026-10-10).

[7] Dai et al. (2024). "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models". arXiv v1. https://arxiv.org/html/2401.06066v1 (Retrieved: 2026-10-10).

[8] Ashkboos et al. (2024). "SliceGPT: Compress Large Language Models by Deleting Rows and Columns". arXiv v2. https://arxiv.org/html/2401.15024v2 (Retrieved: 2026-10-10).

[9] Yang et al. (2017). "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model". arXiv, v3 revised 2018. https://arxiv.org/html/1711.03953v3 (Retrieved: 2026-10-10).

[10] Wu et al. (2025). "BitNet Distillation". arXiv v1. https://arxiv.org/html/2510.13998v1 (Retrieved: 2026-10-10).

[11] Lample et al. (2019). "Large Memory Layers with Product Keys". arXiv v1. https://arxiv.org/html/1907.05242v1 (Retrieved: 2026-10-10).

[12] Berges et al. (2024). "Memory Layers at Scale". arXiv v2. https://arxiv.org/html/2412.09764v2 (Retrieved: 2026-10-10).

[13] Zhu et al. (2024). "LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training". arXiv v1. https://arxiv.org/html/2406.16554v1 (Retrieved: 2026-10-10).
