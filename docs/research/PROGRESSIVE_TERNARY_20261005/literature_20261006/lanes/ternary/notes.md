# Pretrained ternary conversion: primary evidence lane

Cutoff: 6 October 2026. Thirteen primary paper clusters have full text inspected;
their versions, budgets, representation details, direct locators and limitations
are recorded in `sources.jsonl` and `evidence.jsonl`. Paper, project and repository
versions count as one source cluster. CAT-Q and ScaleQ share a method and author
lineage; their agreement is not an independent reproduction. No source in this
lane evaluates the original `google/switch-base-128` checkpoint or establishes
the project's joint expert fidelity, complete storage and native CPU goal.

## What changes the research decision

Genuine training-free ternary conversion exists: [PT²-LLM](https://arxiv.org/html/2510.03267v2)
uses alternating discrete assignments, closed-form grid updates and
activation-aware output reconstruction. It is a closer scientific baseline for
the proposed approach than unstructured random weight replacement. It is still
far from preserving float quality on several checkpoints, especially LLaMA3-8B.

[CAT-Q](https://arxiv.org/html/2606.26650v1) also converts pretrained weights,
including modern MoE models, but employs learned calibration and a soft-to-hard
transition. Its use of the label PTQ must not obscure AdamW and 60 epochs. The
controlled question worth testing is whether smooth staged reconstruction of a
whole expert FFN, rather than independent matrix reconstruction, generalizes
under a bounded budget. A reduced local adaptation would be a new method screen,
not a faithful reproduction of its 8-A100 experiments.

[TWLA](https://arxiv.org/html/2606.13054v2) adds learned structured rotations and
calibration geometry. Such equivalent-coordinate changes are promising for
linear layers, but adaptation around a ReLU expert must explicitly establish
which transforms preserve the original nonlinear function. [ScaleQ/AYOT](https://arxiv.org/html/2608.01078v1)
shows why preserving generic language metrics does not establish preservation of
reasoning trajectories. For this project it strengthens the requirement to
declare corpus coverage and the limited scope of existing masked-span data.

## Representation and accounting traps

[PTQTP](https://arxiv.org/html/2509.16989v1) is an interesting deterministic
discrete search lead: each weight chooses among nine combinations in two ternary
planes, alternating with closed-form ridge scales. It must be classified as a
two-plane approximation, not an all-ternary single code matrix. Its own Appendix
A.3 states `nd/2` bytes for two 2-bit planes plus `4n` bytes for two FP16 scale
vectors. For `n=1024,d=4096`, this arithmetic yields 2,101,248 bytes, about
2.004 MiB, not the reported 1.004 MB example. Independently packed artifact sizes
are necessary before using its nominal 1.58-bit table labels. The strongest
reported quality is not evidence at the same capacity as a single ternary plane.

[PTQ1.61](https://arxiv.org/html/2502.13179v2) is a useful storage-policy baseline,
but is binary plus salient 4-bit channels, rather than ternary. [TTQ](https://arxiv.org/pdf/1612.01064v3)
uses independent positive and negative scales; [TernaryLLM](https://arxiv.org/pdf/2406.07177v1)
and PT² use shifts; [Tequila](https://arxiv.org/html/2509.23809v2) exports an
additional precomputed output bias. These have different deployed functions and
costs from a symmetric `alpha * {-1,0,+1}` matrix. Their extra parameters are
legitimate method components but must be counted explicitly.

## Budgets and limits

Unique calibration corpus size, processed token exposures, optimizer updates,
teacher-generation cost and complete child wall time are different quantities.
Derived from CAT-Q's stated settings, 512 × 2048 × 60 is 62,914,560 processed
token exposures per calibration window before accounting for sliding overlap.
For ScaleQ, the same calculation is approximately 240 million for its default
4 million-token corpus and 60 epochs. These calculations are arithmetic
inferences, not claimed exact implementation counters. Checkpoint pretraining
is inherited by all conversion methods and is absent from added conversion costs.

[Continual quantization-aware pretraining](https://arxiv.org/html/2502.11895v1)
tests controlled partial OLMo pretraining, with 41,943,040,000 total tokens derived
from the stated batch, length and steps. [Tequila](https://arxiv.org/html/2509.23809v2)
uses 10 billion conversion-training tokens. These budgets establish useful
mechanisms and attainable nonzero recovery, but do not support a claim that the
same recovery is attainable with a tiny no-optimizer local calibration run.

[INQ](https://arxiv.org/pdf/1702.03044v2) is the closest historical precedent for
progressive weight batches: freeze the quantized subset and retrain the remaining
float weights. Its random-partition ablation is worse than informed partition;
its actual ternary result loses quality despite the paper's lossless title.
[BitNet pretraining](https://www.jmlr.org/papers/volume26/24-2050/24-2050.pdf)
establishes useful ternary models trained for their representation, not conversion
of an arbitrary independent float teacher.

## Hypotheses and scoped rejection

The literature warrants prospectively bounded tests of activation-aware
asymmetric grid refinement, deterministic assignment/coordinate optimization,
joint FFN reconstruction and soft-to-hard calibration. A separate two-plane
artifact could test whether extra discrete capacity is the binding constraint
while remaining under the actual storage cap. This would be an explicit new
representation arm. Successful local reconstruction would still require
whole-model task evaluation and uncontended native measurements before promotion.

A negative closure may be justified for a named checkpoint, corpus coverage,
method family, quality threshold, actual storage limit and available compute
budget once qualified controls and bounded relevant tests fail. The reviewed
papers contain substantial negative evidence against automatic float parity;
they contain no theorem or exhaustive search establishing universal impossibility.
Do not conflate inability to admit enough naturally routed examples with a failed
fit. Preserve the scope difference between the rejected final-decoder admission
gate and the new final-encoder expert study.

## Retrieval limitations

The September 2026 [Qwen3-4B systems case](https://arxiv.org/html/2609.01962v1)
adds particularly relevant negative evidence: a TWLA-derived weight-only
conversion retained useful capability while losing task accuracy, and its compact
artifact did not yet establish an inference advantage. Its 64 × 2048 calibration
uses a previously rotated checkpoint, so rotation cost is inherited. Its stated
1.641 bits is a symbol-rate estimate for selected linear codes; the complete
checkpoint's 3.96/8.29 ratio is approximately 47.77%, above this project's 35%
threshold. That ratio is an arithmetic inference from the reported rounded
sizes. The slow packed GEMV is one GPU shape and software path, with no native
CPU or end-to-end packed task benchmark. Raw project artifacts were not acquired.

No paper was reproduced, no model or calibration data were generated, and no GPU,
Kaggle, private-service operation or scientific run was used by this lane.
Primary HTML/PDF inspection does not qualify released code or packed exports.
OpenReview's INQ PDF returned a browser challenge; arXiv's full PDF was used.
The TernaryLLM v2 HTML URL failed because the registry lists only v1; its primary
v1 PDF was inspected. PT-BitNet's publisher abstract was found, but full methods
and budget were unavailable in this retrieval, so it is a coverage lead rather
than one of the thirteen reviewed full-text paper records. Those thirteen include
twelve conversion, transition, or alternative-precision studies and a distinct
historical BitNet pretraining control, which is not conversion evidence.
Supplementary material that was
not present in the extracted primary text is explicitly unresolved.
