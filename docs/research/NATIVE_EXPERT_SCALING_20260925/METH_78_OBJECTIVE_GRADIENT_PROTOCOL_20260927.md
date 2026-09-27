# METH-78: frozen objective-gradient attribution at E128 and E1280

**Question.** METH-75's longer continuation loses donor-position top-1
despite wider route coverage and better raw BPB. Loss values alone do not
show which term moves the learned factors or product keys. Measure the
separate local gradients of CE, weighted donor KL, and weighted axis balance
at the frozen METH-71 update-64 checkpoints. This is a mechanism diagnostic,
not a training or quality gate.

**Fixed inputs.** Load the SHA-bound METH-71 update-64 E128 (8×16) and E1280
(32×40) checkpoints without optimizer steps. Use the bound
Qwen2.5-0.5B-Instruct donor and the existing raw training token archive.
For each arm, use raw row 123 at offset 0, truncated to the METH-71 raw
sequence length, and the first METH-43 chat training prompt plus teacher
continuation. No development or external quality set is queried. These
inputs are diagnostic training-source examples; prior exposure is allowed.

**Measurement.** Recompute donor and student logits under the exact
METH-71 objective. On each example separately, backpropagate (1) CE,
(2) KL times the METH-71 weight, and (3) the sum of axis balance times
0.02. Measure the Euclidean gradient norm for all A factors, B factors,
and product-key router parameters separately, plus the all-parameter norm.
The common four-microbatch average is omitted because it scales every
term equally. Zero a group's gradients before each term; do not step an
optimizer. Report each loss value and norm, and ratios of balance and KL
gradient norms to CE norms. An absent parameter gradient counts as zero.

**Decision boundary.** The readout may motivate one future fixed training
rule, but does not identify causation, justify tuning on viewed development
prompts, or reverse METH-72/74/75 failures. Record any OOM/nonfinite
failure. Local RTX 3060 only; stop at 10.5 GiB peak allocated GPU,
20 GiB process RSS, or 12 minutes per arm. No T4.
