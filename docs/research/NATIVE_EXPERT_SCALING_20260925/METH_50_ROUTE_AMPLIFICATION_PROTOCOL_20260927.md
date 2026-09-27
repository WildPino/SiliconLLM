# METH-50: separate factor error from downstream route changes

**Uncertainty.** METH-48 and METH-49 fail prompt top-1 retention at E128
despite small factor reconstruction and document BPB error. Later layer
routers see hidden states perturbed by earlier factor outputs; route changes
may amplify small errors. Measure that mechanism before choosing a CPU
index, hierarchy or joint quantization-aware training recipe. METH-46
measured load on a different continuation; measure the retained METH-47
candidate here. This is a diagnostic on previously viewed inputs and
cannot promote the checkpoint or an int8 format.

Bind METH-47 update-512 checkpoint SHA-256
`0c6f09562efff7e78921036fe6ab030d7377150fe9afbe06b3deb1c029e0aa59`,
Instruct donor weights SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`,
METH-45 external manifest SHA-256
`15a68db9a43a8066e70b1b261f819cab3d6779d8b52633bcce7c725d8d54acfb`,
and METH-49 row-int8 artifact SHA-256
`f3039decaec0600ccae9f93b6c11cb063298a53b469b39e2cbcdba92aceef196`.
Use its exact saved top-1 count, 2,002/2,091, as a reproduction check.

On the 12 frozen prompts, run three otherwise identical BF16 donor passes:
original fp32 expert factors; METH-49 reconstructed factors with ordinary
fine routing; and the same reconstructed factors with the **original
per-layer/per-position top-4 expert IDs forced**. Recompute gate softmax
from each arm's own fine-router scores for the chosen IDs, so the forced
arm changes only the IDs, not the input hidden state or router weights.
First force the original IDs with original factors and require exact top-1
and logits parity to validate the intervention. Capture original/ordinary
int8 selected IDs at each layer and position; report pooled and per-layer
top-4 ID overlap, exact set agreement, first layer with changed set,
original fourth-versus-fifth score margins, and original expert load
concentration. Verify prompt top-1 original-versus-ordinary int8 matches
METH-49's saved 2,002/2,091 count.

**Decision rule.** If forced IDs restore ≥99% original prompt top-1 while
ordinary int8 remains below 99%, downstream route changes are the
dominant observed failure on this set. If forced IDs remain below 99%,
factor arithmetic also fails the ranking gate even when expert identity
is held fixed. Report both effects and do not interpret forcing as a
deployable router. At E128 this only diagnoses precision sensitivity; no
sublinear CPU cost or E>128 learned-route fidelity is inferred.

Local RTX 3060 only; ≤10 minutes, ≤10.5 GiB GPU allocation and ≤20 GiB
RSS, no T4. Stop on source/reload/parity/resource failure before reporting
a causal conclusion. Store per-prompt and per-layer counts plus route stream
hashes, without reselecting inputs or thresholds after observing results.
