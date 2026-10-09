# Engine-first chatbot pipeline: priority and algebraic gaps

9 October 2026. Goal ACTIVE/INCOMPLETE. Operational priority clarification;
reuses the [8 October review](CHATBOT_ENGINE_TARGET_REASSESSMENT_20261008.md),
[method](METHOD.md) and completed evidence. No new model observation in this memo.

## Destination and present position

The desired artifact is a useful pretrained chatbot converted into compact
SSM/SWA control plus selectively read ternary functions executed by the original
LUT machinery. Offline adaptation may take a month or more when its feasibility
and recovery are measured. Generic donor execution does not establish that result.

The current Falcon candidate does implement this architectural direction:
10 recurrent blocks, two SWA blocks, packed ternary experts, original LUT and
integer activation arithmetic. It changes the recurrence geometry, uses signed
SiLU, doubles the original default residual width/layer count and preserves the
chatbot vocabulary. This is a target extension requiring its own quality and
complete speed evidence. It is not identical to the original small model.

| Evidence | Established | Remaining question |
|---|---|---|
| Original trained E32, about 701.7 token/s | Small-model learned quality/parity and native speed; pool fits cache | Useful larger stored capacity with actual DRAM and full router cost |
| Falcon packed C candidate | Original matrix/LUT bodies; 425.21 MB packed artifact; no expert master/reference copies; evolving state | Useful chatbot transfer; fresh generation; accepted rate |
| Old whole C comparison | 32/32 learner greedy IDs agree; strict numerical criterion fails 19/32 rows | Native forward qualification; no source-quality admission follows |
| Balanced recovery | Actual model/Adam restored; completed updates learn from eight domains | A fixed-checkpoint final evaluation is still needed; online training losses are insufficient |

The existing 1e-4 numerical gate and its failure remain recorded. Numerical
transport, donor-relative quality and accepted speed answer separate questions.
Further component diagnosis must identify a variable capable of changing a
pipeline decision; it must not become an indefinitely expanding workstream.

## 1. Stored capacity can grow; active work has explicit conditions

For n experts per layer, width h, selected count k, residual width D and L sites:

\[
P_{bank}=3LDhn,\qquad P_{selected}=3LDhk.
\]

At fixed k,h,D,L, selected matrix work does not grow with n. Complete time is

\[
t=t_{core}+t_{state}+t_{route}(n)+t_{selected}(k,h)
  +t_{head}(V,D)+t_{interface}.
\]

An unrestricted flat linear router requires LDn score products and reading its
weights. Constant k alone does not remove this term. A hierarchical selector
can have logarithmic depth for a fixed branching factor; learned restrictions,
shortlist quality and normalized output mass must then be checked. It does not
automatically reproduce arbitrary flat-router winners. For selected-softmax,
the denominator is over the selected set; a global-softmax denominator is a
different contract. Keep this distinction explicit in each routing variant.

More physical RAM supplies storage. Useful independent functions, inexpensive
addressing and acceptable selected DRAM traffic are additional requirements.

## 2. Dense sums and selected mixtures are different algebra

At one common operand x, splitting the hidden rows of a SwiGLU FFN is exact:

\[
F(x)=\sum_j D_j[\operatorname{SiLU}(G_jx)\odot U_jx]
     =\sum_j f_j(x).
\]

The current initialization partitions two source FFNs into 72 banks per target
site. Its execution instead computes

\[
\hat F(x)=\sum_{j\in S(x),\ |S|=8} \alpha_j(x)\tilde f_j(x),
\quad \alpha_j\ge0,\quad \sum_{j\in S}\alpha_j=1.
\]

Omitting 64 groups and normalizing eight groups changes both content and
amplitude. Scaling by 72 is unbiased only under appropriate sampling/weighting
assumptions, and even then generally only in expectation. Deterministic top-k
does not provide those assumptions. Copying source rows does not solve this.
Moreover, the two source FFNs normally see different intermediate operands:
placing their rows in one site is not an exact composition of the two blocks.

The next algebraic experiment should isolate this mechanism on common operands:
all-group sum, selected sum, normalized mixture and a declared shared-plus-private
construction, with the same precision/operands. Measure residual energy,
covariance/cancellation, amplitude and routing margins. Separate projection,
ternary quantization and changed recurrence from this comparison. Register one
bounded protocol before new observations; reuse actual weights/corpus and never
replay completed whole runs solely to obtain more diagnostics.

## 3. Redundancy is a plausible construction with an explicit identity

Suppose redundant expert e contains a set of source atoms J_e and output
coefficients c_ej. Then

\[
g_e(x)=\sum_{j\in J_e}c_{ej}f_j(x),\qquad
\sum_{e\in S(x)}\alpha_e g_e(x)
=\sum_j\left[\sum_{e\in S(x):j\in J_e}\alpha_ec_{ej}\right]f_j(x).
\]

A sufficient exact reconstruction condition is that every bracket equals one
for every source atom with nonzero contribution. Ordinary duplicate atoms in
normalized mixtures do not guarantee this. Correlated atoms can admit other
representations, but those must be derived or learned and measured.

Adding overlapping branches can enlarge stored capacity while leaving each
selected branch small. It succeeds when selected branches jointly cover the
necessary response, possibly with a compact common term, or recover an acceptable
approximation. Increasing branch width h also raises selected work linearly.
Retain total unique atoms, redundant copies, active products/bytes and actual
quality separately. This turns the human's tree intuition into a testable
coverage/representation problem rather than assuming duplicated storage adds
knowledge.

## 4. More experts cannot repair every information bottleneck

For a deterministic representation z(h), if two histories produce the same
accessible state z, any downstream selector/experts receiving only z produce
the same next-token distribution. Enlarging n cannot distinguish that pair.
This statement applies to all information accessible to the downstream stage,
including recurrent state and SWA memory, not merely one residual vector.

For two equally weighted donor distributions p and r at such a collision, the
smallest achievable mean forward KL over one shared output q is

\[
\min_q\tfrac12[KL(p\|q)+KL(r\|q)]
=JS(p,r),\qquad q=(p+r)/2.
\]

This is an algebraic limit, not evidence that measured collisions exist. The
2048-to-512 source initialization retains 47.1991% of the balanced embedding/
head Gram trace. That is matrix energy, not a fraction of knowledge or a proven
quality ceiling. Width/depth reduction, removal of parallel attention, norms,
SSM channel selection and linear readout are distinct possible bottlenecks.

Consequently stage compression and learning: first establish a recoverable
functional decomposition, then change recurrence/precision/width under the full
budget. High precision intermediates are legitimate conversion scaffolding.
They need not be the deployment artifact. Do not infer a general impossibility
from the current aggressively compressed initialization.

## 5. Chatbot readout materially changes the original cost

Current dimensions are D512/L12/n72/k8/h128/V65537. Shape deductions:

| Term | Matrix products/token | Logical coefficient bytes where specified |
|---|---:|---:|
| Full F32 readout VD | 33,554,944 | 134,219,776 |
| Selected ternary experts 3LDhk | 18,874,368 | 9,437,184 pair-code bytes, excluding scales/padding |
| Flat router LDn | 442,368 | 1,769,472 F32 weight bytes, excluding bias |
| Total recorded matrix products | 69,632,512 | Complete physical traffic remains unmeasured |

The head accounts for about 48.2% of those products. At 50 accepted IDs/s the
whole budget is 20 ms per accepted ID. If the head and selected codes stream
from DRAM on every token, those two terms alone require about 7.18 GB/s; this is
a conditional traffic deduction, not measured bandwidth or achieved speed.
Caching, control matrices, state, dispatch and actual latency must be measured.
Readout factorization or shortlist schemes change representation/selection and
need their own full-output quality evidence. Preserving the original tokenizer
makes tokens/s comparisons interpretable.

## Operational order

1. Close the already running bounded recovery at its original cap. Preserve
   actual checkpoint/model/moments/RNG and the first fault. Independently adopt
   saved outputs/state once. An interrupted run is not eligible full recovery.
2. Assess the retained checkpoint under a separately frozen complete evaluation
   if no final observations exist. Decide whether recovery warrants continuation
   from that exact state or whether initialization must change. Reusing a state
   must not replay completed updates; missing historical support stays missing.
3. Resolve the sum/mixture and information-loss questions with one bounded
   controlled experiment capable of choosing a staged conversion recipe.
4. Qualify an actual useful packed candidate in C, then fresh own-history/chat
   behavior and >=50 accepted batch1 IDs/s on that same artifact. Charge full
   head, router, state, chat and DRAM. Prepared cohort tools are scaffolding,
   not validated results.
5. Expand genuinely useful n at fixed active geometry, with structured routing
   IDs/mass and removal/selection interventions; then another family and scale.

Before T4 work, communicate a concrete recipe, data/teacher coverage, measured
memory/optimizer feasibility, conversion throughput, GPU-hour budget and stops.
Month-plus offline time is permitted; the present 128 FIT templates do not supply
the necessary coverage or a measured long-training budget. No T4 allocation or
new heavy run follows solely from this priority memo.
