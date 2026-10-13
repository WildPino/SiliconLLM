# Selected next: preserve donor output geometry with multi-plane ternary LUT

13 October2026. **PRECISION PREFLIGHT COMPLETE / FULL INDEPENDENT AUDITPASS.**
[Qualified result](TERNARY_READOUT_PRECISION_RESULT_20261013.md),
[closure](ternary_readout_precision_session_closure_20261013.json).
FIT-selected p6 passes every extra1%/donor local gate on FIT+DEV/all144 cases,
all415,586,304 BF16 words independently bit-reproduced. Native LUT is NOT
implemented/timed;next [actual C LUT execution](TERNARY_READOUT_NATIVE_NEXT_20261013.md)
SELECTED/PROPOSED/UNIMPLEMENTED/UNBOUND. Original proposal retained below.

Prior globally-linear
rank envelopes and observable-route local affine banks show large response
losses; fixed-bank exhaustive coverage is now fully independently qualified.
Main expensive readout function should preserve its full2048-coordinate
response space while testing a precision/storage/primitive change. Do not
run another unchanged global/local small-rank output fit as the main route.

## Alignment with engine.c and complete pipeline

Original trained3072-channel/state96 scan is locally qualified/fast (~59us
separate native timing), normalization/output dominates (~1175us) and input
generation separately~2515us. These are separate workloads, not end-to-end
chatbot rates. Source-informed shared core needs cheap actual functions;
discarding most held-out readout response makes that core unusable. Reuse
original engine ternary/LUT primitives while retaining donor output geometry.
This is enabling operator work for the shared core, not proof of conditional
knowledge capacity or a generic dense donor runtime completing the goal.

Source CHATBOT->compact reusable core+selectively consulted useful ternary
functions->engine.c still needs generator/FFN/attention/head/tokenizer/causal
composition, OWN-history quality, RAM-driven n/multiple donor scales and SAME
useful>=50 tokens/s. No local packed matrix count may substitute for them.
This experiment concerns O only; it does not solve expert transfer/routing.

## Algebra: precision reduction without assuming low response rank

Original O2048x3072 is BF16 coefficients decoded to a real/F64 reference.
For fixed plane counts p=4/5/6, Q_p=(3^p-1)/2 (40/121/364). For each output
row, s_o=max_i|O_oi|/Q_p; zero row is explicitly zero, not epsilon-scaled.
q_oi=round-to-nearest-even(O_oi/s_o), clipped to[-Q_p,Q_p]. This scale depends
only on source coefficients, never on DEV or FIT response choice.

Balanced ternary digits t_doi in{-1,0,1} satisfy the EXACT integer identity

    q_oi = sum_(d=0..p-1) 3^d t_doi,
    O_hat_oi = s_o q_oi.

The represented matrix is full size; this imposes no rank-r bottleneck,
although numerical rank preservation still needs measurement. Extra ternary
planes encode precision of the same donor function, not extra intelligence.
Coefficient error bound |O_oi-O_hat_oi|<=s_o/2 when finite scales/rounding
qualify; corresponding output bound per row <=s_o*||Z||_1/2. The bound may
be conservative; measure actual inputs, cancellation and per-case source error.

Source coefficient digit reconstruction/rounding/scales must be independently
verified for every coordinate and every plane count. No silent zero/tiny-scale
fallback; fixed explicit numerical policy before implementation/launch.

## Exact ternary pattern packing and LUT function

Current engine.c (benchmarks/phase60/engine.c:244 onward) uses TWO-trit codes,
int8 activation tables/16-entry pshufb, int32 accumulators, with byte/nibble
packing variants. Its existing literal primitive is not the float243 table
below. Five-coordinate floating tables would be an explicitly NEW backend;
neither existing bit-exact integer qualification nor old speeds transfer.
The current source comments also flag an earlier ternary-byte quality failure;
inspect underlying evidence before choosing activation precision. Source-normalized
BF16 inputs must first be preserved for weight-only numerical qualification.
Any int8 activation variant needs a separate additional error gate.

Proposed groups of FIVE input coordinates, padded final group with zeros.
Each plane/group/output has pattern index sum_k(t_k+1)*3^k in[0,242], uint8.
Given actual own-normalized BF16 Z decoded to F32/F64 reference, a group LUT
lists all243 values sum_k t_k Z_k. One common input LUT may be reused across
all planes/output rows. Then accumulate group lookups for each plane, combine
3^d and row scale explicitly. Mathematical equality to O_hat Z in exact real
arithmetic; new F32 contractions/LUT grouping are not original AVX FMA bits.
Freeze precise accumulation/rounding before qualification, include final
F32->BF16 boundary and native's original mixer-output comparison scope.

3072 inputs need615 groups,2048 outputs. Packed bytes at p4/5/6:
5,038,080 /6,297,600 /7,557,120 plus row scales/metadata/padding; original
BF16 O12,582,912B/nativeF32 O25,165,824B. A full F32 group LUT is615*243*4=
597,780B, substantially larger than the original small-expert LUT. Blocked
table reuse, layout, cache and actual DRAM must be measured. Multiple planes
increase lookup/accumulation work; smaller bytes do not imply faster kernels.
This does not establish constant per-token cost under expert-n scaling.

Reuse/inspect existing meth398 additive integer LUT and meth400 block-local
integer LUT implementations and their actual evidence before coding native
layouts. They reference a distinct Q6 source/operator, so source-specific
parameter/activation identities are not Falcon BF16 proof. Do not repeat an
unchanged old cost campaign. Compare the actual current two-trit/nibble path
and proposed full-rank representation with explicitly qualified precision.

## Fixed preflight and next decision

Implement all three plane counts as diagnostic arms before observing their
responses, at sites0/12/23 and all48 qualified native input/output histories.
No source/GPU/T4/new teacher/RESERVED/optimizer/DEV selection. FIT may determine
a later explicitly frozen candidate rule, with all DEV controls retained.
Use original extra-output each-case1% raw/BF16 and donor local budgets; include
whole/centered/label/domain/worst/cancellation/digit counts. No budget changes.

First qualify numerical packed/digit/dense reconstruction and response
accuracy with full independent audits. Then actual C LUT kernel and primitive
comparison, physical DRAM/batch1 cost on SAME accurate artifact. If no p arm
conserves quality, isolate precision/cancellation/grouping and fix that defect;
if numerical preservation succeeds but kernel is slow, report exact traffic/
lookup bottleneck and change layout/precision allocation. Do not inherit
old synthetic engine speeds or declare50 from arithmetic counts.

Proposed CPU preflight900s/audit1200s/OS6GiB/new orderedE<=1GiB; source/runtime/
storage/rounding/selection caps not bound yet. No hours or T4 allocation.
Initial-layer token-only generator compilation remains an exact-input
alternative. Conditional knowledge transfer needs a separate source-informed
bank/training strategy; the rejected affine banks do not settle its feasibility.
