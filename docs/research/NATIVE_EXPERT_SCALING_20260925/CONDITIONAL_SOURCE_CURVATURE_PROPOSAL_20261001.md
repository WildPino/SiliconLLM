# Proposal: full-input affine response plus conditional source nonlinearity

**Status: proposed; native combined operator and fitted banks do not exist.**
METH-227 shows a24.3% E16->E160 SSE improvement with distinct source-derived
functions, but absolute error fails. METH-228 excludes bias-only repair of
the fixed products: even a validation-label oracle leaves39.48% error.
The next mechanism must represent a non-constant nonlinear response.

## Function and inherited knowledge

One selected function would execute

`W_affine @ x + B @ (silu(G_selected @ x) * (U_selected @ x)) + bias`.

G/U rows are complete original donor input directions, not PCA64-only
features. A shared selected G/U set per parent avoids copying the same
input weights into each child. Both E16/E160 use one selected affine/
nonlinear output function, the same hidden width and parent feature
set. Child W/B/bias are independently learned with a parent prior.
No second active expert, stand-alone trained common or full donor FFN
is needed during inference. The nonlinear branch is necessary capacity,
not a correction of only the cell-mean offset ruled out by METH-228.

For a source-unit set S and center c, initialize B from original D[:,S].
Let h_S(x)=silu(G_S@x)*(U_S@x). Set

`W_affine = J_full(c) - B @ J_hS(c)`

and bias=`f(c)-W_affine@c-B@h_S(c)`, recomputed after coefficient rounding.
Before rounding this reproduces donor value and derivative at c, with
**exact selected-unit nonlinearity** away from c. The unselected response
is still approximated and must be learned/validated. Source-value/gradient
identities and actual native numerical parity are required.

## Initial width and whole-path price hypothesis

Initial width **2048**:42.1% of this donor's4864 source units. This is a
larger nonlinear response than the failed width768 stand-alone student;
the representation also has the full-input affine response and conditional
readouts. It is chosen from the whole active ledger, not a quality result.

Per selected layer, BF16 affine896x896 plus gate/up2048x896 and
down896x2048, with FP32 output bias, addresses12,619,264bytes. Across24
layers:302,862,336bytes. Full-input16-parent/10-child FP32 route keys add
2,236,416 selected bytes. Replacing the old Q8 FFNs and E1280 path in
METH-211's ideal ledger would give

`559,794,176 - 323,592,192 - 11,280,384 + 302,862,336 + 2,236,416 = 530,020,352`.

This is addressed-weight arithmetic, not actual DRAM, full throughput,
native attention/head parity, preserved quality or a completed artifact.
The combined operator must pass the existing <=10ms component budget
before any coefficient training. Previous smaller/separate operator
passes do not prove combined timing or cancellation stability.

With16 shared parent G/U sets, the hypothetical all24-layer resident
function payload is4.846GB at E16 and23.091GB at E160,excluding other
core/route storage. At E1600 it is205.541GB. Stored child capacity grows
with n while active function width stays fixed; useful capacity and CPU
large-bank selection remain measurements. No such full bank is made.

## Proposed conditional learning, to freeze before fitting

Reuse METH-227 full-input keys/fit assignments and the existing captured
actual source targets. Keep its E16/E160/rotated controls; do not move
validation rows or call consumed sources independent quality.

For each parent, select2048 original source-unit rows using **fit-only**
centered variance of its per-unit nonlinear Taylor remainder times the
original down-column squared norm. This score ignores cross-unit
cancellation and is a hypothesis, not an optimality/energy bound.
Freeze lowest-original-ID tie policy before selection. No validation
feature/width choice. Native shape qualification first uses predetermined
first2048 source rows purely as a component fixture, not quality selection.

Learn the complete output readout over features `[x896,h_S2048]`, with
the source-derived rounded initial affine/B prior. An initial proposal
is centered FP64 source-anchored ridge, diagonal parent fit-feature
variance penalty with declared floor,prior strength1024 sample equivalents;
shrink output-mean corrections with the same strength. Parent functions
fit first; each child fits a deviation anchored to its stored effective
parent coefficients in that parent's same feature coordinates. Fold the
deviation into one BF16 affine/B matrix pair plus FP32 bias, so inference
does not read parent and child output weights simultaneously. Exact
regularizer/floor/primal-or-dual solve, intercept and serialization must
be frozen in the later protocol; no bank is yet fitted.

This is a different nonlinear feature representation and source-anchored
function-learning mechanism from both METH-222 and METH-227. It can still
fail from missing nonlinear directions, poor conditioning, overfitting,
shrinkage or unseen input coverage. Complete-function NMSE<=.01,10% gains
over E16/rotated and paired-window P05>0 stay unchanged. A pass would still
require routed-bank C/DRAM/LUT checks, multi-layer causal composition,
independent LLM prediction/generation/tasks and >=50 accepted tok/s on
one artifact, then an actual second family/approximately10B transfer.

## Next exact action

Freeze and implement **METH-229 combined native qualification**, before
row selection or fitting. Use24 distinct source-layer fixtures at existing
METH-125 token0 inputs; source value/derivative identity, BF16 readback,
FP32 oracle and actual combined CPU timing. Declare local resource/output
budget and stop. No T4 or donor-port resume is needed. A native pass
licenses freezing the fit-only nonlinear selection/parent-child method;
an operator failure stops this fixed width/kernel before training.
