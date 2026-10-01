# Proposal: conditional full-input donor tangents

**Status: proposed, not an executed or accepted conversion method.**
METH-222's shared affine/PCA64 corrections fail function accuracy and
useful E16->E160. METH-225's trained nonlinear common also fails its
fixed intermediate accuracy screen. Both recipes stop. The next question
is whether directly compiling distinct local functions from the donor
can put useful capacity in the conditional bank, with fixed active cost.

## Changed representation and source of knowledge

For the actual donor SwiGLU function f(x), take a fit-input centroid c
and its full input Jacobian J(c). A leaf implements

`f(c) + J(c) @ (x - c)`.

For `g=G@c`, `u=U@c`, `s=sigmoid(g)`, the analytic Jacobian is

`D @ (diag(u * (s + g*s*(1-s))) @ G + diag(silu(g)) @ U)`.

G/U/D are all original source matrices. Decode source BF16 weights and
use declared FP32 arithmetic. Store each full896x896 Jacobian BF16 and
its896-element bias FP32. Set stored bias to `f(c)-J_stored@c` with the
same declared arithmetic. Export/read every coefficient before scoring.
The source-derivative implementation must agree with automatic
differentiation on predetermined source inputs before trusting an export.

This is a first-order approximation to a nonlinear donor, not an exact
transfer. It eliminates the previous PCA64 restriction on each expert's
function and the need to learn a stand-alone compact common first.
Coefficients come from the pretrained response and derivative, including
in small-support cells; they are not fitted independent residual matrices
or duplicated experts. More cells might shrink approximation neighborhoods,
but routing, uncovered directions and curvature can still defeat quality.
This hypothesis requires a measured comparison.

## Capacity, route and cost

One selected affine function has1,605,632 BF16 weight bytes and3,584
FP32 bias bytes per layer. Across24 layers it addresses38,621,184bytes,
before routing, other core organs, actual traffic and temporary buffers.
The ideal bank payload across24 layers is617,938,944bytes at E16,
6,179,389,440 at E160,61,793,894,400 at E1600. These are shape arithmetic;
no24-layer bank is constructed or accepted. RAM bounds residency; useful
capacity also depends on source coverage and measured approximation error.

Use full-input fit-space centroids for a bounded hierarchical router;
do not assume a projected-space nearest cell is nearest for donor-function
error. The initial pair should retain16 parents and compare one selected
parent function with one of10 local child functions: E16 versus E160,
one active896x896 function in either arm. Freeze exact clustering, route
arithmetic, empty-cell policy and validation selection before fitting.
No learned common, additional active expert or validation-time correction.
Larger hierarchy/DRAM/LUT cost remains a later measurement, not implied
by this small count pair. Native phase60 integration remains required.

## Next prerequisites and decision

1. Freeze and implement the affine native operator using the audited
   BF16 decode/FMA primitive from METH-224. Export24 distinct actual
   source-layer tangents at existing METH-125 inputs as a component
   fixture; verify analytic derivatives against autograd first.
2. Bind serialized weights/bias to the FP32 oracle on the same existing
   states. Retain median<=1e-4/worst<=5e-4 relative L2 and24-layer
   component median<=10ms with six threads. No timing retry. Declare
   local budget before execution; no new T4/source inference is needed.
3. Only if apparatus/native gates pass, freeze one actual layer12
   E16/E160 function test using the saved METH-222 fit/consumed-validation
   inputs/outputs. Retain full conditional NMSE<=.01, >=10% SSE gain
   against E16 and rotated child choice, and paired-window P05>0.
   Report cell support, distances, distinct stored functions and bytes.
4. A failed fixed tangent geometry stops. A local pass still needs
   actual native routed-bank arithmetic, multi-layer causal composition,
   independent LLM quality, tasks/generation and same-artifact >=50tok/s.
   The real GigaChat approximately10B transfer is a subsequent independent
   family/scale test, with its existing source bindings/calibration reusable.

Neither the source Jacobian nor a larger bank proves arbitrary-n
quality preservation. The scientific change is a new way to transfer
known donor functions into independently selected capacity; its utility
is still unknown.
