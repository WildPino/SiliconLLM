# Next bounded transfer: ALL-FIT values with a source-null-J prior

8 October 2026. Derived algebra and PROPOSAL only. New covariance kernel,
compiler and candidate are NOT implemented, acquired or qualified.
[Completed zero-prior convex result](CHATBOT_KERNEL_RESULT_20261008.md) closes
that recipe at1.5485% encoded FIT RMS and70.4495% on FIRST64 novel states.
Its numerical solution and exact necessary FULL1% failure are independently
verified. The16-jet converter and old227/231 constructions remain CLOSED.
This document supersedes older operational NEXTs; their bytes remain evidence.

## Step back: which pipeline uncertainty is worth resolving?

Final output is a useful pretrained CHATBOT running in the project engine,
after a reproducible conversion. Local errors are admission gates into that
pipeline. They are not chat quality, native parity, accepted tokens/s or useful
large-n capacity. Qwen source/canonical/capture tools exist; a faithful compact
converter and its complete ALL24 native chatbot artifact remain missing.
Preserve the full donor-relative own-history/task AND accepted50 SAME-artifact
contract, useful n/RAM, LUT winner+mass, physical DRAM and actual family/scales.

The new test asks ONE question: can original source derivative information
select an extension of the cheap full-input affine field that transfers beyond
the finite FIT responses? The last ALL-FIT recipe had a qualified convex
minimum, so unfinished iterative optimization is not its remaining uncertainty.
Phi has3845 rows and14352 coefficient features: real coefficient nullity is
at least10507 per output. This is finite-data non-identifiability, not a
semantic-information percentage or a proof of insufficient affine capacity.

Original16 FIT full source Js and the fixed BF16 shared-core Js are already
captured. The value-only ALL-FIT objective did not use them. Older neural
value+J and16 exact-jet recipes did not implement the new ALL-response convex
objective with a soft prior in selector-null input directions. Combining these
two information sources is the explicit new variable. No new teacher capture,
old fit/control replay or post-error regularizer/count/precision ladder.

## Preserved geometry and whole budget

E16/top4/query32, fixed nonlinear shared512/BF16, full896-input affine private
matrices/BF16, F32 biases/router. Use original per-occurrence cached F32 mass
for ALL-FIT value equations and original F64 SMOOTH SHADOW mass for16 derivative
anchors. The two arithmetic contracts are declared separately; do not silently
substitute one for the other or call them native rounded-program derivatives.

Complete290975744 matrix MAC/583842816 logical coefficientB/1047295232 storedB
proposal and full head/attention/context/F32 cache remain priced. New source
prior changes conversion, not inference width/count. These dimensions imply
no physical throughput/DRAM/quality result. Storage O(n D^2) and selected affine
work O(k D^2) are conditional cost formulas; router/normalization, head and
cache costs must also remain bounded/measured. RAM-only storage growth cannot
establish useful n. UniformE160's28 unsupported leaves stay CLOSED.

## Use the exact real row space of the actual stored selector

Let P[32,896] be the original cached F32 selector, promoted exactly to F64.
Its exact real rank32 is already proved by the qualified nonzero modular minor.
Set

    S = P P^T,
    Pi_P = P^T solve(S,P),
    Pi_N = I896 - Pi_P.

Here solve means Cholesky/triangular solves, never a materialized inverse.
In exact real algebra S is positive definite, Pi_P/Pi_N are orthogonal
projectors and null(P) has dimension864. These statements are separate from
rounded-F64 reconstruction/projection checks. The FIRST new32-square S factor
is an operator prerequisite for this new objective, not a repeat of PCA/QR or
the old geometry experiment.

Do not use saved rounded Q Q^T as though it were exactly idempotent. Saved Q
is qualified approximately, but the projector definition above belongs to
the actual finite P. It provides the exact real identities needed below;
numerical implementation must report its actual residuals and preserve failures.
No new basis, selector, source feature, dimensionality or anchor selection.

Within the original qualified selected cells, smooth normalized routing has
gradient rows in row(P). Thus grad(w_e) Pi_N=0 in real algebra and the private
field's projected derivative is

    J_private(x_i) Pi_N = sum_e w_ie A_e Pi_N.

This is a local smooth identity, not a derivative of F32 selection/rounding or
a global bound across cell changes. Private functions retain ALL896 inputs;
query32 is not an asserted32-dimensional function-input bottleneck.

## One fixed convex objective and its small covariance kernel

Reuse ALL saved FIT U=[1,Z],Z=(x-mu)/r,W_FIT,a,mu,r,
alpha=.8153432850804172. No recentering or new normalization.
Let Theta_e[897,896] hold bias/linear coefficients; A_e=Theta_(e,x)^T/r.
Original16-anchor smooth mass W_A[16,16] is reused from the coupled design.
For anchor i, the source-null residual target is

    T_i = (J_source_i - J_fixed_BF16_shared_i) Pi_N.

Both original matrices are reused bytes. No new source/core J.
Use frozen normalizer lambda=78343.61382048983/836.9166991571583:
the qualified weighted FULL-FIT source-value energy divided by the already
qualified16-anchor source-null-J energy. The latter used the OLD numerical Q
projection; reuse its original definition as an energy SCALE, not as an exact
claim about the new projector's energy. Do not recalculate or tune the scale
after observing errors. Set beta=lambda/r^2. Original source-null energy near
94% was isotropic derivative energy, not semantic/reachable-state covariance.

With R=Y-shared and B=diag(sqrt(a))Phi, choose ONE objective

    ||B Theta - diag(sqrt(a)) R||_F^2
    + lambda sum_i ||sum_e W_A[i,e] Theta_(e,x)^T Pi_N/r - T_i||_F^2
    + alpha ||Theta||_F^2.

The16 derivative anchors have equal weight; all3845 value occurrences keep
their previously qualified inverse-exact-x-multiplicity weights. Source J is
used only in the selector-null prior, not as a promised full-J interpolation.
Positive alpha makes this fixed real quadratic minimizer unique.

Define M=W_A^T W_A and H_leaf=alpha I16+beta M. The coefficient prior precision
H acts as alpha I on bias and selector-row-space coefficients, and H_leaf on
the leaf axis of selector-null coefficients. H_leaf is positive definite even
if W_A has a deficient rank. No new W-rank or SVD assertion is needed.

Its prior linear term and mean are

    g_e = (lambda/r) sum_i W_A[i,e] T_i^T,
    Theta0_(e,x) = [solve(H_leaf,g)]_e,
    Theta0_(e,bias) = 0.

g is selector-null by construction. This mean is NEW source-derived soft-prior
coefficients, not a reuse of the failed16-jet bank. Do not acquire g/Theta0
in the first label-free covariance prerequisite below.

The inverse-precision operator is

    H^-1: bias -> /alpha,
          input row-space -> /alpha,
          input null-space -> solve(H_leaf, ...), on the leaf axis.

Avoid a14352-square coefficient normal matrix. Set C=Z P^T and build

    Qinner = C solve(S,C^T),
    Ninner = Z Z^T - Qinner,
    massH  = W_FIT solve(H_leaf,W_FIT^T),
    K_H = diag(sqrt(a)) * [
            (W_FIT W_FIT^T) elementwise (1+Qinner)/alpha
            + massH elementwise Ninner
          ] * diag(sqrt(a)).

In real algebra K_H=B H^-1 B^T is PSD. Since H>=alpha I,
K_H<=B B^T/alpha and condition2(I+K_H)<=1+trace(B B^T)/alpha.
The old trace-based scale gives a nominal~10^6 ceiling; this is not an actual
measured or rigorous condition2 bound for rounded saved matrices.

One3845-square factor solves ALL896 output RHS:

    (I3845+K_H) V = diag(sqrt(a)) (R-Phi Theta0),
    Theta = Theta0 + H^-1 B^T V.

Recovery uses the small S/H_leaf solves for the row/null parts of each leaf;
no explicit inverse or large coefficient normal. This is an algebraic solution
of the new objective. It does not imply preservation of unseen curvature,
fresh dialogue or any useful-large-n/general-family result.

## FIRST next action: implement/freeze covariance prerequisite ONLY

Reuse saved FIT U/W/a and scalar r/alpha; actual P bytes, original16-anchor
smooth W_A, reused source energy SCALE reports and whole budget. Bind actual
code/protocol/used inputs/runtime before the new S,H_leaf,K_H,L observables.
This phase must read no source/core y/J matrices, new prior mean/coefficients,
candidate responses or errors; no development x or old operator/fit replay.

Construct S/H_leaf and their small factors, K_H and L=chol(I+K_H).
Prospective prerequisites: finite C-order F64 arrays; positive lambda/beta/
alpha and positive Cholesky diagonals; small/full relative symmetry<=1e-12;
small/full factor relative reconstruction<=1e-10; projector symmetry/
idempotence and ||P Pi_N||_F/||P||_F<=1e-10. Projector diagnostics are for THIS
new S-solve definition. No jitter, alpha/scale/rank/basis retry on failure.
Real PSD/projector deductions and actual rounded residuals stay separate.

Proposed CPU-only Python/NumPy/psutil five restricted roots/OpenBLAS1,
worker180s/family300s/conservative summed OS4GiB/outputs512MiB/log2MiB through
actual worker exit. One new3845-square kernel/factor,32-square selector and
16-square leaf factors; no14352-square coefficient normal or GPU/T4/install.
Charge conversion/audit workspace separately from inference storage.
FIRST independent kernel/operator assembly audit and known-instance/typedUTC
terminal closure must precede a compiler. Fix its actual independent assembly
and gates before audit observables; do not repeat the completed original kernel.

## Conditional ONE compiler and return to the pipeline

Only a qualified covariance prerequisite permits a separately frozen compiler.
Reuse ALL3845 source BF16 y, saved complete FIT shared-only F64 AND batched-F32
values,16 original source/core Js, and64 existing novel shared-only F64 AND
batched-F32 values. There is now no need to reacquire a source/core response.
New coefficients/field responses, new positive-quadratic residuals and errors
are the new observables. No source/old bank/control/optimizer/audit replay.

Freeze dual/primal stationarity and folded/kernel agreement tests, objective
terms, FULL-FIT F64/encoded response errors, source-null-prior error and original
codec/finite resource contract before acquisition. Require novel1%/ALL16
category3% unchanged; an exact necessary64-prefix/FULL-denominator failure
closes THIS new objective and stops remaining responses/ALL24/native.
An inconclusive prefix never admits fidelity. Distinct coefficients are not
distinct useful functions. No post-error lambda/alpha/width/anchor/precision
ladder. First independent saved-coefficient/error/exact-proof audit required.

If local transfer qualifies, next work must assemble/price ALL24, implement
the needed native conditional operator and canonical chat metadata, export a
complete artifact with no source-weight inference fallback, and verify actual
engine composition. Then fresh EXCLUDED own-history multi-turn/tasks and
accepted50 SAME artifact, cold/first request, DRAM, LUT winner+mass and useful n.
Consumed development can reject conversion; it cannot establish fresh quality.
Generalize through actual family/scale variants after the tractable full path.

If this combined prior fails, preserve its fixed convex certificate/failure
and reassess the representation/source-coverage uncertainty. Affine private
fields have no intrinsic per-cell curvature; a fixed shared512 need not supply
the omitted SwiGLU curvature. That is an OPEN hypothesis, not an established
failure cause or permission to increase n blindly. A later changed experiment
must isolate curvature/source-support/routing with a new prewritten decision,
respect all previous closures, and retain the complete chatbot cost contract.
