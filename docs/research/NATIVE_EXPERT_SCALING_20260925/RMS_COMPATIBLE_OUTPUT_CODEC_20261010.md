# A paired output representation executable by original RMSNorm

10 October2026. Algebra/prospective representations, not a measured compact
codec or causal encoder. Reuses the [readout decomposition](TARGET_READOUT_GEOMETRY_20261010.md)
and [qualified cached state/logit pair](SOURCE_CACHED_FINAL_RESULT_20261010.md).
Original engine.c remains unchanged. Its scalar rmsnorm atlines439-440 uses
epsilon1e-5 and final norm/head atlines642-643, exactly the operators below.

## Correct object to compress

Let f be the source's actually observed postnorm BF16 feature decoded to reals,
W its actually observed BF16 head decoded to reals, s=.01953125. The reference
real-arithmetic logits are sWf; actual BF16 head/scalar rounding is a separate
measured term. Raw hP, projected diagonal gamma and independently adapted head
are not a coherent substitute. Choose a FIXED column-orthonormal P using FIT
information only, define phi=P^T f, and pair it with projected head sWP.

Then projected logits are sWPP^T f. The exact residual is sW(I-PP^T)f.
This removes the spurious norm/diagonal-coordinate mismatch, not discarded
output information. Softmax ignores scalar common shifts; a useful loss is
full-V KL/ID disagreement, not raw feature energy alone. P selection or output
head fitting is still open and must precede evaluation under frozen DEV gates.
Different source state execution formats must be paired with their own labels.

Original target has d=256, n_C(u)=Gamma_C u/sqrt(||u||^2/d+epsilon), followed
by a linear head. A naive phi passed through this norm changes its magnitude.
The following constructions show how the output geometry can be paired with
the existing operators in real arithmetic. They do not show a cheap causal
SSM/SWA/ternary function can produce u, or that255/256 output directions suffice.

## Construction A: all256 directions through inverse RMS

For scalar gain a>0, choose Gamma_C=aI. On ||phi||^2/d<a^2 set

    u = sqrt(epsilon) phi / sqrt(a^2-||phi||^2/d),
    W_C = s W P.

Substitution gives ||u||^2/d=epsilon r/(a^2-r) where r=||phi||^2/d, hence
n_C(u)=phi exactly. This is a finite open-ball inverse, not an inverse of
RMSNorm with zero epsilon or a global linear change of basis. Near the boundary
the inverse is ill-conditioned; far inside, u may become very small because
sqrt(epsilon)=.00316227766. Native residual/activation/quantization precision
and learning gradients could then be problematic. No gain selected or tested.

## Construction B:255 information directions and one norm carrier

Use P with255 columns and choose positive fixed gain a and radius factor rho.
On the declared domain ||phi/a||^2<=d*rho^2 construct

    u = [phi/a ; sqrt(d*rho^2-||phi/a||^2)],
    Gamma_C = I,
    W_C = s*a*sqrt(rho^2+epsilon) [W P , 0].

Then ||u||^2=d*rho^2 and the original RMS denominator is constant
sqrt(rho^2+epsilon). The zero last head column discards the carrier, so
W_C n_C(u)=sWP phi=sWPP^T f exactly. This uses the existing256-dimensional
state/norm/linear head, without a new native operator. It spends one information
direction to preserve varying phi magnitude rather than silently renormalizing
it. This is an algebraic feasible representation, not an exported artifact.

Gain/domain must be chosen without fitting DEV. A source/operator bound can
provide a declared norm domain; an observed FIT maximum alone is not a guarantee
on future inputs. If a DEV/future input violates the domain, retain failure;
do not clip silently or refit the gain on DEV. Large gain also shrinks information
coordinates relative to the carrier and may harm original activation quantizers
or enlarge F16 head values. Radius/gain require explicit conditioning and native
precision gates. Square-root carrier production by the causal core is missing.

## What these identities settle and what remains

Both constructions separate three questions: source-state/logit custody;
retained output information after projection; and whether the original causal
core can reach the paired representation. Only the first has an exact measured
one-case cached control. The identities settle compatibility with the final
RMS geometry on their declared domains in real arithmetic. They do not admit
quality, speed, useful n, DRAM traffic or family applicability. Head F16,
coordinate precision, source BF16 rounding, causal target reachability and
full own-history behavior remain measured gates, not consequences of algebra.

A single paired compact control should follow consistent FIT calibration and
an explicit affordable P/head choice. No rank/width/scale grid, extra fixed-head
oracle iterations or T4 selected by this memo. If a paired static readout still
loses too much information, preserve that local failure and inspect conditional
readout/information geometry; do not infer a universal D256 ceiling from one P.
