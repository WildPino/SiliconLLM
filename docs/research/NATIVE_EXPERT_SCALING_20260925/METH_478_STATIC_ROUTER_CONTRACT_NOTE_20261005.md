# Proposed478: static native router contract, before numerical scouting

5 October2026. Source inspection and real-arithmetic derivations only. No478
protocol/source/binding frozen; no import/compile/query/geometry observation.
This note advances the prerequisite in the [proposal](ALGEBRA_REASSESSMENT_AFTER_477_20261005.md).

## Actual393 arithmetic, not inferred from the storage type

The qualified393 router classifier uses unquantized F32 weights, but its mv
calls dot64: F32 inputs/weights converted to F64, two four-lane accumulators,
four-lane combination and sequential horizontal reduction, final F32 cast.
The separate F32 dot function is not the classifier's path. Source compiler
flags include fno-fast-math and ffp-contract=off. Geometry that assumes an
all-F32 dot has the wrong source contract.

For768columns each lane accumulates96products. Assuming exact F32-to-F64
conversion of the source values, each product of two finite binary32 numbers
is exactly representable in binary64: at most48significand bits and exponent
range safely within binary64. Addition still rounds and cancellation can
produce a small result. No claim of byte-identical arbitrary BLAS reduction.

Let S=sum_j w_j x_j in real arithmetic, B=sum_j |w_j x_j|, u64=2^-53,
gamma_q=q*u64/(1-q*u64). A conservative reduction envelope uses q=D+8=776:

    |t64-S| <= gamma_776 B.

The reduction envelope assumes F64 round-to-nearest and no exceptional
intermediates. It deliberately overbounds the actual lane depth. With round-to-nearest
binary32, gradual underflow and no overflow in the final cast:

    |s_native-S| <= gamma_776 B
                    + u32 (|S|+gamma_776 B) + 2^-150,
    <= [gamma_776 + u32(1+gamma_776)] ||w||_2 ||x||_2 + 2^-150,

where u32=2^-24. The inequalities are deductions under explicit hypotheses,
not observed native certificates. Input conversion/DAZ and final FTZ/MXCSR
must be admitted or bounded separately; finite score witnesses alone do not
prove those flags. Centres stored F64 have products with more than48bits,
so the exact-product argument must NOT be reused for centroid dots.

Node mean/radius/norm computations also require outward rounding or an explicit
error envelope. A rounded arithmetic mean need not give sum(w_i-c)=0; keep
the mean residual term in the mass bound. Near a tie use the original rule:
first/smallest ID wins, strict comparison only. Ambiguous bounds refine/fallback.

## Native normalization is a different numerical composition from real exp

393 chooses the earliest maximum native F32 score. For each expert in ID order:

    d_e = RN32(s_e-s_chosen),
    v_e = RN32(exp_binary64_libm(binary64(d_e))),
    z_native = sequential_binary64_sum(v_e),
    a_native = RN32(1/z_native).

Therefore real Z=sum exp(w_i^T x) bounds alone are not bounds on a_native.
The selected shift depends on the exact chosen score. Float subtraction,
library exp error, F32 cast/subnormals, sequential F64 sum and reciprocal cast
need explicit contracts. C exp is not licensed as correctly rounded by source
spelling or a few toy controls. Do not invent an all-input libm error bound.

Chosen next478 scope: interval admission on ALL cached native query witnesses
with independently checked enclosures and actual linked runtime bindings.
This is an empirical feasibility screen on consumed data, requiring new
whole-model quality/rate for promotion; it is not a universal floating-point
certificate. Any enclosure failure is retained before further work; no
post-observation enlargement. Full arithmetic and interval tolerances must
be concretely declared BEFORE first numerics. A universal contract would
additionally require the actual linked exp implementation's error guarantee.

## Operational consequence

Exact first resumption: complete the group mean/radius/dot/normalization
envelope and its tiny controls in the explicitly empirical full-witness scope,
admit ALL393 native inputs/score/probability joins and complete runtime/assets,
then freeze ONE fixed tree/traversal/work gate/resource protocol. No C cost,
model replay or hierarchy sweep now. No new478 result is claimed.

Sources inspected: meth393_switch_router_audit.c (dot64/mv/ffn),
meth393_switch_router_audit.py (actual compiler flags),
meth393_switch_router_analysis.py (393 probability reconstruction).
477-R1 retained, all previous closures and full goalACTIVE/INCOMPLETE unchanged.
