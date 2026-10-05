# After473: preserve the nonlinear function before choosing its representation

5 October 2026. Analysis and prospective NEW474; no output-space fit or numerical
observation has been performed. 473's five apparatus gates and six independent
retention gates passed. The full goal remains active and incomplete.

## What the new evidence settles

The ideal development preactivation criterion needs total rank7365 at minimum,
against4684 in the fixed70% decoded-factor budget. Numerical guard-band bounds
are7365..7366. Nominal byte floor465329216/605945856=76.7939%. This closes that
particular budget/metric family, including variable ranks, without candidate
construction. It does not forbid looser budgets or another function metric.

The70% storage criterion is a local prospective screening target. It is not an
extra requirement silently imposed on the full goal. A larger representation
could still be useful if it passes actual quality, active cost and useful-n
requirements. The current evidence does not give it those qualifications.

Input support, learned-WI response and complete FFN response are three different
objects. The failed input-PCA fit and the WI spectrum do not measure the number
of directions that the complete nonlinear function must preserve. No additional
source books or rank grid is justified by these two results.

## Why preserving WI can be more demanding than preserving the function

For the continuous approximation, temporarily omit the hidden quantizer and
final casts, and write

    z = A*x,       h = ReLU(z),       f = B*h,
    A = S_I*W_I,   B = S_O*W_O.

With the same activation mask D_x in a local region, the Jacobian is B*D_x*A.
The input/output directions seen through this operator can have a different
spectrum from A. Directions removed by D_x or lying in B's null space can
change preactivations without changing the continuous function in that region.
Across regions D_x varies, so one local Jacobian does not prove a global basis.

The actual source also includes the hidden A16 quantizer and native rounding.
Those terms can change output directions and gate behavior. They must remain
in a complete function comparison; the continuous identity is not a numerical
certificate for the native source. All19962 complete source F32 FFN vectors
are already available and byte-qualified in472, so that source behavior can be
used without another model/native replay.

## Proposed NEW474: a private OUTPUT-space floor for full function error

The new variable is one output subspace PER EXPERT, learned from complete
source functions on development data. It is not the global across-expert output
bases rejected in444/445, the reference0 weight metric in446, or a WI input-PCA
retry. Same107 fixed IDs,21 original fallbacks, full role/novelty/book policy.

For expert e, form development rows sqrt(w)*f_e(x), with the same exact
effective-input/book dedup and equal-book weights. One full SVD supplies a
fixed rank32 left basis P_e in R^(768x32). The basis and any policy for zero,
degenerate or numerical-null modes must be specified before first execution.
No held-out output may select the basis or an expert's policy.

For this fixed orthonormal P_e and any possible coordinate function g_e(x),

    ||f_e(x) - P_e*g_e(x)||_2^2
      >= ||(I - P_e*P_e^T)*f_e(x)||_2^2.

Proof: the error splits into orthogonal residual and in-span coordinate error.
The optimal pointwise coordinates are P_e^T*f_e(x). Thus projection of the
known source output gives an optimistic floor even for an arbitrarily powerful
coordinate map. It is not a converted model or a map computable without source
work. A failing floor rejects THIS learned output space; it does not reject
all other spaces or nonlinear representations.

Evaluate all original primary complete-function gates from472: every107
novel-validation expert<=5% equal-book RMS, all3065 natural queries<=5%, and
every64 natural-validation book<=10%. Original fallbacks are exact. All failures
remain visible; no output-energy or health selection. A pooled mean alone is
especially inadequate after the25/37 energy concentration seen in472.

The floor initially uses ideal F64 projection of the stored source F32 vectors.
Before using it to reject an actual F32 output-factor family, qualify the exact
arithmetic scope and the final-rounding allowance prospectively. Native F32
rounding can perturb a real subspace, and an oracle floor is not an actual
candidate's error. Freeze source/protocol/bindings/null-mode policy/byte bounds/
gate decisions before first compile, import or numerical operation.

## A concrete candidate would be possible only after that floor passes

Keep original I8 WI, WI row scales, ReLU and hidden A16. A potential decoded-WO
replacement uses stored F32 P_e and C_e approximating P_e^T*B. This would avoid
the source full-width WO, rather than compute it and project afterwards. Its
actual coefficients, rounding, full FFN errors and native C cost still need
separate qualification. The oracle does not supply any of those results.

Nominal bytes with same107/21/header policy and no separate WO row scales:

    B_output = 353188928 + 15360 * sum_e(r_e).
    uniform rank32 => 405781568 bytes, about66.967% of original bank.

This count retains all original WI coefficients/scales and all fixed fallbacks.
It is prospective accounting, not a new exported ABI. Changing the rank, basis,
fallback policy or source-data ladder after observing validation is forbidden.

Proposed scope: CPU-only one bounded screen, hard180s/2GiB/new96MiB, exact output
and workspace bounds still to freeze. Reuse qualified19962 source vectors/query
provenance; no GPU, framework, tokenizer, source replay or adaptation. Full
spectra and top32 bases supply witnesses; primary function projection errors
can be independently rederived without SVD refit.

If the floor fails, close this learned rank32 output-space recipe. If it passes,
one separately frozen real output-factor construction can test the entire
native nonlinear function, followed by the original whole-cost/quality chain.
Do not turn PASS into inherited token rate or final-model quality.

## Return to the full goal

Memory reduction serves useful conditional capacity, while actual active work
serves token rate. Existing whole-cost evidence458 and component evidence455
show that WO accounts for a limited fraction of runtime. Even a good output
representation might primarily increase resident capacity; its speed impact
must be measured against the whole model. No logical-byte or cache assumption
substitutes for physical DRAM measurements.

The completion chain remains: real conditional representation in engine.c ->
whole all-bank/new-state composition -> fresh donor-relative prediction,
generation and tasks AND>=50 accepted batch1 IDs/s on SAME artifact -> causal
useful-n growth, winner AND normalization mass, CPU LUT and DRAM scaling ->
actual applicability to multiple families/scales/~10B/~100B as resources allow.

473 ruled out one metric/budget pair and improved the decision rule. NEW474 is
a proposal to ask directly about function information, with no success claimed.
