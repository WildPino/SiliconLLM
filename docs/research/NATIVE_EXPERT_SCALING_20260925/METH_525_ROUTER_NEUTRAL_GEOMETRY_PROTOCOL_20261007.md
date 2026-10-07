# METH525: one router-neutral norm-slice query per source parent

7 October2026. Prospective geometry-only protocol, frozen with its source code
before binding and the first numerical observation. Goal ACTIVE/INCOMPLETE.
[Selected question](METH_525_ROUTER_NEUTRAL_SOURCE_COVERAGE_NEXT_20261007.md):
[524](METH_524_SOURCE_HYPERPLANE_TREE_RESULT_20261007.md) cannot distinguish
adequate from undersampled regions using zero dev residual. Change the source
domain coverage variable, without relaxing the closed524 split criterion.

## Question, reusable evidence and decision

Can one deterministic point per exposed parent cross an omitted source ReLU
boundary while preserving ALL ideal parent-router scores and the observed input
metric norm? If all127 points are certified and their physical F32/I16 bridges
keep the original parent and cross the chosen code-plane sign, a separate next
protocol may acquire one new source function label at each frozen point.

Use original523 development medoid anchors,512 hinges and source reference signs
whose complete numerical prefix was independently audited. Its original900s
resource failure remains explicit; do not relabel523 as overall successful.
Reuse original479 full cached score vectors; NO original router query replay.
Reuse all17540 original493/499 input/UID wires and11721 dev/5819 consumed split
only for wire/identity admission. Selection uses the127 dev anchors and source
weights, never consumed labels, errors, counts for fallback tuning or responses.
Parent0 is unexposed and has no anchor/query; all parents1..127 participate.

This is a finite source-region information probe, not useful expert count,
function fidelity, whole quality, timing or transfer completion.

## Actual source contracts, independently recoverable in local C

Original380 payload `results/native_expert_scaling/meth380_switch_w8a8_export/weights.bin`:
7541946880B, original full SHA
`6bee473e1797332b4650b69a3ba6771128d24e7395d4b961515d844fd71c0cfe`;
fresh-used extents and unchanged size/mtime are checked before and after science.

| Organ | Encoding/shape | Offset/bytes | SHA256 |
| --- | --- | --- | --- |
| decoder.block.11.layer.2.mlp.router.classifier.weight | F32,128x768 |5662135296/393216|021bd4d9a0dc3bab78abeae03a7dfbe7bfa19c623e6b29d58cc7c4fd7e7b1846|
| decoder.block.11.layer.2.layer_norm.weight | F32,768 |5056186368/3072|23e7f70694ef54901fd7a3010ed5228e756a156adc2ff85264a69531a57be97f|

`meth393_switch_router_audit.c`, `meth393_switch_router_trace.h`,
`meth461_switch_common_input.c` and `meth479_source_router.c` are bound source
contracts. Source393/461 feed normalizes hidden state into literal F32 x, then
computes F32 router scores from F64 AVX dot products on that same x. There is no
router bias. Dot order:96 blocks, four independent low/four high lanes, separate
multiply/add, low+high then scalar four-lane sum and cast F32. First lowest ID
wins exact ties. Softmax subtracts winning score in F32, evaluates exp in F64,
casts each exponential F32, sums all128 in sequential F64, returns F32(1/sum).

The source WI receives Q16(x), not literal continuous x: maxabs and
alpha=maxabs/32767 are F32 (zero alpha becomes1), F32 divide, round nearest even,
clip +/-32767. WI is I8[3072,768] with positive original F32 row scales. Selected
dot products are exact I64. Row scales/alpha preserve code-dot sign. No hidden
requantization, WO projection or full source response is performed in525.

Source RMS normalization squares hidden values in F32, accumulates F64, casts
mean F32, then uses F32 sqrt/scale and two F32 output products. Therefore the
continuous y=G^-1*x construction below preserves an observed weighted norm shell;
it is NOT claimed to be exactly an output of that discrete norm operator or a
point reachable by an actual decoder trajectory. Natural held-out quality and
whole-model validation are still necessary.

Canonical479 `unique_inputs.bin`: M479UNI1, header24B,238872 records of3720B,
total888603864B, SHA
`f80e46251c7a13ea86483883f49cfa3d15ceeff08ad95fe9b7928a3debfa1c57`.
Each anchor joins493 canonical UID to its input bytes, winning parent and selected
F32 probability. Cached scores are read, not recomputed. All254 original WI
codes/scale extents plus router/norm yield256 extents/301587456 fresh hashed B.

## Algebra: affine router slice intersected with a sphere

Let R be the actual128x768 F32 matrix and G its actual diagonal norm weights.
Any zero G coordinate explicitly stops this fixed variant. Define C=R*G and
y0=F64(x0/G), treated as the represented anchor coordinate. F32*F32 products in
C are exact F64. Division and return-to-x bridges are separately accounted for.

With P the true orthogonal projection onto row(C), a=P*y0, b=y0-a, r=||b||:

    y'=a+v, v in ker(C), ||v||=r
    C*y'=C*y0, ||y'||=||y0||.

Thus ideal ALL scores, winning parent AND normalized mass are invariant.
Physical F32 scores, exp, full denominator and probability are recalculated
only at constructed new points; neither probability byte equality nor invariance
after quantization is assumed.

For an omitted WI code row w_j, h_j=w_j*G (positive SI scale cancels),
k_j=(I-P)h_j, c_j=h_j*a. On this slice, if dim ker(C)>=2, its exact reachable
interval is [c_j-r||k_j||,c_j+r||k_j||]. Test ALL3072 rows for each parent, retaining
the512 original hinges as excluded;2560 rows per parent can be selected.

Only a rigorously sign-straddling interval is eligible. Desired target t_j is
half the opposite-side endpoint: upper/2 for reference sign0, lower/2 for sign1.
One fixed depth, no grid, rounding threshold tuning or alternate target trials.
Let n=k/||k||, u=(t-c)/||k||, A=n*b, w=b-A*n, z=sqrt(r^2-u^2). The nearest point
on this fixed target plane is

    y'=a+u*n+z*w/||w||,
    distance^2=2r^2-2(A*u+||w||*z).

This follows by maximizing b*v at fixed n*v=u and ||v||=r; Cauchy-Schwarz fixes
the remaining tangent direction. It minimizes distance to the chosen midpoint
plane, NOT distance to the zero boundary, function error or natural likelihood.

Unlike the earlier proposal's possible axis fallback, this frozen implementation
declines w=0 or numerically uncertified tangent/radicand cases. No random/axis
repair or alternate branch is attempted. That is a recipe applicability stop,
not proof that no same-score point exists.

## Exact rank and qualified represented projector

Remove only exact zero and +/- duplicate R rows, keeping the first identity.
Other dependencies are not guessed. Main embeds exact F32 rational entries into
GF(2147483647), performs Gaussian elimination and retains pivot columns. Auditor
decodes IEEE754 bits directly into GF(1000000007). Both primalities are checked
by exact trial division through integer sqrt; modular I64 products stay below
2^63. A full-row-rank minor in either field proves real rational row independence.
Any remaining rank deficiency stops this variant; it is not a general theorem
against router-neutral construction. Nonzero G preserves rank.

Main reduced SVD of C.T produces represented Q,s,Vt. With binary64 unit
roundoff U=2^-53 and gamma(n)=nU/(1-nU), qualify orthogonality, reconstruction,
and rowspace residual using Frobenius upper bounds and absolute-product dot
reduction bounds. With r0 kept rows:

    oq=||Q.T Q-I||F + gamma(769)||abs(Q).T abs(Q)||F
    ov=||Vt Vt.T-I||F + gamma(r0+1)||abs(Vt)abs(Vt).T||F
    rec=||C.T-(Q*s)Vt||F
        + gamma(r0+3)||abs(C.T)+(abs(Q)*s)abs(Vt)||F
    sigma_lower=s_min sqrt(max(0,1-oq))sqrt(max(0,1-ov))-2rec
    rb=||C-(CQ)Q.T||F
       + gamma(768+2r0+5)||abs(C)+(abs(C)abs(Q))abs(Q).T||F
    cnorm=||C||F (1+gamma(C.size+1))
    base=2(rb+cnorm*oq/(1-oq))/sigma_lower + 2oq/(1-oq)
    projector_error_bound=2base.

Weyl's perturbation inequality supplies sigma_lower; residual/sigma and the
orthogonality defect bound the difference between QQ.T and the true row projector.
Final factors2 also guard floating evaluation of positive bound expressions.
Eligibility: exact full kept-row rank, null dimension>=2, sigma_lower>0,
s_min/s_max>=1e-6, base<1e-6. Unsupported conditioning does not loosen limits.

Auditor independently contracts the stored SVD factors with explicit einsum,
recalculates every certificate constant and qualifies it with gamma(10000)
reduction slack. An independently generated reduced QR span, triangular inverse
residual and QR reconstruction provide a separate singular lower bound and
projector certificate. Compare spectral QQ.T versus QR QR.T under joint bounds.
Stored qualified Q bits define the candidate representation in BOTH passes;
changing to a new QR candidate recipe would change the frozen selection.

## Forward bounds, statuses and selection

Main propagates projector and arithmetic errors through a,b,r,k,||k||,c,
interval endpoints,t,n,u,A,w,||w||,radicand,sqrt, distance and the final vector.
All formulas are in the frozen `meth525_geometry_math.py`; the audit uses separate
`meth525_geometry_audit_math.py`, with explicit contractions for independent
reductions, and imports no main selection/math module. Main BLAS1 products versus
audit explicit sums are qualified by joint propagated bounds, not forced equal.

Require endpoint intervals strictly straddling0, ||k||>2*its bound, r>its bound,
(r-dr)^2>(|u|+du)^2 and ||w||>2*its bound. Square-root perturbation uses
delta_arg/(computed_root+positive_lower_root), and normalize using denominators
||k||-delta_k and ||w||-delta_w. Positive final bound evaluation uses endpoint2x,
distance4x and vector8x inflation; endpoint/distance arrays round bound upward.
None is chosen after source values. Standard dot/norm gamma limits and no
overflow/invalid/divide are required; underflow is ignored conservatively at this
source scale, and tiny zero-row score bounds round to the least positive F64.

Per-parent panel F64[3072,7]: c, r||k||, endpoint bound, t, distance^2,
distance^2 bound, ||k||. Status U8[3072]:0 eligible representative;1 retained hinge;
2 no certified straddling interval;3 degenerate/uncertain construction;
4 duplicate represented plane. Aliases use exact primitive I8 code vector divided
by row GCD, oriented by first nonzero positive entry, paired with oriented
reference sign. Keep lowest j. Positive scale cancels; symmetric midpoint targets
make aliases the same ideal point. Selection compares all representative distance
intervals. Lowest distance then lowest j is reported, but physical construction
requires its upper bound strictly below every other representative lower bound.
Unresolved order produces no query, no silent tie promotion.

At each resolved y', x'=F32(y'*G), independently qualify its physical metric norm
and ALL128 score differences from cached x0. For each router row i:

    score_bound_i=4[ ||R_i G||*point_error
        + sum |R_i|*(|y'G-x'|+|y0G-x0|)
        + (gamma(769)+2^-24) sum |R_i|*(|x'|+|x0|) ].

Store upward-rounded128 bounds and full scores/exponents/probability. Main must
lie inside ALL bounds. Norm bound is4[point_error+||x'/G-y'||
+gamma(771)(||y0||+||x'/G||)]. Auditor independently reconstructs both sets using
math.fsum and qualifies the retained error/relative error fields. Audit computes
new router vectors with scalar lane order, Q16 with independent tie-even logic,
and selected code dot using Python exact integers; scores,128 exponents,selected
probability,codes and alpha must match main BYTE. Both passes retain actual
winning ID/full denominator/probability change and source sign.

New3D control R=[1,0,0], G=[2,1,1], y0=[1,3,4], h=[0,-1,0] gives
y'=[1,-5/2,sqrt(75/4)], norm^2=26, router score2, preactivation5/2. Main runs
the actual panel routine; audit proves square/norm/target with exact Fraction.

## Frozen criteria and scope

ALL five fixed eligibility gates must pass:

1. Actual invertible G and certified full-rank/conditioned projector.
2. ALL127 have a resolved certified new geometry.
3. ALL127 retain the original physical winning parent.
4. ALL127 cross their selected omitted source WI code sign after Q16.
5. ALL127 physical metric relative norm errors are<=1e-6.

Decision is `ELIGIBLE_ONE_NEW_SOURCE_QUERY_PER_PARENT_NOT_TRANSFER_PASS` or
`ROUTER_NEUTRAL_HALF_ENDPOINT_GEOMETRY_CRITERION_CLOSED`. Apparatus gates7 main,
8 independent audit,5 terminal admission distinguish trustworthy failure from
scientific eligibility. The per-parent geometry and identities freeze BEFORE any
source function labels. No FFN response, full3072-row WI projection, hidden codec,
WO,readout fitting,model/native/C/corpus/T4 call; those counters are zero.
Selected WI integer checks and NEW128-score vectors are counted explicitly in
each numerical pass, at most127 each. Auditing them is required independent work,
not an original-query or source-label replay. This is a physical-arithmetic
reference, not an actual C execution or measured DRAM contract.

## Resource and retention contract

Binding<=180s/512MiB; main<=180s/512MiB; audit<=240s/512MiB;
combined output<=32MiB. CPU logical10/BLAS1, Python3.12.10/NumPy2.4.6/psutil7.2.2/
threadpoolctl3.7.0 from bound511 runtime; isolated -I -B -X utf8 and empty redirected
cache. Full used input/runtime hashes and all256 source extents are fresh in
each pass, not an unpriced setup. Maximum390144 plane rows/127 constructed
points. Panels21848064B,status390144B plus small geometry/certificate arrays and
JSON fit32MiB. No concurrent science or timing. Preserve approved publisher and
all foreign tracked/untracked work; exact three foreign hashes are bound.

Source freeze commit before binding; binding commit before main; completed main
commit before the one independent audit. Namespace525 is exclusive. Preserve
first faults/partial inventories BEFORE a numbered repair; never replay completed
main/native/source/audit work. Hard timers/OS peak and actual executor exits are
retained. Three owned PID+creation instances must be closed. Typed UTC Event1000
queries have positive controls179810/179791 and zero relevant faults. Finalizer
reads only JSON/hashes/process/event receipts; no numerical replay.

Run isolated runtime with `meth525_prepare_binding.py`, then
`meth525_geometry_main.py --binding-sha <bound SHA>`, then
`meth525_geometry_audit.py --binding-sha <SHA> --main-sha <retained SHA>`.
`meth525_windows_terminal.ps1` and `meth525_finalize.py` admit terminal evidence.
Exact argv, revisions, hashes,resources and commands are recorded in outputs.

If eligible,526 must separately recover/price original physical FFN response
semantics and freeze the acquisition/comparison before querying these immutable
points. A new label can falsify single-reference coverage but does not prove
natural usefulness, global coverage or enough redundant functions. Same-shaped
one/multiple/rotated functions, prospective priced scarce fallbacks, all six/rare
1% gates, true original winner+mass cost, useful large n/RAM, actual kernel/DRAM,
fresh donor quality AND SAME>=50/s and additional actual families/scales remain
required. This fixed linear-router/ReLU variant does not generalize to saturated
router rank, other discrete domains or smooth SwiGLU without new proof/evidence.
