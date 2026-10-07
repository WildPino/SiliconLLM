# METH526: exact certificate on the SAME retained source-router representation

7 October2026. Prospective protocol/code frozen BEFORE binding, integer encoding,
products, norm bounds or first numerical observation. Goal ACTIVE/INCOMPLETE.
[Selected uncertainty](METH_526_EXACT_PROJECTOR_CERTIFICATE_NEXT_20261007.md):
[525](METH_525_ROUTER_NEUTRAL_GEOMETRY_RESULT_20261007.md) has18 unresolved nearest
orders under a conservative5.60e-7 projector bound. All109 physically qualified
queries and the CLOSED525 decision remain preserved. No526 query promotion.

## Decision and inputs

Determine whether exact dyadic arithmetic gives a strictly smaller rigorous
projector bound on the SAME saved Q and original C=R32*G32. This changes numerical
information, not source functions, candidate selection, targets or labels.
The certificate could permit a separate bound-refinement-only inquiry. It cannot
itself certify the old18 nearest orders or qualify their unconstructed queries.

Bind admitted525 main/audit/retention/windows and complete original output
inventory before values. Source525 main SHA
`171af522bb131a772a4b7ad68ce266ce977a2c695375b3513fe37b8852c7a306`, admission
`78db99d3c33ae8afa85d0f9c89df7679852927e28c93fc21db4e0316bffc66f3`.
Read retained `basis.npy`, exact768x128 little F64. Source R F32[128,768] at
original380 payload offset5662135296/393216B, SHA
`021bd4d9a0dc3bab78abeae03a7dfbe7bfa19c623e6b29d58cc7c4fd7e7b1846`;
G F32[768] offset5056186368/3072B, SHA
`23e7f70694ef54901fd7a3010ed5228e756a156adc2ff85264a69531a57be97f`.
Fresh hash two used extents/396288B and unchanged original payload size/mtime,
not all unused7.5GB. Fresh used legacy outputs, runtime and committed source
hashes are charged. No old canonical888MB score/input data is used in526.

Admitted525 establishes all128 source rows independent with two modular proofs,
null dimension640, nonzero G, and a positive qualified lower bound
sigma_lower=0.00919757405700082. Reuse this exact retained F64 lower bound;
do NOT rerun source rank/SVD/QR or use observed QR proximity as a true error bound.
F32*F32 products in C are exact F64 (<=48 significand bits, exponent in F64 range).

## Exact dyadic representation and products

Let Q=A/2^k, C=B/2^l with signed integer A,B and the minimal common denominator
powers. Main uses exact `float.as_integer_ratio`, scales to NumPy object arrays
of Python arbitrary-precision ints. Auditor independently decodes original IEEE
F64 Q bits and original F32 R/G bits; multiplies the latter dyadic pairs exactly,
with no original-C floating product or main math import.

Prospective applicability limits: k<=128,l<=160; maximum A and B absolute-value
bit lengths<=127. Explicit stop/fault on unsupported input, with first evidence
retained, never automatic widening. There is no integer overflow in object/Python
arithmetic. These limits imply product/wire bounds:

    |A|,|B| <2^127
    |A.T A|,|B A| <768*2^254 <2^264
    |A.T A-2^(2k)I| <2^265
    |(B A)A.T| <128*2^(264+127)=2^398
    |2^(2k)B-(B A)A.T| <2^399.

Therefore ALL integers fit the fixed signed512-bit wire, with no post-observation
width selection. Retain three exact product identities:

    Gram_defect=A.T A-2^(2k)I                 shape128x128
    CQ_numerators=B A                       shape128x128
    rowspace_residual=2^(2k)B-(B A)A.T       shape128x768.

Together with A768x128,B128x768, these encode EXACT Q.T Q-I, C Q and C-C Q Q.T.
Main NumPy object contractions are exact arbitrary-int multiply/add, not BLAS
floating arithmetic. Auditor checks EVERY product cell with independent Python
scalar sums, independently decoded original operands and verified intermediate
CQ. No approximate/Freivalds/subsample audit. ALL327680 retained integer cells
are covered:196608 operands from bit decoding plus131072 product/residual cells.
Each pass computes37748736 core integer product terms, plus sourceC encoding
and212992 exact squared-norm terms. All are real charged numerical work.

## Fixed safe binary wire

`meth526_contract.py` shares only I/O/schema. Each matrix has32B header
`<8s6I`: magicM526INT1,width64,rows,columns,denominator_power,signed_bits512,kind.
Each element is64B little signed two's complement. Exact file size32+64*rows*cols,
no pickle, compression, custom integer alias or mutable byte width.

| Kind/file | Shape | Denominator power |
| --- | --- | --- |
|1 Q_numerators.bin|768x128|k|
|2 C_numerators.bin|128x768|l|
|3 Q_gram_defect.bin|128x128|2k|
|4 CQ_numerators.bin|128x128|l+k|
|5 C_rowspace_residual.bin|128x768|l+2k|

Five files total20971680B including headers. Save original common powers,
observed operand bit lengths and all proof hashes. The independent audit verifies
the wire against decoded operands/products, not merely against duplicated hashes.

## Norm and projector proof, with exact outward conversion

For each integer matrix T with denominator2^p, its Frobenius norm is
sqrt(sum(T_ij^2))/2^p. Compute exact integer S=sum(T_ij^2), m=ceil_integer_sqrt(S),
then m/2^p is a rational upper bound. Retain S,m,p and reduced numerator/
denominator. Auditor verifies every square sum, m^2>=S and (m-1)^2<S for m>0.
No floating dot, norm or square-root residual contaminates this bound.

Let oq bound||Q.T Q-I||F, rb bound||C-CQQ.T||F, cnorm bound||C||F, and sl be
the admitted positive source singular lower bound. Require oq<1. Q then has full
column rank. For its true orthogonal span projector Pq:

    ||QQ.T-Pq||2 <= oq <= oq/(1-oq)
    ||C(I-Pq)||2 <= rb+cnorm*oq/(1-oq).

Equal source/Q ranks imply
||Pc-Pq||2<=||C(I-Pq)||2/sl, where Pc is the true source row projector. Triangle
inequality gives a valid bound on||Pc-QQ.T||2. Preserve the SAME conservative525
inequality/factors for comparison:

    base=2(rb+cnorm*oq/(1-oq))/sl +2oq/(1-oq)
    projector_upper=2base.

All norms, sl embedding, additions/products/divisions and this inequality are
exact rational arithmetic via Fraction. Final binary64 conversion is the LEAST
F64 >= the rational value: convert, compare exact dyadic Fraction.from_float,
nextafter upward only if necessary. Auditor requires upper>=rational and its
previous F64<rational, for each norm and the final projector upper. Factors4
are inherited conservative surplus, not empirical fitting or a newly reduced
safety factor. Exact representation residuals may still dominate; improvement
is not assumed.

## New controls and fixed criteria

Different dyadic control Q=(1/2,1/2).T,C=(1,1): A=(1,1).T,B=(1,1),k1,l0;
Gram integer=-2,CQ integer=2,residual integers=(2,2), each scaled as above.
Residual squared numerator8 gives ceil root3 and upper3/4. Main runs its object
product/norm functions; audit derives these with scalar integers/Fraction.
Also signed-I512 edge round trips and exact upward1/3 conversion. No control
is run before this source/protocol freeze.

All THREE eligibility gates are fixed:

1. Exact finite dyadic encoding and ALL integer identities verified.
2. Original positive qualified singular lower bound and oq<1.
3. SAME representation projector upper strictly below525's retained upper.

Decision `EXACT_PROJECTOR_CERTIFICATE_ELIGIBLE_FOR_BOUND_REFINEMENT_NOT_QUERY_PROMOTION`
if all pass, otherwise `EXACT_PROJECTOR_CERTIFICATE_NO_STRICT_BOUND_IMPROVEMENT`.
First unsupported encoding/algebra/resource faults remain faults, with explicit
partial evidence and no silent variant; no general impossibility conclusion.
Apparatus7 main/8 audit/5 terminal gates distinguish valid failure from eligibility.

Source FFN/hidden/WO,readout/model/native/C/corpus/T4/new router/selected WI-dot
queries and old525 geometry/query recomputation are all zero. Source original
router/norm exact matrix products are actual numerical work, counted explicitly.
No candidate plane bounds/orders or geometry points are recomputed in526.

## Resources, sequencing and retention

Binding180s/main180s/audit240s, each512MiB; combined outputs32MiB. Fixed matrix
files20.97MB plus JSON/binding/audit summary fit the output cap. Object allocation
and independent Python integer loops are charged. CPU10/BLAS1; bound511
Python3.12.10/NumPy2.4.6/psutil7.2.2/threadpoolctl3.7.0, isolated -I -B -X utf8,
redirected empty cache. No concurrent scientific jobs/timing or new resources.

Exact source/protocol commit before binding, binding commit before main, completed
main commit before the ONE independent exact audit. Exclusive526 namespace;
fresh used data/runtime/source hashes before arithmetic. Preserve ALL525 outputs
by fresh original-inventory SHA before and after the inquiry, including109 query
records. Preserve all foreign untracked work and exact three tracked SHA.
Preserve first faults and partial inventories BEFORE numbered repairs; no
completed main/audit/source/native replay. Hard time/OS peak and actual executor
exit receipts are retained. Three owned PID+creation instances must be closed,
typed UTC Event1000 query available, positive controls179810/179791 present,
relevant faults0. Metadata finalizer performs only hashes/JSON/process checks.

Commands use isolated bound runtime with `meth526_prepare_binding.py`, then
`meth526_exact_main.py --binding-sha <SHA>`, then
`meth526_exact_audit.py --binding-sha <SHA> --main-sha <retained SHA>`.
`meth526_windows_terminal.ps1` and `meth526_finalize.py` admit terminal evidence.
Exact argv, revisions, original dependency/wire hashes and resources are stored.

## After this result

If eligible, separately derive and freeze a bound-refinement-only continuation
on retained525 coordinates, with full old floating reduction errors still
covered. A smaller projector bound does not authorize merely scaling down old
distance bars. Preserve109 physical queries BYTE and do not recompute them.
Only after qualified ALL candidate order inequalities can18 previously
unconstructed queries be constructed/physically checked under a new protocol.
No source labels before an admitted immutable geometry plan.

If exact representation error or ties still stop the recipe, report the failure.
An explicitly different feasible-point selector answers another question and
cannot retroactively admit525. Required new source-function semantics/labels,
source-weight-preserving redundant functions, prospective priced fallback, all
six/rare1% gates, useful distinct n/RAM, original winner AND normalized mass
cost, compact active core, actual C/DRAM, fresh donor quality AND SAME>=50/s and
actual other families/scales/~100B remain open. A tighter numerical certificate
alone is not transferred pretrained capacity or goal completion.
