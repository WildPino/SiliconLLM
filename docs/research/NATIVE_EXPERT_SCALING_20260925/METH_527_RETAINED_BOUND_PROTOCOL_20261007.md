# METH527: scalar bound refinement on ORIGINAL525 distance records

7 October2026. Prospective code/protocol freeze BEFORE binding, row energies,
refined bounds, diagnostics or numerical controls. Goal ACTIVE/INCOMPLETE.
[525](METH_525_ROUTER_NEUTRAL_GEOMETRY_RESULT_20261007.md) remains CLOSED;
[526](METH_526_EXACT_PROJECTOR_RESULT_20261007.md) admits a tighter certificate
on the SAME saved Q and original C. This inquiry changes error information only.
Original targets, nominal distances, tie identities, anchor UIDs and all109
qualified physical queries are preserved. No projection or query replay.

## Question and fixed decision

Can complete outward forward bounds certify ALL127 original nearest-target
orders? Original109 orders must remain resolved. Retain the central D2 of every
original representative, intersect two valid error bars by taking their minimum,
and require selected D2+bar strictly below EVERY competitor D2-bar. Endpoint
arithmetic is outward. Original sorting remains (central D2, neuron index).
The original targets are halfway into the opposite-sign feasible interval.
No selecting another neuron or changing target/alias rules in this inquiry.

Four eligibility gates: exact original G/ALL source row energies/outward bound
contract; ALL nominal identities unchanged; ALL109 orders preserved; ALL127
orders resolved. Decision is
`ELIGIBLE_ALL127_RETAINED_ORDERS_FOR_NEW18_PHYSICAL_QUERY_CHECKS` if all pass,
otherwise `RETAINED_NEAREST_TARGET_BOUND_REFINEMENT_CRITERION_CLOSED`.
Even eligibility is only a prerequisite for a separately frozen new query study;
it does not qualify the18 previously unconstructed queries or source functions.

## Bound original data and new exact row energies

Bind admitted525/526 binding/raw/audit/retention/windows, complete main output
inventories, terminal receipts, unchanged payload size/mtime, exact three foreign
tracked SHA, runtime and committed scientific source. Admission525 SHA
78db99d3c33ae8afa85d0f9c89df7679852927e28c93fc21db4e0316bffc66f3;
526 SHA50f5faa4bd1d605d41d85313c720affae2e9c42e13c9cce62181beba62253294.
All old output bytes are hashed before main, before audit and after both.
Use original525 saved panel[3072,7] and status for each e1..127, original radius
and y_norm. All2560 omitted representatives/status0 and512 retained/status1
are bound; no status2/3/4 assumed away after observation.

Original G F32[768] offset5056186368/3072B, SHA
23e7f70694ef54901fd7a3010ed5228e756a156adc2ff85264a69531a57be97f.
Read ALL127 original WI I8[3072,768] code extents,299630592B, before/after science.
Total128 used extents299633664B. No SI/WO/old888MB canonical inputs are used.
Original h_j=WI_j*G is exact binary64 (I8<=127, F32<=24 significand bits).
The NEW information is an exact norm bound H_j, replacing transient absolute
projection norms by inequalities. No hQ, yQ, rank/SVD/QR, or old geometry occurs.

Main decodes exact G=v/2^p by as_integer_ratio. Prospective caps p<=64,
all v nonzero, maximum |v| bit length<=52, WI>=-127. Stop with retained fault
on unsupported input; never widen after values. Exact row energy

    E_j=sum_k WI_jk^2*v_k^2; ||h_j||=sqrt(E_j)/2^p.

Since |v|<2^52, E<768*127^2*2^104<2^128. Split v^2 into SIX fixed20-bit limbs
(120bits, covers104). Each I64 dot is bounded by
768*127^2*(2^20-1)<2^44, with nonnegative exact integer operations. Recombine
Python ints; ALL390144 energies must fit unsigned128 bits. Main performs
1797783552 limb product terms. Auditor independently decodes original F32 bits,
directly sums299630592 arbitrary-integer square/product terms with original I8
codes, and verifies EVERY retained energy. No main math import or subsampling.

For each E, m=ceil_integer_sqrt(E); H is LEAST F64>=m/2^p, verified by exact
Fraction comparisons including predecessor. No floating row norm is reused.
Header32B `<8s6I`: M527ENR1,width16,E128,H3072,denominator2p,unsigned_bits128,
reserved0. All energies little unsigned16B, e0 allzero/not queried.
File6291488B. Five F64[128,3072] .npy arrays store H,new/intersection error,
lower radicand and lower root; one U8 array stores status (0not queried/hinge,
1new certified,2original retained). No pickle or dynamic integer wire width.

## Complete scalar forward bound

Exact target uses Pc, the true orthogonal source row projector. Q is the saved
binary64 representation; PB526 bounds ||Pc-QQ.T||2, OQ526 bounds||Q.T Q-I||2.
Both include exact represented-matrix proof, not QR proximity. With U=2^-53,
gamma(n)=n/(2^53-n), embed gamma by least upper F64. Every positive addition,
product/division is rounded then nextafter(+infinity); denominator subtraction,
positive lower square and root are rounded downward. All arrays must be finite.
Constants gd=gamma769,gn=gamma770,gp=gamma1040,g2,g3 cover ORIGINAL525 reductions,
including both projection contractions, dot/norm, divisions and scalar products.

Here r,yn,c,t,kn are ORIGINAL computed radius, norm, panel c/target/normal norm;
H is the new exact upper h norm. Upper bounds B,N,A refer to norms of ORIGINAL
computed vectors b,n,a; these vectors are NOT reconstructed. delta quantities
bound discrepancy from exact Pc geometry. T=128*(1+OQ) bounds||abs(Q)||2^2 via
Frobenius norm and qop=1+OQ bounds||Q||2^2. All following operations are outward:

    Y=yn/(1-gn); B=r/(1-gn); N=(1+g2)/(1-gn)
    A=(qop+gp*T)*Y
    da=PB*Y+gp*T*Y
    db=da+g2*(Y+A); dr=db+gn*B
    PH=(qop+gp*T)*H
    dk=PB*H+gp*T*H+g2*(H+PH)
    dkn=dk+gn*kn/(1-gn)
    dc=H*da+gd*H*A
    de=dc+r*dkn+kn*dr+dr*dkn+g3*(abs(c)+r*kn)
    dt=(2*de)/2

dt covers the same half opposite endpoint's rounding/error; doubled endpoint
surplus is preserved. Require den=kn-dkn>0. The original normalization n=k/kn
and scalar u=(t-c)/kn have forward bounds:

    dn=(dk+dkn)/den+g2*N
    Ua=up(abs(fl(t-c))/kn)*(1+g3)
    du=(dt+dc)/den+Ua*dkn/den+g3*Ua
    Aa=N*B*(1+gd)
    dal=dn*B+db*N+dn*db+gd*N*B
    W=(B+Aa*N)*(1+g3); TN=W*(1+gn)
    dw=db+Aa*dn+N*dal+dn*dal+g3*(B+Aa*N)
    dtn=dw+gn*W.

Ua bounds |original u|, Aa bounds |original n.b|, W bounds original tangent
b-(n.b)n, TN bounds its original computed norm. abs(fl(t-c)) receives nextafter
up BEFORE division; this also bounds exact subtraction. These inequalities use
Cauchy-Schwarz and norm submultiplicativity, retaining every dot/norm/rounding
contribution despite not rereading transient vectors. No bar division by631.

Require lower r-dr>0 and lower radicand

    Rlo=down(down((r-dr)^2)-up((Ua+du)^2))>0
    Slo=down(sqrt(Rlo))>0.

Store Rlo/Slo; independent audit checks EXACT dyadic Slo^2<=Rlo for every safe
row, avoiding an untested libm lower-root assertion. The original computed
sqrt(fl(r*r-u*u)) has upper S=r*(1+g3); argument error and root error obey

    darg=2*r*dr+dr^2+2*Ua*du+du^2+g3*(r^2+Ua^2)
    ds=darg/Slo+U*S.

The actual divided-difference denominator is sum of old/exact roots; Slo alone
is a conservative positive lower bound. Include sqrt rounding U*S. Then

    Csum=Aa*Ua+TN*S; Cup=Csum*(1+g3)
    dCross=Aa*du+Ua*dal+dal*du+TN*ds+S*dtn+dtn*ds+g3*Csum
    dD=4*r*dr+2*dr^2+2*dCross+g3*(2*r^2+2*Cup)
    NEWbar=4*dD.

Exact D2=2*r_true^2-2*(alignment_true*u_true+TN_true*sqrt_true)>=0.
Clamp of old computed D2 to nonnegative cannot increase its error relative to
this exact D2. Factor4 is unchanged525 surplus. These are bounds on ORIGINAL
floating transients, not newly recalculated central values. If any positivity
condition fails, retain original bar. If valid, intersect by min(old,new).
All old resolved orders must survive. The audit separately executes scalar
math.nextafter equations, verifies ALL325120 eligible rows, every exact lower
root square, status/intersection and127 complete competitor inequalities BYTE.
Numerical agreement is supported by the displayed forward-error derivation;
duplicating formulas alone would not establish the mathematical inequality.

## Controls, nonselecting diagnostic, scope

NEW energy control v=(2,1,1,...), WI=(-3,4,0,...), E=52,H=8. NEW scalar radical
control Pc projects onto third coordinate, y=(-3,4,1), h=(1,0,0), r5,ynsqrt26,
c0,kn1,t2.5. Exact D2=65-8*sqrt(75/4). Main runs actual bound functions;
independent audit verifies a rational interval for sqrt(75/4) from D2+/-bar
and exact Slo square. These controls are first executed after binding.

For EACH retained winner report a nonselecting relative displacement interval:
sqrt(max(0,D2-bar))/Yup through sqrt(D2+bar)/Ylo, with yn lower/upper from gn
and all endpoint operations outward. Auditor checks exact squares of both roots.
This describes ideal continuous geometry; it does NOT prove typical activation
reachability, the exact source F32 normalizer image or label informativeness.
No distance threshold is added after observation. The diagnostic will help
reassess whether another query or compact-function architecture is more useful.

Apparatus7 main/8 audit/5 terminal gates independent of four scientific eligibility
gates. Source-response/native/model/readout/corpus/T4/actual C/DRAM, new queries,
router score vectors, selected WI dots and old geometry/projection replay ALL0.
Actual new coefficient energies are charged numerical work. Original523 resource
failure and fixed525 closure remain explicit, regardless of527 outcomes.

## Price, sequencing and retention

Binding180s/main180s/audit240s, each512MiB, CPU10/BLAS1. Combined outputs32MiB;
energy6291488B + five F64 arrays15728640B + U8 array393216B + headers/JSON.
Independent direct arbitrary-integer energy audit is charged, not a hash-only
claim. Bound511 Python3.12.10,NumPy2.4.6,psutil7.2.2,threadpoolctl3.7.0;
isolated -I -B -X utf8, redirected empty cache. No concurrent scientific jobs.

Exact code/protocol commit BEFORE binding; binding commit BEFORE main; completed
main retained and committed BEFORE ONE independent audit. Exclusive527 namespace,
no overwrites/replays. Retain first faults and partial inventory BEFORE numbered
repair; unsupported/resource failure stays explicit. Fresh original used SHA and
payload stat before/after, ALL525/526 output bytes unchanged, foreign work retained.
Three actual executor exit0 PID+creation instances closed, typed UTC Event1000
queries available, positive controls179810/179791 present and relevant events0.
Finalizer only JSON/hash/process metadata,60s/128MiB. Actual exit receipts stored.

Run meth527_prepare_binding.py, meth527_refinement_main.py --binding-sha SHA,
then meth527_refinement_audit.py --binding-sha SHA --main-sha SHA; terminal
meth527_windows_terminal.ps1 and meth527_finalize.py. Exact argv/revisions/SHA
and resources retained. No completed source/main/audit replay.

After ONE bounded refinement, reassess total goal. Even ALL orders do not require
immediate source labels: compact source-atom functions and priced uncertainty
fallback may be more direct. Useful much larger n/RAM, LUT winner AND normalized
mass, conserved pretrained capacity, compact active work, actual C/DRAM, fresh
whole donor quality AND SAME>=50 accepted IDs/s and actual families/~100B remain
open. This finite numerical certificate cannot complete or quantify the goal.
