# METH519: dev-only local matched WI/query geometry and ideal capacity

7 October 2026. PROSPECTIVE: freeze before any519 fitting, projection or capacity
observation. Startup HEAD420807e; merge has not changed this research checkout.
Five519 source drafts exist; no numerical job or observation. Goal incomplete.

## Question, changed variable and reused evidence

518 proves the stored517 global query-only rank32 basis fails the byte gates
even with ideal projected angles: natural possible ceiling0.069002%, only
2.119739 potentially omitted rows per3072. Change ONE variable: a different
rank32 subspace for EACH original parent, fitted to BOTH normalized native WI
directions and that parent's DEVELOPMENT query directions, fixed mixture1:1.
No rank/mixture grid, selector search or parent cherry-picking. This is additional
source-derived conditional dictionary storage, not additional learned knowledge.

Use admitted504/517 QUERY contracts, admitted518 obstruction, cached norms and
original380 bank11 native WI/WO. ALL128 physical parents,3072x768 WI,17540 UIDs
(11721 dev/5819 already consumed),19962 occurrences,ALL64 consumed books and
rare dev-count1..4/5..15 stay included. These consumed cases are not fresh held-out.
Original source7,415,217,408 distinct parameters/12x128 parents remains unchanged.

## Development fitting and observation order

For nonzero native I8 rows w_j and original global-A16 queries q_i define

    M_e = mean_nonzero_WI(w_j w_j^T / ||w_j||^2)
          + mean_nonzero_dev_e(q_i q_i^T / ||q_i||^2).
    J_e(V) = trace(V^T M_e V), V^T V=I, rank32.

In exact real arithmetic the top32 eigenspace maximizes this mean weight plus
query energy. It does not maximize each pair, imply exact reconstruction, fix
angles or imply model quality. No-dev parent uses only the WI term. Zero norms
are excluded from normalization but retained in physical capacity/cost accounting.
No consumed query, sign, hidden activation or output label enters fitting.

Main uses normalized rows with sqrt norms, NumPy2.4.6 eigh, descending stable
eigenvalue order; each vector's largest absolute coordinate (first tie) is positive.
Freeze B_e=round-to-even(256V_e) into I16. ALL128 bases and their physical WI
projections are written before ANY consumed query projection/capacity evaluation.
Store top32 F64 vectors, all768 scalar eigenvalues per parent and fit receipts.
Check selected residual ||MV-Vlambda||/||M||<=1e-10 and orthogonality defect<=1e-10.

Independent audit rebuilds the moments as x^T(x/norm2)/count, without the main's
sqrt-normalization path; verifies stored selected eigenpairs, canonical signs,
rounding, dimensions and exact basis bytes. Main moment SHA is provenance for
its temporary F64 moment, not an independently reproduced SHA of the alternate
rounding path. Partial receipts do not independently prove full-spectrum optimality.
Certificate correctness depends on actual integer Gram/radii, for ANY stored B.

## Exact certificate capacity algebra

S=256, G_e=B_e^T B_e, R_e=max_i sum_j |S^2 I-G_e|_ij <S^2;
delta_e=S^2-R_e>0. For W=||w||^2,Q=||q||^2,tw=wB,tq=qB,
pw=||tw||^2,pq=||tq||^2, define

    RW=W*S^4-delta_e*pw >=0,
    RQ=Q*S^4-delta_e*pq >=0.

The517 upper endpoint is S^2*dot(tw,tq)+R_e*ceil_sqrt(pw*pq)
+ceil_sqrt(RW*RQ). Even the most negative possible projected dot cannot make
this nonpositive unless (1-R_e/S^2)*(pw/(S^2 W)+pq/(S^2 Q))>=1.
For positive norms this is EXACTLY

    delta_e*W*pq >= RW*Q,
    equivalently pq/Q >= RW/(delta_e*W).

Equality is possible; W0 or Q0 counts possibly free-zero without division. This
is NECESSARY ONLY. Possible counts are upper bounds on certifiable row skips,
not sound masks. Actual angle and ceil roots can reduce them further.

Main exact Fraction thresholds sorted per parent, zero-weight sentinel-1, row-ID
ties, bisect_right on pq/Q. Audit independently sorts by integer cross-products,
binary-searches each query and verifies both boundaries. ALL393216 native rows
and17540 query norms/projections/counts/cost records are verified, not sampled.
2500 main rank1 controls and2500 independent nonorthogonal rank2 controls.

## Physical format and finite range proofs

Global32B <8s6I: M519L001,128,768,3072,32,256,reserved0. Each parent:

| Field | Bytes |
| --- | ---: |
| <IIQ parent/dev UID count/R_e |16|
| 768x32 I16 B |49152|
| 32x32 I64 exact Gram |8192|
| 3072x32 I32 exact native WI projection |393216|
| 3072 U64 projected row norms |24576|
| 3072 U64 RW |24576|
| 3072 U32 floor_sqrt(RW) |12288|
| Per-parent total |512016|

Complete local index32+128*512016=65,538,080B. ALL128 dictionaries are charged,
including uncaptured parent0. Original WI+WO fallback605,945,856B retained.
Solver vectors25,165,824 raw B, spectra786,432 raw B; NPY capacity17540x24B
holds possible U16/remaining U16/minimum byte U32/Q U64/pq U64. Fit/terminal
JSON and NPY headers stay within combined128MiB; no full query-row mask matrix.

|w_i|<=128, |q_i|<=32767, |B_ik|<=256. WI projection absolute prefix sum
<=768*128*256=25,165,824<I32; query projection prefix sum<2^33. F64 integer
products/partial sums are exact below2^53, then audit uses independent I64 MACs.
Exact R<S^2 gives pw<=131072W, pq<=131072Q, within signedI64 (including each
square). RW<=W*S^4<I64, RQ<=Q*S^4<2^72; products/threshold cross-products fit
U128 and are checked independently. Recover W=(RW+delta*pw)/S^4 with exact
divisibility. Original A16 query is never independently requantized by subspace.

## Fixed full price and decision gates

Prospective complete executor introduced logical bytes per UID506084:
global+local header48, local basis49152, original-query energy read1536,
projected WI/pw/RW/root454656, mask384, full-predicate scratch28,
projected-query write256, RQ write16, root write8. Baseline original WI
3072*(768+4)=2,371,584B; each unresolved original row costs772B.
Gram bytes are stored and audited; inference uses the admitted scalar R_e.
Capacity diagnostic's24B record does not replace the charged28B full work.
Transform768*32 MAC, prospective row prefix3072*32 MAC, norm/radius/root/
U128/predicate/gather/mask/scratch/fallback work remains charged. Current519
does NOT evaluate row prefix angles, emit masks, or measure an executor's rate.

Ideal minimum remaining rows=3072-possible. Five frozen optimistic gates:
consumed UID AND natural occurrence mean minimum byte ratio<=.75; natural
p95<=1; ALL64 consumed book means<=1; full fallback+index ratio<=1.20.
Failure closes THIS fixed local matched32 representation/certificate/price before
actual-angle/C work. All pass permits a later actual-certificate/hidden-byte/full
operator-cost screen only. No automatic C/whole promotion, quality or50/s claim.

## Conversion price, resources and terminal admission

Weight moments at most128*3072*768^2=231,928,233,984 MAC; dev query moments
11721*768^2=6,913,327,104 MAC; sum238,841,561,088 MAC.128 eigensolves of768x768,
WI projections9,663,676,416 MAC and query projections431,063,040 MAC are new
mathematics, distinct from full WI/WO function evaluation (zero). Audit rebuilds
all moments and projections once with the alternate arithmetic. No model/native/
corpus/optimizer/GPU/T4 calls. No large old completed computation replay.

Commit sources/protocol first; binder<=180s/512MiB, commit binding before main;
sole main and independent audit<=300s/2GiB each, combined new outputs<=128MiB.
CPU10/BLAS1, isolated511 Python3.12.10/NumPy2.4.6/psutil7.2.2/threadpoolctl3.7.0,
-I -B -X utf8 and empty pycache. Fresh used data/runtime/science and ALL512
bank11 WI/WO code/scale extent SHA before numerical work. Whole7.54GB payload
SHA is retained with current size/mtime; it is not claimed freshly rehashed.
Three foreign tracked bytes preserved. Exclusive namespace and hard timer;
preserve first faults/partials before literal numbered repairs, no completed phase
replay. Report actual limits, elapsed time, OS peak, hashes and process instances.

Typed UTC Event1000 query availability, positive records179810/179791, zero
matching PID/create-time faults, actual exit0 and three exact process-instance
closures required before admission. Independent audit includes ALL physical
inverse/capacity/views/books/rare/economic decisions. Compact transfer, same
artifact whole donor-relative quality AND50/s, useful much larger n, CPU LUT
winner+normalized mass, actual DRAM and additional actual families/~100B remain
open irrespective of this finite local screen.
