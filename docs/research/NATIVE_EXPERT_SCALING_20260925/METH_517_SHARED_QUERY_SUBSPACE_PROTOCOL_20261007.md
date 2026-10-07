# METH517: shared rank32 integer input subspace with exact fallback

7 October2026. Prospective freeze. [Selected517](METH_517_SHARED_QUERY_SUBSPACE_NEXT_20261007.md)
changes directional information after516/513 exact but uneconomic certificates.
Previous goal turn: PROGRESS (implemented and independently admitted516 negative
economics). HEAD9cb4e81 at startup, three preserved foreign tracked SHAs only,
no live science job. Routine Graphify off; no agents/new resources/model/GPU.

## Question, domain and decision

Can one dev-derived shared rank32 basis certify enough WI zeros to repay its
transform, prefix dictionaries, exact radius predicates and unresolved full WI?
ALL17540 original QUERY UIDs (11721 dev/5819 already consumed),19962 occurrences,
64 consumed books/rare parents of original128 decoder bank11 ONLY. Original
Switch7,415,217,408 unique parameters/12x128 parents globally. Parent0 physically
included despite no captured UID. Source original WI/WO/p/hidden/max/A16 unchanged.
No all-bank/256/new held-out/other-family/large-n/DRAM or50/s claim from this screen.

ONE uncentred second moment of ALL dev q: integer covariance exactly summed in
F64 (<2^44 absolute sums), numpy.linalg.eigh, descending eigenvalues stable
returned-index ties, eigenvector largest-absolute low-coordinate pivot positive.
ONE fixed top32, no rank grid. Physical B=round-to-even(256*V32), I16; |B|<=256.
Eigensolver approximation chooses efficiency only; exact stored Gram determines
certificate soundness. Save covariance/all solver vectors/values and basis-stage
hash BEFORE projecting consumed queries or checking their hidden/sign labels.
Independent I64 covariance and residual/orthogonality<=1e-10 qualify solver
metadata empirically, not an exact algebraic low-rank reconstruction claim.

On all exactness and five inherited economics pass select priced C next; on
economic failure close THIS shared rank32/S256 certificate before C. First
Gram/range/resource fault invalidates admission; preserve before literal repair.

## Exact fixed-point algebra (no floating sign cutoff)

Let C=B/256, G=B^T B integer, R=max row sum |65536 I-G|. Reject geometry if
R>=65536. Symmetry gives rho=R/65536>=||I-C^T C||2. For original integer w,q:
tw=B^T w,tq=B^T q; pw=sum(tw^2),pq=sum(tq^2),A=tw^T tq.

    RW=||w||^2*256^4-(65536-R)*pw >=0,
    RQ=||q||^2*256^4-(65536-R)*pq >=0.

From the exact identity in Selected517,

    256^4 * w^T q <= A*65536 + R*sqrt(pw*pq) + sqrt(RW*RQ).

Final rejection iff `A*65536 + R*ceil_sqrt(pw*pq) + ceil_sqrt(RW*RQ) <=0`.
Ceil roots are exact unbounded integers: main isqrt(n)+(r*r!=n); independent
audit uses0 for zero,1+isqrt(n-1) otherwise. This more conservative integer
upper endpoint handles equality/zeros without epsilon. Non-rejected rows keep
full original WI/scales/ReLU/global max/A16; rejected cached values must be BYTE+0.
ALL full hidden/max/scales/codes and source parent/p unchanged, WO extent SHA
unchanged: finite exact composition with previously qualified source evaluator.
No new full WI/WO/model trajectory replay is needed or claimed.

Exact cheap necessary prefilter: L=floor_sqrt(RW)*floor_sqrt(RQ) <=sqrt(RW*RQ).
Candidate iff A<=0 AND -A>=ceil(L/65536). U64 product L is safe from the native
range bound; division/remainder avoids overflowing L+65535. ALL candidates run
both exact wide ceil roots/final predicate; others stay unresolved. No approximate
floating mask is used. Store floor_sqrt(RW) and recompute query floor root exactly.

|w|<=128, |q|<=32767,D768. Gram bound yields pw<=2*65536*||w||^2,
pq<=2*65536*||q||^2; both fit I64. RW fits U64/I64; RQ can require73 bits.
RW*RQ <=(768*128^2)*(768*32767^2)*256^8 <2^128;
L<2^64; Gram term/central scaled sum fit signed128. Native C would need a priced
U128 sqrt primitive; this Python screen uses actual arbitrary-precision ints
and counts that work. Projected row tw fits I32 even using the loose entrywise
bound768*128*256. Query tq fits I64. Absolute sum of prefix products is bounded
by2*65536*ceil_sqrt((768*128^2)*(768*32767^2))<2^53, so main F64 integer GEMM
is EXACT even for these projected integer operands; asserts integrality/ranges.
Independent audit uses I64 products/sums, not the main floating path.

Main2500 exhaustive rank1 small controls, independent2500 rank2 nonorthogonal
controls plus201 ceil-root controls, zeros/equality/sign/extremes, exact residual
identities. Source512 WI/WO code/scale extent hashes refreshed each phase,
605,945,856B complete original fallback. Whole7.54GB SHA/stat retained only.

## Physical format and price

Header40B `<8s6IQ`: M517P001,128,768,3072,32,256,reserved0,R. Shared B768x32
I16=49152B; exact Gram32x32I64=8192B. Numeric parent order:tw3072x32I32=393216B,
pw3072U64=24576B,RW3072U64=24576B,floor_sqrt(RW)3072U32=12288B.454656B/parent;
ALL128 complete index58,253,352B. Original full fallback stays stored/charged.
No coefficient quantizer approximation: stored tw is the exact integer projection
of original source WI into the actual encoded basis; representation redundancy.

Per UID introduced logical bytes=header40+B49152+original query norm read1536+
parent454656+bitmap384+work28+projected query write256+RQ write16+root write8
=**506076B**, plus772B per unresolved row. Baseline original WI2,371,584B.
Gram payload is verified/stored but no per-query full Gram traversal is needed,
only the charged header's R. ALL query transform768*32 MACs, prefix3072*32 MACs,
query norm768 and projected norm32 products,3072 U64 floor-root products and
comparisons, each candidate's TWO U128 ceil roots/Gram/final terms, mask words,
query-wide RQ/root, unresolved768 MACs per row charged in views. No speculative
cache residency or measured physical DRAM assumption. Original common WO/output
omitted from BOTH WI-only denominators; full physical storage included.

Five frozen gates: consumed UID mean AND natural consumed occurrence-weighted
mean<=.75,natural p95<=1,ALL64 consumed book means<=1,bank+index<=1.20.
Rare dev1..4/5..15 and all UIDs retained. No changed cutoff,rank,basis/role/book
selection or omitted transform/wide-arithmetic cost after observing outcomes.

## Budgets and execution

Commit all science/protocol before fresh binder<=180s/512MiB; commit binding
before sole main. Main and sole independent numerical audit each<=300s/2GiB;
all serialized new outputs<=96MiB (full fit metadata and physical index included).
Isolated511 Python3.12.10/NumPy2.4.6/psutil7.2.2/threadpoolctl3.7.0,
CPU10/BLAS1,-I -B -X utf8,empty isolated pycache. No new native/compiler/corpus/
model/T4 operations. New dev covariance/WI projection/current-query projections
are real mathematical source-derived work, separately counted from old function
evaluation. Batch128 avoids whole53,882,880-cell dot serialization. Hard deadlines,
exclusive namespaces, immutable faults before numbered repairs; no completed
main/audit replay. Fresh used inputs/runtime/science/extents and foreign SHAs;
final source stat/empty-cache/resources. Typed UTC Event1000 query availability,
positive record controls179810/179791,zero matching instance faults/actual shell
exit0/three exact PID-create-time closures before terminal admission.

Independent audit: dev-only physical basis/covariance/solver receipt, exact Gram
defect, all128 original I64 projected rows/norms/radii/floor roots, all17540
query/central/final predicates/masks/work/function bytes, independent occurrence
book/rare summaries and unchanged frozen decisions. No rate/quality promotion
without subsequent SAME whole-artifact proof, n/mass/DRAM/family gates.
