# Router derivative support and normalized group displacement

10 October2026. Algebra plus small producer-metadata observation; actual
gradient/archive-bit adjudication PENDING. Current24-case producer and consumed
protocol remain immutable. This is a possible correction to the converter,
not useful-chatbot, CPU/GPU parity or large-n evidence.

## Question and evidence reused

Does a mathematically inactive router row receive numerical gradient noise,
which our per-row normalized displacement then promotes to a full step?
Reuse the qualified first58-token gradient, its stored GPU routes and BOTH
already completed campaign proposals. No neural/native/GPU replay is needed
to answer the stored-support question. The first new inspection reads only
1,567,404 bytes of metadata/source/route files,1.6821934s,OS peak122,273,792B.
[Metadata capsule](categorical_router_support_metadata_20261010.json), tool
`benchmarks/native_expert_scaling/inspect_categorical_router_support_v2.ps1`.
All input hashes/extents match producer metadata before and after the read.
This does NOT independently read actual gradient tensors or displacement bits.

The first inspection script failed before group-stat analysis: PowerShell
enumerated empty HashSets through a pipeline, yielding an empty container;
`.Add` then saw null. Original script preserved; v2 uses an explicitly allocated
object array. [Mechanical fault record](categorical_router_support_metadata_first_fault_20261010.json).
The fault belongs to the metadata tool, not the model or campaign.

## Exact local algebra

For fixed selected set S and p=softmax_n(s), define P=sum_(i in S)p_i and
m_i=p_i/P. Assume P>0 and finite scores. In real arithmetic,

    m_i=exp(s_i)/sum_(l in S)exp(s_l).
    dm_i/ds_j = m_i*(1[i=j]-m_j), j in S;
                0,                  j outside S.

This is piecewise support with fixed IDs, not a derivative through selection
changes. For any upstream vector v, even a coarse adjoint from the ternary/AQ
bank, the selected-mass VJP has exactly zero coordinates outside S.

In the all-n softmax then gather/renormalize implementation, h=dL/dp satisfies
h_j=0 outside S and h_i=(v_i-sum_l m_l*v_l)/P inside S. Consequently

    rho=sum_j p_j*h_j=0,
    dL/ds_j=p_j*(h_j-rho).

Floating arithmetic can leave rho nonzero, giving an unselected residual
`-p_j*rho`. Other implementation errors must still be excluded; the stored
counts alone do not locate a particular kernel or prove this is their cause.
No auxiliary all-n router loss or weight decay is part of this frozen rule.

Over the full causal computation graph, union all selected IDs for each site.
A row outside that union has zero ideal direct router weight/bias derivative.
This statement does not assume bank/core gradients are exact derivatives of
the discrete C engine. It isolates one algebraically cancelling subgraph.

## Why the proposal can amplify a tiny residual

Consumed `original_categorical_trust_step.py:43` groups bank router weight
and bias by expert row. It computes radius R=max(||w||_2,1e-5*sqrt(dim)) and

    delta=-alpha*R*g/||g||_2, for ||g||_2>0;
    delta=0,                 for ||g||_2=0.

Before F32 storage, every nonzero row gets norm alpha*R. For g=lambda*r with
lambda>0, the step is independent of lambda. The update has no continuous
extension at g=0: different vanishing residual directions give different
finite displacements. F64 norm calculation does not restore structural zeros
to nonzero F32 gradient entries. Merely shrinking the noise need not shrink
this displacement. An arbitrary magnitude threshold would need a principled
error bound and could suppress real small gradients; none is adopted here.

## Observed producer metadata, not yet independent tensor proof

GPU full-history selected unions by site:218,237,225,229,247,227 of1152.
Thus5,529 expert rows lie outside those unions. Weight and bias constitute
11,058 separate proposal groups. Every excluded group reports nonzero gradient
norm AND moved coefficients for BOTH trial alphas.

| Alpha | Excluded groups moved | Excluded coefficients moved | Maximum displacement/radius |
|---|---:|---:|---:|
| 0.0001 | 11,058 /11,058 | 1,420,538 | 0.00010005784296255074 |
| 0.001 | 11,058 /11,058 | 1,420,906 | 0.001000057891982378 |

Excluded weight-row norms range1.34096e-10..1.58195e-8; bias norms
1.67158e-14..8.40091e-10, as reported by completed producer group statistics.
The slight radius overshoot is compatible with final F32 storage; full audit
must recompute it rather than treating that compatibility as proof.

This suggests inactive-row movement can scale with total n even at fixed k.
It does not establish its quantitative importance for loss, routing margins,
generalization or current cross-runtime discrepancies. CPU/GPU compare the
same stored weights; this update hypothesis does not directly explain their
different forward paths. Current quality remains poor.

## Stored adjudication and subsequent decision

1. Finish frozen campaign and exhaustive audit, including actual90 gradient
   tensors, both reconstructed92-master proposals and group-stat recomputation.
   No failure-based shortening of custody or quality assessment.
2. After closure, use stored routes and audited gradient/archive coefficients
   to count outside-union nonzero entries and actual moved groups/coefficients.
   Check no all-n auxiliary objective or omitted routing event can justify them.
   Reuse the full campaign audit evidence; do not rehash/reload the entire model
   merely to produce the same answer. A small additional observation must name
   its missing evidence and freeze inputs/cost before execution.
3. If support leakage survives those checks, the smallest causal intervention
   can reuse the SAVED gradient: set router weight/bias gradient rows outside
   the full selected union to exact zero before the same grouped proposal.
   Other gradients and alpha stay fixed. Compare one NEW finite C candidate
   with the corresponding already retained candidate and baseline. This
   isolates inactive-row movement without a new neural backward. It does NOT
   repair selected-row derivatives or the upstream gradient entering the core.
   Prediction to predeclare: excluded rows copy exactly; whether native loss
   and future margins improve is an open result, not an acceptance assumption.
4. A separate NEW adjoint variant would require one new backward at the same
   existing state/history, preserving C baseline, head, inputs and forward
   route arithmetic. Verify outside-union derivative zeros and actual native
   finite displacements, then broader history/held-out/own-history quality.
   No gate relaxation or claim from zeros alone. Do not perform both variants
   by default; choose based on the audited defect and remaining uncertainty.
   Exact budgets and protocol remain to be specified before either launch.

One candidate, UNIMPLEMENTED/UNTESTED, keeps actual forward mass but substitutes
the algebraically equivalent selected-score derivative:

    actual = p.gather(-1, ids) / p.gather(-1, ids).sum(-1, keepdim=True)
    selected = softmax(scores.gather(-1, ids), -1)
    mass = actual.detach() + (selected - selected.detach())

With finite positive mass, the parenthesized difference is bitwise numerical
zero; the intended forward is preserved while gather confines score-gradient
support to S. Bit identity must be measured in the real checkpointed graph,
not assumed from this snippet. This is a changed floating adjoint and a NEW
variable. It does not make the whole quantized learner an exact C derivative.
Direct selected softmax in the FORWARD would change its rounding path and is
a separate runtime proposal, requiring separate original-C qualification.

Even successful correction would leave dense converter masters/gradient
storage O(n), exact structured CPU lookup/mass cost, real DRAM, useful capacity
transfer and same-artifact quality/speed unproven. This algebra addresses a
specific converter defect rather than closing the full goal.

Conditionally, if excluded row radii stay bounded below as n grows, their
pre-rounding total squared displacement is alpha^2*sum_excluded R_j^2,
which grows with excluded count. This follows from the rule, not a measured
large-n result. A support-correct proposal removes this inactive contribution;
consulted support can still grow with history length and data coverage.
