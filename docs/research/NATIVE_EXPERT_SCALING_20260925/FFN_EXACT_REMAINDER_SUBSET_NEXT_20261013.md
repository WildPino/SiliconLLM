# Next question: conditional exact nonlinear remainder coverage

13 October2026. HISTORICAL PROPOSAL,NOW EXECUTED/fully audited.
[Result/current next question](FFN_EXACT_SUBSET_RESULT_20261013.md) supersedes
the proposed statuses below. No selected candidate admitted.

## Why this variable changes the experiment

The affine pilot omits two exact terms. Keeping the entire quadratic Taylor
approximation still does not yield a generally small remainder. An exact
source-neuron remainder can retain both terms together,including their
cancellation,instead of adding more polynomial orders. This is a conditional
full-output affine operator plus selected original nonlinear neurons.

For routed center mu and neuron i:

r_i(x)=uc_i*[s(g_i)-s(gc_i)-s'(gc_i)(g_i-gc_i)]
       +(u_i-uc_i)*[s(g_i)-s(gc_i)].

Candidate F_S(x)=F(mu)+J(mu)(x-mu)+beta*D[:,S]*r_S(x).
S=all4608 reconstructs the smooth source exactly. This is a control and
can cost more than the original FFN;it is not a speedup or desired outcome.
S small is the actual hypothesis. The affine expert includes the first-order
action of omitted neurons;selected neurons supply only missing nonlinear
information. Full2048 output coordinates are retained.

## FIT-only selection and complete measurement

Keep existing64 Euclidean FIT cells/routes. Proposed neuron budgets
32/128/512/2048,plus4608 identity control. For each cell select by
mean_FIT[r_i(x)^2]*||beta D[:,i]||^2,descending score,ties by source index.
The scores measure individual projected energy,not joint optimality:cross
covariances can change the quality of a subset. Empty FIT cells require a
predeclared global-FIT fallback. No DEV-specific selection or reranking.
Preserve rankings and actual FIT count per cell. Audit both the FIT rule
and complete predicted outputs,not only energy scores or aggregate errors.

Use actual operands at0/12/23/all48 histories,with the same smooth proxy
control versus source BF output. Apply unchanged local eligibility gates
FIT mean<=.10/worst<=.20;smallest passing budget selected on FIT. DEV mean/
worst/domain gates unchanged. No RESERVED,new source model,GPU or optimizer.
Whole-model quality is still mandatory for an eligible local operator.

Before execution freeze producer,auditor,binding,byte budget and terminal
ownership. Compute storage from every complete predicted trajectory rather
than guessing;ordered E family and bounded CPU time/OS use. All-neuron control
and complete audits remain required even if a small subset fails.

## Algebraic resource accounting before a physical implementation

One full affine2048x2048 operator has4,194,304 coefficients. Selecting r
neurons adds two r-by2048 input rows and2048-by-r output columns:6,144r
coefficients,plus centers/scalars/IDs. Original FFN has28,311,552 coefficients.
For r=32/128/512/2048,the totals are4,390,912/4,980,736/7,340,032/16,777,216.
For r=4608,total32,505,856 exceeds the source. These are logical counts;
precision,DRAM bytes,conversion work,routing and measured CPU latency are
not supplied by this assay. Ternary quantization adds its own error budget.

If neither moderate budget qualifies,report the coverage failure and inspect
the full-output residual covariance/sign changes before choosing another
representation. Do not hide it by DEV tuning or relabeling redundancy as
transferred knowledge. Even a successful subset must still address the
24-layer shared core/attention/head and end-to-end20ms budget for useful50.
