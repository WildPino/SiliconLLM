# Next routing question: source-sensitive input geometry

14 October2026. PROPOSED/NOT EXECUTED. Only after the complete sketch audit
and explicit current-environment custody reconciliation. No new admission.

The input-energy PCA rule fails FIT at0/12 even with512 directions. The
full-source importance oracle succeeds with the same active counts. This
does not show that512 directions cannot predict useful IDs:the PCA objective
preserves input energy,while the decision concerns nonlinear source functions.
Change the metric rather than silently enlarge rank or choose using DEV.

## Exact source differential

For each source neuron,combined remainder
r_i=u_i*s(g_i)-s(gc_i)*u_i-uc_i*s'(gc_i)*(g_i-gc_i),
u=Ux,g=alpha Gx,s=SiLU. At fixed routed center:

grad_x r_i = a_i U_i + alpha b_i G_i,
a_i=s(g_i)-s(gc_i),
b_i=u_i*s'(g_i)-uc_i*s'(gc_i).

With w_i=||beta D[:,i]||^2,FIT token-weighted source derivative metric is
M=E_FIT sum_i w_i*(grad r_i)^T*(grad r_i).
It is PSD and targets variation of individual projected remainder functions.
This is a proxy for importance selection,not the derivative of the squared
energy ranking or an optimal output-loss metric;cross-neuron covariance
and discrete top-k boundaries remain important.

Compute it without allocating one2048x2048 matrix per token/neuron:
M=U^T diag(w E[a^2]) U+alpha^2 G^T diag(w E[b^2]) G
 +alpha*(U^T diag(w E[ab])G+G^T diag(w E[ab])U).
Only FIT source functions used to fit metric. Verify per-neuron derivative
against central finite differences,direct metric actions on fixed vectors,
symmetry/PSD/source identity and all averaged coefficients. No DEV metric.

## Balanced input reconstruction

Let Sigma=E_FIT[delta^T delta] be the uncentered moment from the qualified
PCA assay. On its numerically supported eigenspace,form
T=Sigma^(1/2) M Sigma^(1/2). For its top eigenvectors V_k,
encoder E_k=Sigma^(-1/2)V_k,decoder B_k=Sigma^(1/2)V_k.
delta_hat=delta E_k B_k^T. These factors are dual,not assumed orthogonal:
E_k^T B_k=I. They minimize the stated linear rank-k weighted reconstruction
objective E||M^(1/2)(delta-delta_hat)||^2 on supported Sigma. This surrogate
does not prove minimum FFN error or ranking accuracy.

Freeze a relative Sigma eigenvalue cutoff before observations,report dropped
trace/rank and inverse conditioning. Save all Sigma/M/T spectra/factors,
source-gradient coefficients/finite differences/action certificates. Offline
project source U/G using B_k,then select IDs using only delta E_k and those
projected rows. True selected source functions and full outputs remain exact.

Keep64 existing centers/active counts512,2048,32/ranks64,128,256,512.
FIT chooses smallest passing rank;DEV tests that choice,no promotion. All
previous thresholds/source controls/complete prediction and ID audits remain.
Freeze scripts,binding,exact E payload and bounded CPU resources before runs.
No GPU/optimizer/new whole-model substitution/RESERVED. This changes the
importance feature objective;it is not a repeated token-only GG or fixed
output-bank factorization. Actual compiled cost/precision/DRAM and whole-model
quality remain required;middle active width2048 is still a cost concern.

If this also fails,retain it and reconsider nonlinear predictors,source-
informed partitioning or joint adaptation before claiming that cheap routing
or more experts is impossible. Final engine/multiple donors/large useful n/
same useful50 and conversion costs remain open.
