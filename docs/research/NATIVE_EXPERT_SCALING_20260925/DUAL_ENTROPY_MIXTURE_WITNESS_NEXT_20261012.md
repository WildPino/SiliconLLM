# Strengthening the reader diagnostic with entropy-mixture dual witnesses

12 October2026. **PROPOSED/UNIMPLEMENTED/UNBOUND**;current all-label audit must
complete first. New variable:dual distributions outside the current exponential
family,not more unchanged projected-descent iterations. Current native upper
DEVcaseKL.04835 is useful encoder-room evidence,but all numerical lower bounds
remain0. That does not establish whether further optimization can reach budgets.

## Algebra independent of optimizer convergence

For teacher q,m=A^T q,entropy H(q),and any simplex distribution r:

`L_q(v)=KL(r||softmax(A v))+H(r)-H(q)+(A^T r-m)^T v`.

Therefore over K={||v||<=16,v_last>=0},

`min_K L_q >= D(r)=H(r)-H(q)-support_K(m-A^T r)`.

Let p be a recorded candidate distribution and g_p=A^T p-m.
The existing point-gradient bound is the particular witness r=p,with
`D(p)=H(p)-H(q)-16 sqrt(||g_first256||²+min(g_last,0)²)`.
It is not the strongest possible dual witness when optimization is incomplete.

Choose r_lambda=(1-lambda)p+lambda q,lambda in[0,1]. Exact moment relation
A^T r_lambda-m=(1-lambda)g_p gives

`D(lambda)=H(r_lambda)-H(q)-(1-lambda)*support_K(-g_p)`.

D is concave in lambda,D(1)=0. Its interior may be positive even when D(0)<=0.
For strictly positive q,p,the left derivative at1 is
`D'(1)=-D(0)-KL(p||q)`.
Thus KL(p||q)>-D(0) yields a positive interior witness in exact arithmetic.
This uses reverse KL/tail information that the current weak point support does
not expose. A fixed lambda ladder can discover an actual witness without proving
that lambda search or primal code search reached an optimum.

These bounds concern the real augmented-head half-ball;the actual positive
hemisphere is a subset. They are not automatically interval certificates or
global F32-machine floors. All moment/entropy/numerical margins require full
independent verification before making a reader-domain obstruction claim.

## Planned finite test and data scope

Use all8808 existing labels,current actual head and recorded source references.
Two frozen candidate families are available:final relaxed current p(v),and final
actual native best-code p from stored F32 scores. Any simplex p is allowed in the
dual identity;native rounding does not require p to lie in the real family.
For the native family recompute g_p=A^T p-m;do not reuse a real-code gradient.

Prospective lambda ladder [0,.25,.5,.75,.9,.99,.999,1],same for every label.
Recompute full source q from raw H/actual source head and normalize explicitly;
retain normalization differences against qualified m/H. Evaluate all mixture
entropies/full moments/supports,select best per-label dual witness. Selection is
an oracle mathematical witness,not fitting a head or a held-out learned model.
No new source-model forward or data query;no T4. Native versus real source paths
must stay distinct. Fixed CPU/GPU precision/caps are still to be implemented.

Handle zero/underflow probabilities explicitly:entropy0log0=0;reverse KL or its
derivative formula cannot be treated as finite after silently clipping q. Use
stable source log probabilities and separately bound/declare any numerical-tail
approximation. A numerical derivative diagnostic is not a certified proof.

Independent audit must verify all lambda entropy/moment/support values,chosen
witnesses/source distributions and original case/domain-weighted lower bounds.
Positive witnesses change reader/decoder choice;zero witnesses only leave this
dual family uninformative. No repeated long primal run solely to shrink weak gaps.

## Link back to the converter

If a current fixed-reader obstruction is isolated,conditional low-rank decoder
directions or changed source-derived basis can be considered with explicit active
cost and native norm transport. Body experts inside an unchanged fixed head cannot
silently increase its output image. A better reader still requires causal core/
expert transfer and same useful artifact50. No variant implemented/admitted here.

Exact source-head moments can support a changing source-derived decoder without
replaying full teacher probabilities:if fixed source head is A_source and target
logits=A_source y,compile m_source=A_source^T q,H(q). Then arbitrary decoder y
uses logZ(A_source y)-m_source^T y-H(q). This is sufficient within that fixed source-
head column space,not arbitrary new readers. Current257-moments do NOT support
such decoder changes;2048-feature compilation and active readout cost remain
unimplemented. This provides a concrete algebraic path if the new witness isolates
missing reader directions,while the full causal pipeline requirements remain.
