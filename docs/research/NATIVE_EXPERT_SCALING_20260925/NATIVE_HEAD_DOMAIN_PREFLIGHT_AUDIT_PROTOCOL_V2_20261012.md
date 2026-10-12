# Native head-domain initial moments: independent full audit

V2 adds complete original stored native score comparison, with no tolerance or
initial gate change. Source V1 was frozen but never executed; kept intact.
Every577249896 original F32 native score is compared to promoted-F32 real-domain
logits. Recompute original native KL with full teacher probabilities and verify
stored deltas/IDs. Exact-real identity: loss difference is log E_p exp(delta)
minus E_q delta, so absolute difference<=max(delta)-min(delta). Verify per-row
bound with1e-9 numerical slack; retain maximum raw score/KL differences and ID
changes. This identifies whether1e-5 guard was stricter than actual rounding;
it does not silently promote the original false flag or assert an interval bound.
Bind full original score extents before/after. Same900s/4GiB limits.

12 October2026. Verify producer report/terminal/binding SHA and all input/output
extents before and after. No new fitting, optimization, source model or GPU.
Same complete4422 FIT/4386 DEV label scope and actual F32 head/bias/epsilon.

Actual source BF16 head/gamma/multiplier read independently; source logits sum
eight256-wide products, teacher log probabilities via full logaddexp reduction.
Moments and predicted expectations accumulate sixteen4096-row vocabulary chunks
plus tail, unlike producer full-vocabulary BLAS. Current code/norm reconstructed
from original tF32; logits use explicit head and bias decomposition. Loss computed
directly as full q*(logq-logp), not sufficient-moment objective subtraction.
Every8808 teacher moment/entropy/ID/code/loss/gradient verified. Full half-ball
linear minimizer explicitly constructed per row, scalar fsum computes supporting
lower instead of producer closed support formula. Check minimizer norm/lastsign.

Array tolerance norm(x-y)/max(1,norm(y))<=1e-9; code1e-12; scalar delta<=1e-8
timesmax(1,abs(expected)); probability mass1e-11. Original sphere1e-12 and
native KL delta1e-5 remain;the V2 diagnostic stores its false flag and per-row
deltas. Audit verifies the flag and every delta;it does not override the gate.
Independent first-row two-scale finite derivatives
same eps2e-6/2e-7 and relative2e-6 gate. Both producer flags must also pass.
Full summary/weights/counts/timing arithmetic checked; raw cost timing remains
unqualified, no independent benchmark inference. Lower is F64 numerical, not
interval certified; no capacity floor admission or native optimized-code upper.

Deadline900s/OS4GiB/no subprocess; terminal even on failure. Freeze own source/
protocol before execution, bind exact completed report/terminal hashes in launch.
No partial deadline results admitted; no restart or mutation of consumed namespace.
PASS admits reproducible sufficient moments/initial objective and approximate CPU
cost for choosing a separately frozen bounded algorithm; whole goal stays active.
