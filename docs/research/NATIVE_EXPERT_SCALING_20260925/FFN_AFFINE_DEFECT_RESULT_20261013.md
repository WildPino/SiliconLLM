# Full-output affine FFN pilot and nonlinear defect

13 October2026. Goal ACTIVE/INCOMPLETE.

## Qualified affine failure

[Frozen pilot](FFN_AFFINE_PILOT_PROTOCOL_20261013.md),source b22ee53,
binding b240d5f,producer57ef5c7,[report](ffn_affine_pilot_result_20261013.json).
[Complete audit](ffn_affine_pilot_adjudication_20261013.json) passes every
432 saved prediction,all144 source-proxy controls,nine bank/derivative
controls,independent direct routing and chain-rule evaluation. Maximum
stored F32 prediction relative error2.644e-8;finite difference1.563e-9.
Auditor455.640s/OS935,526,400B,terminal0/owned process absent.

All nine variants fail the frozen FIT eligibility criteria. No count is
selected;no whole-model substitution is authorized by these local results.
Errors below compare with actual source BF output,case-weighted means and
worst case. Every output retains all2048 coordinates.

| Site | Cells | FIT mean | FIT worst | DEV mean | DEV worst |
|---|---:|---:|---:|---:|---:|
|0|1|.757809|.870410|.743898|.866897|
|0|16|.725057|.824017|.713485|.824601|
|0|64|.621696|.711117|.621892|.676703|
|12|1|1.041351|1.084408|1.050489|1.100292|
|12|16|.926558|.974657|.960052|1.034672|
|12|64|.836879|.887009|.927598|1.028852|
|23|1|.285766|.435334|.303446|.408352|
|23|16|.142465|.224331|.147208|.207198|
|23|64|.126993|.212719|.134470|.199768|

This rejects the tested Euclidean1/16/64-cell construction. It does not
prove that more or differently placed affine cells cannot work. The pilot
evaluates implicit source Jacobians;no physical ternary bank,scalable routing
or CPU speedup is demonstrated. Original quality gates remain unchanged.

## Exact omitted information

For u=Ux,g=alpha Gx,uc=Umu,gc=alpha Gmu,du=u-uc,dg=g-gc,s=SiLU,
the first-order feature is a=s(gc)u+uc*s'(gc)dg. Its exact remainder is

Rgate=uc*[s(g)-s(gc)-s'(gc)dg],
Rcross=du*[s(g)-s(gc)].

F(x)=beta D[a+Rgate+Rcross]. Each term is projected through the actual
source down matrix before measuring its energy;their cross inner product
is necessary because projected contributions can cancel. This identity
locates omitted nonlinear information without changing the model.

[Diagnostic protocol](FFN_CURVATURE_SPLIT_PROTOCOL_20261013.md),source
87570c1;[independent audit protocol](FFN_CURVATURE_SPLIT_AUDIT_PROTOCOL_20261013.md),
source3c0a24e. The pre-execution norm-indicator correction is explicit in
the protocol;no diagnostic observations preceded it. All48 histories,
three sites,existing64 FIT cells/routes. [Diagnostic](ffn_curvature_split_20261013.json)
and [full independent audit](ffn_curvature_split_adjudication_20261013.json)
PASS. Maximum reconstruction relative error6.820e-16;maximum metric difference
5.552e-16. Producer291.562s/OS1,308,585,984B,auditor307.984s/
OS1,378,631,680B,both terminal0. Shared source decoder/NumPy BLAS scope
is explicit;no independent CUDA arithmetic is claimed.

DEV case means against the smooth source proxy:

| Site | Affine error | Gate term norm | Cross term norm | Affine + exact cross error | Full quadratic error |
|---|---:|---:|---:|---:|---:|
|0|.621901|.097974|.610236|.097974|.566415|
|12|.927585|.264810|.992441|.264810|.506929|
|23|.134429|.149612|.147118|.149612|.233021|

The cross term carries a large projected contribution at0/12;exactly
restoring it leaves the gate term. At23,the projected terms partially cancel:
mean cross inner product/reference energy is-.013526. It is-.098310 at12
and+.002198 at0. Do not square the mean norms to reconstruct mean energy;
the per-case normalized energy identity was independently checked instead.
At23,removing only the cross defect increases mean error,and the complete
quadratic Taylor correction also increases it. More polynomial orders or
separately retaining one term are not justified by these results.

## Decision and exact next action

The original smooth source function reconstructs exactly when BOTH exact
remainders are retained. Next test whether a small FIT-selected set of source
neurons supplies enough of their combined remainder. [Defined next question](FFN_EXACT_REMAINDER_SUBSET_NEXT_20261013.md)
keeps all output coordinates and current64 cells;budgets32/128/512/2048 plus
all4608 control. The FIT score is a heuristic,not an oracle or an optimality
bound. Producer/auditor/binding remain to implement and freeze before runs.
No new subset observation has occurred.

[Session closure](ffn_affine_defect_session_closure_20261013.json) checks all
four producer/auditor receipts,owned PID absence,every distinct source/input/
output hash,the complete441-file E family1,711,100,700B and protected files.
The closure command has its own terminal receipt. No failed variant erased
and no gate relaxed. Affine producer449.188s/OS1,016,119,296B.

## Scope toward the goal

The known-corpus native recurrent quality bridge remains qualified;its
source FFN/attention/head still use GPU. Final CPU cost,compact source
functions,ternary experts,structured CPU IDs AND masses,actual DRAM,useful
large n,conversion costs,multiple donors and same-artifact useful50 remain
open. All-neuron curvature evaluation is a diagnostic,not a cheap expert.
