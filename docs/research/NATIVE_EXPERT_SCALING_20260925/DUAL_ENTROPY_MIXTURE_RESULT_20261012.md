# Entropy-mixture dual witnesses: positive but far below quality budgets

12 October2026. COMPLETE/full independent audit PASS. Goal ACTIVE/INCOMPLETE;
no owned run live. New dual family strengthens zero point lowers on ten labels;
it proves no numerical reader obstruction against the current quality budgets.
Zero elsewhere does not prove optimumzero or head adequacy.

[Producer](../../../benchmarks/native_expert_scaling/dual_entropy_mixture_witness.py),
[frozen protocol](DUAL_ENTROPY_MIXTURE_PROTOCOL_20261012.md),
[result](dual_entropy_mixture_result_20261012.json).
All4422 FIT/4386 DEV,full65537 words,two fixed candidates(final relaxed real p,
actual saved native p),eight lambda values[0,.25,.5,.75,.9,.99,.999,1].
Source full F64 head/operator q from raw H/source weights/gamma/multiplier/epsilon.
No primal updates/head fitting/new source generation/GPU/T4/RESERVED/longalign.

## Algebra and observed lower

For any simplex r,L_q(v)=KL(r||p_v)+H(r)-H(q)+(A^T r-m)^T v.
Over real half-ball K radius16,last>=0,the dual lower is
H(r)-H(q)-support_K(m-A^T r). Choose r=(1-lambda)p+lambda q;explicitly normalize
represented probabilities and retain normalization drift. All mixture entropies/
257 moments/supports/raw/display lowers stored. Display=max(0,raw-1e-8).

| Numerical real-domain lower | FIT | DEV |
|---|---:|---:|
| Mean per-case | 6.932626e-7 | 1.664479e-7 |
| Worst label | .00118297753 | .00062604605 |
| Positive labels | 5/4422 | 5/4386 |
| Original quality lower-budget violation | False | False |

Current oracle upper remains native caseKL.017254/.048346 andcase disagreement
1.5324%/2.3003%,all budgetsFAIL. These tiny lowers cannot explain that full residual
or establish that the real reader cannot meet .01case/.1worst/.05every-case budgets.
An uninformative certificate family is not a compatibility theorem. Bounds are
numerical real geometry,not interval/globalF32-machine or autonomous-model floors.

FIT chosen positive witnesses:two lambda.75,two.99,one.999;one chooses actual-
native p family. DEV positives:one.9,two.99,two.999,all relaxed-real family.
All other selected displayed ties choose family0/lambda0 as frozen first-argmax.
No chosen model weights or held-out learned-model validation inferred from selection.

## Numerical information controls and complete audit

All source/candidate represented probabilities strictly positive:zero words0;
no clipping,underflow-tail log bounds remain negative infinity. Stable-logit reverse
KL/derivative masks all admitted for numerical diagnostic;no interval proof.
Explicit probability/mix masses,source moment/entropy drift,lambda1 endpoints and
unequal-spacing concavity checked at original tolerances. Direct selected moments
verify producer's linear mixture reconstruction,not a silently reused gradient.

[Independent auditor](../../../benchmarks/native_expert_scaling/dual_entropy_mixture_audit.py),
[protocol](DUAL_ENTROPY_MIXTURE_AUDIT_PROTOCOL_20261012.md),
[adjudication](dual_entropy_mixture_adjudication_20261012.json).
Every8808 source row and**140,928 family/lambda rows**:source partial256-feature
sums,head partial32-feature sums,full logaddexp normalization,4096-vocabulary
entropy/expectation blocks. EVERY lambda moment directly multiplied over full
vocabulary,not producer linearity shortcut. Explicit minimizing support endpoints,
all entropies/masses/moments/lowers/derivatives/zeros/selected/case/domain values
verified. Max independent relative1.18379e-12,scalar0;full before/after hashesPASS.
INDEPENDENT_STORED_PASS. Previous oracle/candidate evidence reused only through
complete qualified report/binding/extent chain. No producer-equation replay.

## Cost and custody

CPU6threads/affinity0..5,batch64;full raw/q/native inputs retained unchanged.
Prepare5.516s/closed0;source/protocol ee96017,
[launch](dual_entropy_mixture_launch_20261012.json)40ab689.
BindingSHA068525a07392f90fdd9aeb4f4766fdfdf91df453c0418cf90f42bb6d1f1662b8.
Capture26766 CLOSED0,PID18480/create1791827570.9167583,result301.253s/terminal301.337s,
OS3,544,940,544B. ReportSHA
aeb0530c989f2f85ea1fde5dba3e7360aa3738689ffe77cf0331b10f9074d50f;
terminal93349dd700ff7245ea1dbf958daa5792afeb5922c8277c08b71b502c0362e087.

Audit source/protocol49779cd,[launch](dual_entropy_mixture_audit_launch_20261012.json)
0f81aa3;22384 CLOSED0,PID31512/create1791828613.985088,result440.055s,
OS3,481,612,288B;adjudicationSHA
7b69cda06846aa540608d6780ff08dd62e25f62a8e3c6b4d68e85e2b5aa1da11.
All900s/6GiB/512MiB capture and900s/6GiB audit caps PASS.
Output full grids/moments/source-control packets369,261,792B;timing admission=false.
[Session closure](dual_entropy_mixture_session_closure_20261012.json) contains all
receipt hashes/actual terminal seconds/creation fields,preserved engine/foreign
hashes/process inventory. No owned run live;no mechanical fault found.

## Selected next toward an actual converter

[Compile source-head2048-feature moments](SOURCE_HEAD_FEATURE_MOMENTS_NEXT_20261012.md)
PROPOSED/UNIMPLEMENTED/UNBOUND. Current257-feature moments freeze the reader;
source-head moments allow a changing source-derived decoder with exact full
categorical supervision. This enables joint decoder/history-to-code training,
instead of more unchanged per-label oracle descent with weak floor certificates.
Actual native quantization/export and causal/full-state/own-history tests remain
required. No conditional decoder/core implementation or n/DRAM/multiple-source/
same useful artifact50 admission. Follow the causal transfer boundary after this
compiler;do not turn successful statistic compilation into the final goal.
