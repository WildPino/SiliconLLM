# Recovery geometry: output support, correction direction and common mean

9 October2026. Two saved-only CPU families COMPLETE/resource PASS. No source
forward/generation, learning, C export, native or RESERVED query. They reuse
the four actual source x/y and calibrated/256-step recovered responses from
[local function recovery](CHATBOT_SOURCE_FFN_LOCAL_RESULT_20261009.md).

## 1. Learned correction versus required residual

For r=y-y_before and c=y_after-y_before, norm(r-c)^2=norm(r)^2+norm(c)^2-2dot(r,c).
All eight F64 identity defects<=1.8e-16 and independently recomputed aggregate
errors match retained reports within1e-12. The diagnostic source-target oracle
scalar dot(r,c)/norm(c)^2 cannot be used at deployment.

| Site / DEV | Learned error | cosine(c,r) | Oracle scalar | Oracle residual error |
|---|---:|---:|---:|---:|
|0 / short|.539483|.301006|.706577|.534907|
|0 / long|.538734|.298628|.688070|.533394|
|23 / short|.191030|.931807|1.186343|.177167|
|23 / long|.224401|.905480|1.184995|.212901|

The actual site0 learned correction points poorly toward the required DEV
residual; an amplitude-only change cannot meet .10 error on these operands.
Site23 direction is better but its amplitude oracle still fails .10.
This does not rule out different corrections, training or functions.

## 2. Linear support of the actual FIT operands and responses

Thin F64 SVD of uncentered274x2048 FIT X/Y; numerical cutoff1e-10*s_max,
energy rank is the smallest count covering>=99% squared singular energy.
Each numerical rank is274; small BF16 modes are retained under that cutoff.

| Site / matrix | Stable rank | Energy99 rank | DEV short outside full FIT span | DEV long outside full FIT span |
|---|---:|---:|---:|---:|
|0 / X|1.82087|181|.646023|.607821|
|0 / Y|32.26932|203|.864042|.875930|
|23 / X|1.99845|220|.548287|.634265|
|23 / Y|1.07790|29|.114422|.141242|

Outside-span values are normalized Frobenius residual NORMS, not energy
fractions or missing-knowledge fractions. The source inputs have limited linear
FIT coverage at both sites. Final-site output concentration differs markedly
from the first-site outputs. This is descriptive: a nonlinear model is not
constrained to output the FIT span, and these data do not prove causation or a
generalization lower bound. Every singular value and energy99 projection is
retained in the result JSON. This result motivated a separately frozen followup.

## 3. Mean versus varying output: NEW FIT-only constant control

Adaptive followup froze9fce840/c27e6a31 before observations. For each case,
norm(error)^2=N*norm(mean(error))^2+norm(error-mean(error))^2. The same exact
decomposition separates the actual squared-error reduction into mean/varying
gain. A single equal-case average of the two FIT mean residual vectors defines
the prospective constant control; DEV never fits it. The F64 control has no
deployed BF16/operator contract and is not an exported candidate.

| Site / DEV | Source mean energy fraction | Before L2 | Learned L2 | FIT-only constant L2 | Mean share of squared-error gain | Centered L2 before -> after |
|---|---:|---:|---:|---:|---:|---:|
|0 / short|.097992|.560921|.539483|.559989|17.31%|.559082 -> .539394|
|0 / long|.014517|.558897|.538734|.560999|10.21%|.557370 -> .538978|
|23 / short|.900364|.488125|.191030|.198627|93.55%|.590428 -> .466969|
|23 / long|.854142|.501664|.224401|.235961|91.25%|.602311 -> .491925|

Centered L2 uses the case's centered source output as denominator. It is
different from the registered whole-output error and does not replace or relax
that failed gate. Constant control preserves varying-error energy within1e-12;
all F64 decompositions and retained-error checks PASS. At site23, most gain is
mean correction; some varying response is recovered, but its remaining error
is .467/.492. At site0, the small gain is mostly varying yet total error remains
.539. Do not describe the58% final-site L2 gain as broad conditional knowledge
preservation. This finding is compatible with seeking common/private response
components, without admitting the old unexecuted compact common-bank proposal.

## 4. Resources, provenance and decision

| Family | Freeze / binding | Wall seconds | Worker seconds | Held OS bytes | GPU bytes |
|---|---|---:|---:|---:|---:|
|Geometry|51b1c97 /b6290fe6|22.203|18.484|512,696,320|0|
|Centering|9fce840 /c27e6a31|4.985|3.828|453,840,896|0|

Exit0, all input/resource checks PASS; both jobs terminal. Geometry result
SHA7f8f9b06affded0f4fdccff20918d799bf11926a17e5b2fd3b32dc86a8ef9411;
centering1e43c2af3a0554726b6705e5b95122e60daae9465176f851772480ddedc7711d.
Centering saves two2048-coordinate F64 constant controls plusJSON:3files45,723B.
Bindings/terminal JSONs/logs beside this record provide exact commands, runtime
scope and all packet hashes. Tools:[geometry](../../../benchmarks/native_expert_scaling/chatbot_source_ffn_recovery_geometry.py),
[centering](../../../benchmarks/native_expert_scaling/chatbot_source_ffn_correction_centering.py).

Selected next uncertainty is broader function coverage under unchanged original
engine arithmetic, with mean and input-varying fidelity measured separately.
Do not repeat the finite balance grids or prolong fitting on these two prefixes
by default. More full-source/native runs are gated on function recovery;
compact core/selection and useful large-n transport remain additional stages.
No engine kernel replacement or T4 job follows from these descriptive results.
