# M406: ONE regularized full-basis input map with unchanged validation gates

Freeze before map outcomes.405 acquired96 SAME source/prefix pairs: ALL13
apparatus PASS, but decoder0/1/4/5 failed768-direction conditioning at1e-6.
No unregularized maps were fitted. New variable: full768 ridge, avoiding
unbounded inversion of weak directions while preserving capacity beyond32.
This is not relaxed validation or a repair to the frozen405 scientific outcome.

Reuse ALL96 exact405 trace identities, fresh SHA,18/6 book split/all4 consumed362
cases. These are calibration books, not new original-relative quality. Fit F64
development-centered source256 X -> source128 Y using SVD; reproduce retained
405 singular ratios (rtol1e-9/atol1e-14). ONE penalty per bank determined solely
from development: lambda=(1e-5 * largest singular value)^2. Matrix V @ diag(s /
(s^2+lambda)) @ U.T @ Ycentered. Affine means unpenalized. No penalty/rank grid,
hyperparameter tuning or validation selection. The1e-5 scale is predeclared
before new prediction errors, not chosen from validation. All768 columns used.

Export F32 full matrix and two means. SAME local eligibility as404/405, EVERY
bank: validation squared error<=0.5 times prediction by developmentYmean alone;
median row relativeL2<=0.25;95th<=0.50; finite and F32/F64 prediction relativeL2
<=1e-5. Retain mean-only/identity/F64 controls. Effective ridge degrees of freedom
is descriptive only. Fail closes THIS fixed full affine ridge interface before
actual source-function/output-space/selector/model construction. Success licenses
only NEW source-function response and output-space tests. Neither is quality,
useful384 choices, inherited rate, DRAM or another-family proof.

BLAS1, MAIN<=2min /checkedRSS<=1GiB; CPU only; no model/source-weight reads,
network/GPU/T4. Twelve matrix/mean bytes28,385,280, matrix coefficients7,077,888;
NPZ overhead/hash separate. No measured engine cost. Bind committed controller,
protocol/helpers/405 raw and all192 traces; retain first result/failure.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth406_switch_full_ridge_input_map.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth406_switch_full_ridge_input_map_result.json
```
