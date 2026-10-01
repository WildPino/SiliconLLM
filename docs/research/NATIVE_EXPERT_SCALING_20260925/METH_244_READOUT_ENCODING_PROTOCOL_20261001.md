# METH-244: frozen fit-only continuous/encoded error ledger

## Question and bindings

METH-240/242/243 all retain accurate absolute functions yet E160 loses
22%–24% to E16. Both source-precision and parent-value-only intercept
changes fail. Determine whether the **continuous fixed-fit solutions**
already lose useful count gain or whether readout encoding removes a
gain. This changes the next choice between function/fit coupling and
representation; it is not a new fit recipe or held-out result.

Bind METH-240 result/snapshot/source/capture/native/route identities and
METH-243 result SHA256
`ac40b7453cf6414df62911b7d3cf87b1aeb403b8a4005921506470345af25770`.
Read only first512 fit windows. No validation,new source,source compilation,
tau/count/codec change or checkpoint replacement.

## Exact replay and ledger

Replay all16 parent+160 child centered FP64 ridge systems with the actual
METH-240 stored priors and fixed parent feature variances,tau1024. Use
the same primal/dual solve,cast-to-FP32,mixed codec and corrected stored
intercept. Require every regenerated code/scale/index/BF16 escape/bias
exactly equal the frozen physical snapshot before interpreting metrics.
Normal residual<=1e-7;route/count/labels exact;actual separated mixed
operator's pooled E16/E160 fit scores exactly equal METH-240.

Retain nondeployable FP64 solution Wraw=Wp+delta and
braw=bp+gamma*meanR-delta*meanZ before rounding/encoding. Compute per-cell
fit outputs with (1)FP64 raw solution,(2)FP64 decoded stored coefficients/
bias,and(3)actual FP32 separated mixed accumulation. Let e=raw-target,
d=decoded-raw;record SSEdecoded=SSEraw+||d||²+2<e,d> with relative closure
<=1e-9. Record actual-vs-decoded arithmetic difference,prior error and
stored parent error on each exact child support. Pool each arm over the
same65536 inputs/energy;all176 rows retained. Raw solutions inherit
stored priors,not unencoded ancestors;declare this diagnostic scope.

Fixed decision: if raw E160 fit SSE<=90% raw E16 **and** encoded E160
penalty(actual-raw SSE/energy)>=50% of its positive actual E16 loss gap,
the diagnostic exposes continuous local count gain and licenses a separately
priced/frozen encoding change. Otherwise do not retry codec alone as the
count solution;change continuous hierarchy/function/fit coupling first.
Neither branch establishes held-out capacity,deployable quality or rate.
Any exact replay prerequisite failure makes diagnosis uninterpretable;
retain failure stage/rows and repair apparatus before scientific claims.

## Budget and command

One local3060 run,six threads,10min after imports,20GiB RSS,10.5GiB GPU.
176 replayed solves,JSON only,no new weights/T4/download/native timing.
Commit runner/protocol before execution;partial/failure rows preserved.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth244_readout_encoding_audit.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth244_readout_encoding_result.json
```
