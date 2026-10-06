# 490: post-hoc decomposition of admitted error

6 October 2026. The completed R1 science and independent audit have already
failed the fixed recipe. This is descriptive arithmetic, not a new fit, fresh
cohort, independent repetition, altered scientific gate or feasibility search.
Inputs are admission/RAW/RET/binding, twelve exact retained histories and the
already audited prediction/diagnostic records, each checked against its SHA.

For the branch of the source winner, sign t is +1 for right and -1 for left.
Let z* be the retained standardized F64 function value, zC the native F32 logit,
qC the emitted F32 probability, and qS the original selected conditional mass.

```
L(z) = log(sigmoid(t*z)) = -logaddexp(0,-t*z)
E_ideal = |L(z*)-log(qS)|
E_native = |log(qC)-log(qS)|
D_export = |L(zC)-L(z*)| <= |zC-z*|
D_round = |log(qC)-L(zC)|
|E_native-E_ideal| <= |log(qC)-L(z*)| <= D_export+D_round.
```

The derivative of L has absolute value at most one. These are real-algebra
inequalities. Derived F64 comparisons use an explicit 2e-12 tolerance; they
are not an outward-rounded general certificate. Record all-state/role counts,
error quantiles and ideal errors larger than eta plus observed numerical drift.
Also describe the undivided log(1.01) root threshold without promoting it to a
new leaf criterion: later levels could cancel signed errors, and no later
candidate exists in this inquiry.

For every development bank use the original omega, y=qS_right and z*:

```
BCE(y,sigmoid(z*)) = H_Bernoulli(y) + KL(Bernoulli(y)||Bernoulli(sigmoid(z*))).
```

Preserve target endpoints, distinguish weighted mean KL from uniform errors,
and verify the final loss against the retained fit. Describe all 33 observed
losses, gradient BEFORE update32 and magnitude of that update. These do not
certify stationarity or the affine optimum, nor justify another Adam sweep.

One helper execution, exclusive output and first-fault retention. Budget30s,
256MiB peak working set,128KiB output,BLAS1. No source/native/fit/audit replay,
no quality/rate/DRAM claims. Goal remains active/incomplete.
