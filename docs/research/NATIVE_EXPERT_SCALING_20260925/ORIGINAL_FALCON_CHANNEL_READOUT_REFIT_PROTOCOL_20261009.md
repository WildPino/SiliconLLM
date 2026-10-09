# Fixed-channel closed-form readout repair, before new observation

9 October2026. Previous384-channel fixed-source-column output has actual local
FAIL even with full-source denominator. That does not test manipulating output
weights to recover correlated omitted contributions. This is the new variable.
No source-model/optimizer/native/RESERVED/T4 calls. Original engine/Adam25 and
previous failed records remain unchanged. Not an original-engine variant yet.

Reuse EXACT FIT-selected channel IDs/constant denominator from successful terminal
of original_falcon_channel_transport_20261009;no channel reselect/grid/dose.
Sites0/12/23,ALL24 FIT+24 DEV histories/12 domains. Same384 channels,2 layouts,
3 denominator modes. Source y/state256/gate/nonlinear generators retained.

For each site/layout/mode,generate same BF16 encoded features X(T,384),reference
captured BF16 output Y(T,2048). Use equal-FIT-case F64 CUDA Grams/cross-moments:
G=mean_cases(X^T X/T),C=mean_cases(X^T Y/T).
One ridge lambda=1e-6*trace(G)/384,declared before observation;no lambda search.
Beta=(G+lambda I)^(-1)C via CPU F64 symmetric eig;new output=X(F32)*Beta(F32).
Previous FIT scalar alphas are NOT applied;linear map incorporates fitted scale.
This is a constrained linear estimator on actual source features,not arbitrary
free features or a general rank384/capacity theorem. No DEV fitting.

18 actual closed-form fits. Record F64 eig/normal equation residual<=1e-8;
F32 exported-coefficient equation residual<=1e-4. Save G/C/F32 Beta/eigens in
numeric NPZ per site. Fit loss quadratic must match direct F32 prediction loss
over complete FIT histories to absolute delta/target energy<=1e-4.
Cache FIT encoded BF16 features once,then evaluate fitted map;no source replay.
DEV map predictions computed once. All144 full source gate/norm/out_proj baseline
reconstructions must match captured output to relative RMS<=1e-4.

Same predeclared engineering budget as the prior assay:EVERYsite DEV output
mean<=10%,case worst<=20%,domain mean worst<=20%,centered mean<=10%.
Full source denominator is an oracle diagnostic and cannot admit a layout.
If more than one deployable combination passes,minimize maxsite mean,then sum,
then lexical layout/mode. Failure retains CHANNEL_READOUT_REFIT_BUDGET_FAIL;
do not close all mixed features/nonlinear decoders or relax thresholds.
Record full/centered/label-position errors,domain tails and chosen combination.
No whole-chatbot/state96/P256/depth/native/quality/speed admission from this.

One separately frozen family900s/reserve90s,heldworker+launcher OS6GiB,
GPU4allocated/5reserved GiB,output256MiB/log4MiB. Larger output allowance pays
for18 actual G/C/coefficients/eigen records (~191MB),not increased active work.
Reuse the original channel assay's held launcher/schema with experiment field
CHANNEL_READOUT_REFIT_V1 and new worker/protocol/binding/results namespace.
Pre/post all byte hashes,principal runtime binaries,exact PID/ctime/exit/OS held
through exit/GPU peaks/first faults. Single owned job,declared publisher exception.
No overwrite/restart solely on observation timeout. Preserve/adopt partial fits
and missing-only comparisons with a new binding on interruption.

```text
original_falcon_channel_readout_refit.py --bind --out <new binding>
original_falcon_channel_readout_refit.py --launch --binding <binding> --binding-sha <SHA> --freeze <commit> --directory <new namespace> --out <new result>
```

Python312 -I -S -B -X utf8;repo root. Freeze before observations.
