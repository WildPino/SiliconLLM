# METH-206: annealed bias calibration for deployed hard routing

METH-205 reproduces METH-204 and finds a balanced pooled soft
surrogate at temperature 0.05 but severe deployed argmax load.
Exact q-vector repetition is below 2.344%, so it does not force
the 25% hot-parent failure. Change only the bias-fit temperature
schedule; preserve the exact saved projection and nine shared keys.

Bind the METH-204 router/result and METH-205 result hashes. Use the
unchanged METH-204 collection, structural bypass, parity and six
fit/reserved/source evaluation cells. Before recalibrating each layer,
reconcile its saved-router raw/chat metrics on the same first 1,024
draws with METH-204. Start from its stored parent biases and apply
100 pooled soft-count updates at each fixed temperature
0.05, 0.01, 0.002 and 0.0004. Each update adds
`T*log((N_p/9+1)/(soft_count_pj+1))` and centers the nine parent
biases. Do not fit keys or select temperature using reserved/source
metrics. Save all four per-layer calibration-stage hard-load summaries.
Export projection, unchanged keys and terminal bias with exact readback.

Use every original METH-204 content gate unchanged: per-cell/layer
load ratio <=1.25, hot-parent share <=25%, coverage >=4,000,
standardized selected-score advantage >=0.05 and raw-score argmax
agreement >=15%. Stop before reserved/source if either fit cell
fails; stop before source if reserved fails. Only a six-cell pass
licenses native CPU routing cost followed by matched specialist
training and untouched quality. The saved keys' signal is not proof
of useful child functions.

Local RTX 3060, six host threads, <=30 minutes, <=20 GiB RSS,
<=10.5 GiB allocated GPU memory, <=1 GB stored result/router.
Stop on binding/parity/reconciliation/readback/nonfinite/budget
failure. Preserve apparatus outputs and failures. No T4 or B training.
