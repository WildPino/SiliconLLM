# METH-319: full-width additive decoder exceeds native cost budget

Freeze `3f2b1fb`, exit0,9.985s total. Exact318 executable/source/spec retained;
fresh Q4/BF16-embedding bindings match. Removing OMP_PROC_BIND allows startup
on this host. This supports a settings incompatibility for the observed binary,
without identifying an OS/runtime root cause.318 empty-log failure preserved.

| Fresh process | Repetition medians (ms) | Max/min | OS peak RSS (bytes) |
| --- | --- | --- | ---: |
|1|52.817 /53.768 /55.526|1.0513|3,216,236,544|
|2|39.609 /38.527 /38.124|1.0390|3,216,306,176|
|3|48.670 /50.679 /52.237|1.0733|3,216,244,736|

All nine medians and24 fixed-input medians exceed14ms; pooled median ratio
1.456465>1.10. Complete542.987MB addressed-weight accounting still passes,
but bytes alone do not predict useful execution speed. No physical DRAM or
causal/accepted decode rate was measured.

All three full selftests are EXACTLY equal:16,960 scalar integer/scaled output
rows,13,240 stored code and13,240 palette bank-edge checks,32,131 exhaustive
decoded-weight/input cases,128,524 tile lanes,14,641 mixed adjacent pairs,
all25 scalar source-router/selection/gate controls. Float relativeL2
3.578e-8 and64 actual Q6 head rows1.531e-7 pass. Bad-bank/bias/palette/index/
S8 saturation/packed-head faults detected;1536 source embedding values exact.
All90 output/route hashes repeat exactly; input union22..33 source-sized banks
per sparse layer. Fully allocated3,186,946,048bytes;10,275,979,264 synthetic
stored coefficient positions,1,428,619,264 active encoded coefficients plus
actual original head/router. No donor knowledge in generated palette weights.

[Full raw result](meth319_additive_i8_execution_repair_result.json)
SHA `61f5cd62e35cd90e96b75ce74704d9c845741509f669304103a0e1ffc0a93f97`.
Logs under `results/native_expert_scaling/meth319_additive_i8`;
same binary under318 folder. Scope input-ready per-layer contexts, all matrix/
norm/router/SwiGLU/mixture/head/embedding operators, no causal attention/cache/
layer composition. Gates: descriptor/numeric/capacity/repeated hashes PASS,
14ms and pooled stability FAIL.

**Decision:** close unchanged full-width decoder BEFORE source-aware book
training,640-bank allocation or any quality/rate promotion. No wait/affinity
tuning. New candidate selection may investigate donors with inherently much
smaller active core/FFNs, retaining real pretrained expert banks; their
metadata/weights/whole-model quality and cost require NEW evidence. That does
not reopen GigaChat's failed compact fields or unqualified generic port.
