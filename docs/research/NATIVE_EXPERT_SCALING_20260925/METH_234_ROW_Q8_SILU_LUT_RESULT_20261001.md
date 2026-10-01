# METH-234: full source coverage plus SiLU lookup passes the native prerequisite

Protocol/implementation frozen at `d7706ec`; session17088 completes,exit0.
All168 projection/bias segments are exactly equal to METH-232; only the
nonlinear operator changes to a shared513-sample FP32 lookup. All4864
original source units remain in every one of24 distinct source layers.

| Frozen prerequisite | Measurement | Limit | Result |
| --- | ---: | ---: | --- |
| Lookup scan max absolute SiLU error |.0002440028|<=.001|Pass|
| Segment reads |169 exact,168 controls unchanged|All exact|Pass|
| C/GPU LUT median relative output L2 |4.377542e-7|<=1e-4|Pass|
| C/GPU LUT worst relative output L2 |9.093106e-7|<=5e-4|Pass|
| Pooled original-source function SSE/energy |.0001943932 (0.01944%)|<=.01|Pass|
| Six-thread24-layer component median |**8.522295ms/token**|<=10ms|**Pass**|

All three fixed timing passes8.497492/8.522295/8.549601ms pass. Worst
individual layer source SSE/energy=.0005541079 (0.05541%),reported without
a separate layer gate. Source screen uses all256 existing states/layer;
numeric screen uses384 outputs. Lookup scan is discrete,not a universal
analytic bound. These comparisons jointly include row quantization and
lookup approximation against BF16-weight/FP32 source FFNs.

**Decision: native prerequisite passes; freeze full-feature conditional
transfer next.** This is an implemented changed nonlinear operator with
measured cost/fidelity, not a complete LLM or an E16/E160 learned comparison.
No original BF16 intermediate-rounding quality, held-out prediction,
generation/task, native routed bank, n-scaled DRAM/LUT or >=50 accepted
tokens/s is established. Timings across METH-232/234 are separate fixed
runs,not a statistically matched speed claim. METH-232 stays rejected.

[Raw result](meth234_row_q8_silu_lut_native_result.json),SHA256
`972a451f89fdbcc567b6f1f52df0e22bd2185f28d52cf3179b560b8ba8fba815`.
Local ignored fixture
`results/native_expert_scaling/meth234_row_q8_silu_lut_fixture.bin`,
314,894,364bytes,SHA256
`51f170fee96ac262881e14d72caddd58311f7bb8ab05d43a74f93e36fd72adf5`.
Shared table2052bytes,SHA256
`76431bc2595aaab8d441012fded3e658c31fc6f6af84b1498fac8c241e2ac805`.
C output SHA256
`b92a37e0c20aecbe7a3ef41a4d6953f3291df9d508f5b0290bc63cba5706cc9a`;
executable SHA256
`d71ecbb20d8c7931402f3df7dbb2a2b2947f626cbf6a9839aeb997bc4fe2b210`.
All source/script/binary/segment/layer/row hashes and metrics retained.
Runtime26.063s after imports,RSS1.680GB,GPU peak154.454MB,local3060,no T4.

Next: [full-feature output prior proposal](FULL_FEATURE_OUTPUT_PRIOR_PROPOSAL_20261001.md).
First qualify source derivative transfer into the nonlinear output basis
without a full affine branch; then separately freeze actual row-Q8 E16/E160
fitting/rotated controls. Sparse GigaChat/wider scales are still unqualified.
