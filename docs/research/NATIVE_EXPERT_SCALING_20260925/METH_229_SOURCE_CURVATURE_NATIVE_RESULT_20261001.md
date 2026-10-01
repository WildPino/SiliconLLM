# METH-229: combined affine and source-nonlinear C component passes

The [protocol](METH_229_SOURCE_CURVATURE_NATIVE_PROTOCOL_20261001.md),
runner and C source were frozen at `9603083`. Session31425 completes,exit0.
All24 complete unrounded combined-function gradients agree with full
source Jacobians: worst relative Frobenius1.112099e-7 vs<=1e-5. Stored
center worst relative L2=1.394568e-8 vs<=1e-5. All120 binary segments
read back exactly; every source layer has a distinct affine fixture.

Native384 stored-output oracle comparisons have median relative
L2=1.343775e-7,worst3.452093e-7,passing1e-4/5e-4. Three actual combined
24-layer passes:8.941789,8.786087,8.874827ms/token; median**8.874827**
passes<=10ms,six CPU threads. GPU work is synchronized before timing.

**Decision:** combined operator passes; freeze source-anchored nonlinear
selection/readout learning at H2048 and E16/E160. This302,862,360-byte
fixture uses first2048 original source rows, not selected/fitted functions
or a trained bank. No nonlinear accuracy, route, large-n DRAM/LUT, full
accepted rate or multi-family quality is established. The whole530.020MB
selected-ledger calculation remains an unmeasured complete-path hypothesis.

[Raw result](meth229_source_curvature_native_result.json),SHA256
`f2e94ed991048709ff4c61dbb456ab95077865a36e3dc574dfc3a343d4330dbc`,
retains every identity/segment/source hash and numerical/timing row.
Ignored local binary `results/native_expert_scaling/meth229_source_curvature_fixture.bin`
SHA=`23287d3cd98378939f93885e5062737d9f08facd5d477ad20fd9523410ea2c56`.
Executable SHA=`8035049d788e602d7c10927f8bd3bdd469b8e0bcb4689687691171a018b7bb1f`.
Runtime13.750s after imports,ending RSS2.030GB,GPU peak128.9MB,local3060,
no T4. The cuBLAS context initialization warning is preserved. No job active.
