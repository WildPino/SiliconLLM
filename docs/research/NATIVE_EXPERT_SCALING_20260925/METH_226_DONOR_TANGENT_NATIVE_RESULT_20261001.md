# METH-226: source derivatives and full-input affine C operator pass

The [protocol](METH_226_DONOR_TANGENT_NATIVE_PROTOCOL_20261001.md), source
and runner were frozen at `bb3d9a0`; session25224 completed,exit0.
All24 original-layer analytic Jacobians agree with complete autograd
Jacobians: worst relative Frobenius1.993039e-7 vs<=1e-5, worst element
error/source peak1.940033e-7 vs<=1e-4. Stored-center worst relative
L2=2.532930e-8 vs<=1e-5. All48 binary segments read back exactly and
all24 full896x896 BF16 weight hashes differ.

Native384 outputs agree with decoded stored weights/bias: median
relative L2=8.620400e-8,worst4.706543e-7, passing1e-4/5e-4.
Three24-layer component timings are0.653896,0.591563,0.675367ms/token,
median**0.653896**,passing<=10ms with six threads. This38,621,208-byte
fixture contains source-layer tangents at existing token0 centers; it
does not contain a routed/learned bank or preserve nonlinear donor
accuracy by construction. No large-n ratio or full token rate follows.

**Decision:** apparatus/component pass; freeze the full-input E16/E160
conditional-function pair. The previous nonlinear common recipe remains
stopped. Complete-function error<=.01 and final LLM gates are unchanged.

[Raw result](meth226_donor_tangent_native_result.json), SHA256
`c8356317ce5d6499660e43b23c09251a2a4196128a60fa2035a15b84c9615667`,
retains every derivative, center/source/segment hash and numerical row.
Local ignored binary `results/native_expert_scaling/meth226_donor_tangent_fixture.bin`
SHA256=`024f20a271bba6e0306b5c2aa9ac18668e891798605d870e62465612605e7f2b`.
Executable SHA=`5c7c689eb189da427533e5c4075337e52e598488852b492f8187fddf6fb494d7`.
Runtime6.125s after imports,ending RSS1.103GB,GPU peak131.3MB,local3060,
no T4. Autograd emitted a cuBLAS context initialization warning and then
completed every check. No job is active.
