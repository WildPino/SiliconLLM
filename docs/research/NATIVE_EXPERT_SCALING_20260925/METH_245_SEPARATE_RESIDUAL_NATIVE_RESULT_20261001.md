# METH-245: separate residual arithmetic passes, four-team kernel misses cost

First FP32 SVD construction stops before factors/native execution;see
[apparatus repair](METH_245_SVD_APPARATUS_REPAIR_20261001.md). Repaired
construction frozen at `2cf694c`;session83343 completes,exit0.
All217 base segments byte-unchanged,289 segment readbacks exact,24
finite/nonzero/distinct factor pairs and spectral energy controls pass.
Actual combined C/GPU384-vector median/max relative L2
**4.161875e-7/9.411517e-7**. Pooled source-function SSE/energy
**.000137500193654236**,worst layer.00041817395061654727,passes1%.

Fixed three24-layer timing passes are **10.404904/10.390391/10.217023ms**;
median**10.390391>10ms**. **Stop this fixed four-team rank32 operator
before learned factorization.** No rank/retiming/threshold retry. Numeric/
source-fidelity acceptance alone cannot qualify the full operator.

[Raw result](meth245_separate_residual_native_repair1_result.json),SHA256
`7022e2063afef2339342ff515ab4788732b7d795b8b2c690c511b4f5a5e5d61d`.
Ignored fixture `results/native_expert_scaling/meth245_separate_residual_fixture_repair1.bin`,
326,580,256bytes,SHA256
`065b7b7363ba1a334c43a4bcdebb11145c0906d5fcccf622d5c800eb5258095d`.
Output SHA256 `d6f2abed9dd65810192accc840469f976354e71db57f3ef8fda2ffaf7a15d773`.
Runtime45.641s,RSS1,792,319,488bytes,GPU peak288,945,664bytes,native peak
342,781,952bytes,local3060/six threads,no T4. Source-derived fixture only;
no learned conditional/fresh quality,n-scaled DRAM or accepted full rate.

Next: [METH-246](METH_246_SINGLE_TEAM_RESIDUAL_PROTOCOL_20261001.md) changes
execution organization to one OpenMP team per layer with shared barriers.
Rank,all physical factors and every reduction order remain fixed;require
all384 output vectors bitwise equal before any new kernel cost acceptance.
This is a changed kernel,not another timing of the rejected executable.
