# METH-246: one-team rank32 operator passes with bitwise arithmetic conservation

Frozen at `139dc1d`;session44513 completes,exit0. Same326,580,256byte
source-derived fixture,all289 segment/header/EOF reads exact. All384
output vectors/**344,064 FP32 elements are bitwise equal** to METH-245.
The256-state aggregate timing checksum also matches exactly. Original
GPU numeric median/max4.161875e-7/9.411517e-7 and source SSE/energy
.000137500193654236 carry for these unchanged physical functions.

Single-team fixed timing passes **9.636953/9.807457/8.963098ms**,median
**9.636953<=10**. All gates pass. This qualifies the changed execution
kernel;METH-245's four-team kernel remains cost-rejected. No retiming,
rank change,factor recomputation or altered threshold. Timing is a source
FFN component,not trained-bank cost or full accepted-token rate.

[Raw result](meth246_single_team_residual_native_result.json),SHA256
`bb84bdb1544025866e7411dd830ffea3bc5daf4b73447bf194c45c55ba9da202`.
Output SHA256 `8a3fb9148007d902ffc7d6d02e2d34ba8c36a7dfddb5862bda9e243a10c442ac`;
old/new payload SHA256
`43f62afad8ef6deeba9e06e05587ea4eba0603f74abc8ac293fb3cf4e5ad9373`.
Runtime8.203s,harness RSS1,254,846,464bytes,native peak342,781,952bytes,
six host threads,no GPU computation/T4. All source/fixture/code hashes retained.

Next: [METH-247](METH_247_WEIGHTED_RESIDUAL_PAIR_PROTOCOL_20261001.md)
factorizes the actual frozen raw E16/E160 solutions into this representation,
retaining the stored fitted parent. It must conserve actual function/count
gain;source operator qualification cannot establish learned factor fidelity.
