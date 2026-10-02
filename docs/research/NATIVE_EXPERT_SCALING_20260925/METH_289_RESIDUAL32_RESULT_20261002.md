# METH-289: FP32 residual stream fails unchanged full smoke

Frozen `d68ad62` before observations; session86581 terminal exit0.
70.688s total, actual CPU compilation/execution69.250s. Same276 archive,
all725 fields/no fallback; original285 GPU reference reused by pinnedSHA.
Only two residual BF16 roundings removed. Original norm/qkv/o/attention/
FFN returns/final norm/head and weights/routes unchanged, checked through
literal source equivalence. No apparatus failure or repair in this run.

| Arm | Top1 | Hidden maximum relativeL2 | Logit maximum relativeL2 |
| --- | --- | --- | --- |
| Compact-core-only | 48/48 | .0522616096 FAIL | .0601920299 FAIL |
| Complete E1280 | 48/48 | .0562371872 FAIL | .0568670407 FAIL |

Allfinite; all12 fresh CPU caches byte exact to sequential under this
recipe; all12 erased-history controls detected. Four unchanged5% guards
FAIL. Higher residual precision does not repair original285 numerical
equivalence; do not adopt this fixed recipe or run native quality/K64/rate.
Original285 and288 failures remain preserved. Top1 equality alone does
not override those stops or establish whole donor-quality conservation.

All resource gates pass; no GPU/competing model job/accepted-rate inference.
[Raw result](meth289_residual32_qualification_result.json) SHA
`091f9894325fe4f24e4808585964a322c615112238fb12ffbc56463d76d3e1b1`
retains full arrays/cache flags/actual loader and source/executable/raw
hashes. Reproduction in [protocol](METH_289_RESIDUAL32_PROTOCOL_20261002.md).

## Next uncertainty

Before more precision variants, test GPU-reference stability on causal
equivalents: original full prompt's eight tail positions versus separately
truncated full-prefix executions ending at each same real position. Both
use no cache/same archive/weights/IDs/causal attention; only GEMM/attention
batch geometry changes.264's cached donor discrepancy is known, but does
not answer this no-cache, candidate-own-loader, numeric-envelope question.
Freeze protocol/gates/data/cost before observations, preserve original285
failure and numerical limits. Such a diagnostic cannot by itself promote
any failed native recipe or relax the goal's donor-quality/>=50 requirement.

Useful newly learned large-n capacity, actual CPU LUT/router/DRAM cost,
same-artifact quality/rate and other-family10B/100B are still open.
