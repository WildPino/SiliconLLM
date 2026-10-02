# METH-288: norm accumulation alone fails unchanged whole smoke

Original freeze3b6dc9e; original launch24758 exits1 before CPU observations
on Windows engine-path spelling. [Narrow path repair](METH_288_BINDING_REPAIR_20261002.md)
frozen11a0627; original failure preserved, no scientific choices changed.
Repair run3400 terminal exit0, measured FAIL. Same276 archive/all725/no
fallback; only norm-square accumulation usesF64, mean castF32 before
original inverse/BF16 boundaries. Original285 GPU reference reused bySHA;
no new GPU execution or source selection.

| Arm | Top1 | Hidden maximum relativeL2 | Logit maximum relativeL2 |
| --- | --- | --- | --- |
| Stored compact core only | 48/48 | .0466255955 PASS | .0556873083 FAIL |
| Stored complete E1280 | 48/48 | .0443350561 PASS | .0425241329 PASS |

All12 CPU cache rebuilds byte exact; all12 erased-history controls detected;
allfinite. Unchanged5% ablation logit guard fails. Norm64 alone does NOT
repair285: ablation maximum worsens5.4224% to5.5687%. Complete-bank scoped
improvement does not override the failure. Stop this fixed recipe before
quality/K64/rate; do not adopt it or regrade.

71.250s total, CPU compilation/execution69.922s after bindings; no competing
GPU/model job, no accepted-rate inference. All resource gates pass.
[Raw repair result](meth288_norm_reduction_qualification_repair1_result.json)
SHA `5c5b5e22eab0ce4e3b2efbfd4acee6c8f245176c06e386b2c80101d949903130`.
Original binding failure SHA
`bd391d1406bdfa1414ede4c6092e7544016ab043166c7da8e5edec778caab580`.
Source/executable/reference/native hashes and commands retained in raw/
[protocol](METH_288_NORM_REDUCTION_PROTOCOL_20261002.md)/repair record.

Next hypothesis: keep residual state FP32 while retaining original norm
reduction and all BF16 projection/FFN return boundaries. This changes
execution precision deliberately rather than claiming byte equivalence.
287 measured small local differences amplified through the rounded residual
stream. Whether fewer residual roundings preserve reference/quality is
unknown; a new frozen recipe must face unchanged285 gates first. No new
useful capacity/DRAM/LUT/family/>=50 claim; all original stops preserved.
