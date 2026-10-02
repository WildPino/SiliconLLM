# METH-285: whole native prefix/cache smoke fails

## Binding and decision

Code/protocol frozen at `dfb4292` before observations. Same original276
archive SHA `4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`,
725 directly bound fields, no source/checkpoint fallback. Whole forward is
verbatim284 through285's model header. Fixed six consumed278 prompts,
entire147/152/166/161/209/184-token prefixes, last8 positions each arm.
GPU own loader/no cache versus actual phase60 sequential CPU execution.

**Measured FAIL:** fixed compact-core-only ablation exceeds the prospective
5% full-logit relativeL2 maximum. Stop before native quality/K64/rate.
Complete-bank passing its smoke guard does not override the failed ablation.
No threshold/data/row/route/weight changes or apparatus repair occurred.

| Arm | Top1 matches | Hidden maximum relativeL2 | Logit maximum relativeL2 |
| --- | --- | --- | --- |
| Stored compact core only (not donor) | 48/48 | .0440816134 PASS | .0542237870 FAIL |
| Stored complete E1280 | 48/48 | .0475705527 PASS | .0479056798 PASS |

All rows finite. All12 independent CPU full-prefix cache rebuilds are byte
exact against sequential execution. All12 erased-history controls change
outputs (logit relativeL2 approximately1.054..1.178). This qualifies scoped
CPU cache data flow, not CUDA cached/full equivalence or universal contexts.

## Resources and reproduction

Session72992 terminal exit0; scientific decision
`whole_native_prefix_smoke_fail_stop_before_quality_and_rate`.
121.266s total: GPU54.641s, subsequent CPU compilation/prefix/cache/negative
66.610s. End GPU-reference process RSS3,330,539,520bytes, peak allocated
CUDA1,442,444,800bytes. Local RTX3060/six threads; no overlap/rate inference.
All resource stops pass. Native output66,023,692bytes; GPU NPZ58,688,674bytes.
Command and every source/executable/raw-file hash are in
[raw result](meth285_whole_prefix_qualification_result.json), SHA
`c9b87bdaa42fed8e9313d6634671ded9f3fa31993eb9f46e23a43c95885f6eda`.
Use [frozen protocol](METH_285_WHOLE_PREFIX_PROTOCOL_20261002.md) for command.

## Next decision

Numerical localization on consumed data may distinguish norm/projection/
RoPE/attention arithmetic from accumulated full-model drift. Actual GPU
attention backend must be observed before attributing a cause. A new
execution recipe needs a new prospective record and unchanged285 smoke
limits; preserve this failure and original source. GPU276 quality passes
remain scoped to that GPU equation. Useful huge n, actual DRAM/LUT/routing,
same-artifact >=50 and cross-family10B/100B remain unfulfilled.
