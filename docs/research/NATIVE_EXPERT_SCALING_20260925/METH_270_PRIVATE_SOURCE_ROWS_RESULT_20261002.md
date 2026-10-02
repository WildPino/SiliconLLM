# METH-270: fixed128 private source rows stop at fidelity

Apparatus frozen in `f15b00f` before observation, following the
[prospective protocol](METH_270_PRIVATE_SOURCE_ROWS_PROTOCOL_20261002.md).
Session23631 exited0. All24 layers/all6144 consumed source-state controls
complete; no apparatus repair. No CPU timing or complete model was built.

| Metric against actual source BF16 MLP | Original32 | Fixed128 |
| --- | --- | --- |
| Mean per-state relative squared error | .0001972897928984215 | .0001683502832747763 |
| Energy-normalized squared error | .00003673979734634541 | .00003319058617155685 |
| Maximum layer energy-normalized squared error | .00030542892636731267 | .00023848708951845765 |

Mean error improves14.6685285632%, but its ratio .8533147143676851 exceeds
the frozen .85 gate. Maximum-layer gate passes. Preserve the near miss as
a failure; do not loosen the threshold, increase the row count or promote
the fixed128 coefficient-ranked recipe. All old32 IDs/source BF16 rows
remain exact, all289 nonprivate segments are byte identical, all361 new
segments read back exactly, and all6144 old baselines reproduce269 exactly.

The new fixture is337,596,452bytes,8,262,144bytes above252. SHA256
`30f999d88dc38019d443b8e1afd5ebd899e89a713d6c67b440b1d135e4cc7fed`.
Selection/scoring/export takes14.125s after imports; end RSS2,303,774,720bytes,
peak allocated GPU197,893,632bytes. No overlapping model job remains.

Raw [result](meth270_private_source_rows_result.json), SHA256
`b14fdd1b7c8867ff01ee36600c8d8903ab74f1abeb60cd4051dfe9b0e93308bd`.
Code: `benchmarks/native_expert_scaling/meth270_private_source_rows.py` and
`meth270_private_source_rows_cpu.c`; exact command in protocol. The code
record binds original donor/archive/helpers,252/253/269 and125vectors.

This is consumed component fidelity, not generated semantic quality,
independent evaluation, useful learned-n growth or CPU cost.259 remains
closed by267. A different selection mechanism must be prospectively fixed;
coefficient discrepancy alone does not account for nonlinear activation,
downstream projection or the mean-state metric. That is a hypothesis to
test, not a demonstrated cause of this failure or of semantic errors.
