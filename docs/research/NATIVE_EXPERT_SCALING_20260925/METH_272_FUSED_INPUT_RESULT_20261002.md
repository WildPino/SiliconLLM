# METH-272: fused shared gate/up dots stop at cost

Frozen `9824ce9`,session30895 exits0. Same271 fixture/selector/weights and
arithmetic,changed shared input-dot loop only. No repair/retry/model job.
All344,064 native check values are bitwise equal both original271 and the
contemporaneous unchanged control; all256-state checksums equal79.303890132.
All361 physical segments and prior source/numerical bindings pass.

| Arm | Three ms/24-layer token passes | Median |
| --- | --- | --- |
| Unchanged271 |10.864646 /11.228494 /10.636138|10.864646|
| Fused272 |10.711030 /11.324342 /10.991488|10.991488|

Changed/control ratio1.0116747476171797: no measured improvement. Both
prospective gates fail: absolute<=10ms and at least5% faster than control.
Do not retime or promote this fixed fused kernel. The contemporaneous
control also exceeds10ms;271's earlier cost failure remains unchanged.
No claim of statistically general slowdown or a specific hardware cause.

CPU-only19.235s after imports,harness RSS43,433,984bytes; native peak
353,652,736bytes control/354,246,656bytes changed,six threads,Ryzen5 3600X,
clang21.1.8/original253 flags. No active job remains. Raw
[result](meth272_fused_source_input_result.json), SHA256
`7109476722266d7dcc37abbfbde30bb11ff2c8772cbf7abcc73a4570149b343f`.
Exact commands/gates in [protocol](METH_272_FUSED_INPUT_PROTOCOL_20261002.md).

Reused [METH-233 phase diagnosis](METH_233_ROW_Q8_PHASE_DIAGNOSTIC_RESULT_20261001.md)
measured5.896ms input,3.004ms scalar SiLU,2.947ms down on its earlier
operator.234's LUT removed that nonlinear bottleneck and achieved8.522ms.
Those old phases lack271's private128/escapes/rank32 execution and cannot
locate its present cost. Before another optimization, instrument the actual
unchanged271 kernel's shared/private/readout/residual/team boundaries,
verify bitwise outputs and preserve diagnostic overhead. No new full
archive. This source CPU work enables transfer but does not establish
useful new n,dynamic route/LUT/real DRAM quality,whole accepted>=50 or
family/10B/100B applicability;259 still closed by267.
