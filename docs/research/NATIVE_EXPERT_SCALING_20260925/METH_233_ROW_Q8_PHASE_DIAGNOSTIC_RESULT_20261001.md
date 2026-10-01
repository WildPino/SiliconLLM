# METH-233: scalar nonlinear phase is material

Frozen at `359ac37`; session53582 completes,exit0. All original source,
fixture/vector/result bindings pass. The instrumented output check is
**bitwise identical** to METH-232 including its header, SHA256
`4425933904056b541c6730adcf0823efe859ccfcecf2d8facfd551c765bb1633`.

| Instrumented phase | Median ms per24-layer token |
| --- | ---: |
| Input decode plus gate/up |5.896257|
| Scalar SiLU/product |**3.004282**|
| Down plus bias/finite check |2.946925|

Input/down sum is74.642% of the sum of phase medians,below90%; thus the
frozen decision is `activation_or_instrumentation_share_requires_further_diagnosis`.
The scalar nonlinear phase is about25.36% of this measured sum,so a changed
nonlinear operator is worth qualifying. Phase timers add overhead; these
figures do not isolate expf alone, DRAM bandwidth or price a lookup variant.
Instrumented totals11.859249/11.850929/11.260880ms are retained solely for
diagnosis; METH-232 remains rejected and no performance gate is reopened.

[Raw result](meth233_row_q8_phase_diagnostic_result.json),SHA256
`f94bd41fc4e59071b31098bb2229ab21184fdd905ca05d52bad2c1ffdc2660a0`.
Runtime10.813s, native peak working set331.063MB, CPU only,six threads,no T4.
All source/executable/check hashes and nine phase timings are retained.

Next: [METH-234](METH_234_ROW_Q8_SILU_LUT_PROTOCOL_20261001.md), same physical
row-Q8 matrices plus a shared FP32 SiLU lookup with linear interpolation.
This changes the nonlinear arithmetic. Measure approximation/source error,
C/oracle fidelity and fixed cost before any conditional readout fitting.
No inference that useful expert count, complete model quality or accepted
full rate already follows from these component measurements.
