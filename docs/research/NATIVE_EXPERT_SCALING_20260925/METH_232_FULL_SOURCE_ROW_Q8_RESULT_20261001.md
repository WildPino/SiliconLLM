# METH-232: full-source row-Q8 fidelity passes, native cost fails

Protocol and implementation frozen at `71aa167`; session60444 completes,
exit0. All original24 FFNs retain all4864 nonlinear source units. Actual
row-scaled int8/FP32 C arithmetic, not a scaled-weight proxy, is exercised.

| Frozen prerequisite | Measurement | Limit | Result |
| --- | ---: | ---: | --- |
| Code/scale/bias segments |168 exact readbacks|All exact|Pass|
| Distinct source code matrices per organ |24 gate/24 up/24 down|24 each|Pass|
| C/oracle median relative output L2 |3.712874e-7|<=1e-4|Pass|
| C/oracle worst relative output L2 |4.993800e-7|<=5e-4|Pass|
| Pooled original-source function SSE/energy |.0001944004 (0.01944%)|<=.01|Pass|
| Six-thread24-layer component median |**11.023079ms/token**|**<=10ms**|**Fail**|

Three fixed timings:10.780695/11.023079/11.093767ms/token; all above10ms.
Worst individual source layer SSE/energy=.0005542784 (0.05543%), reported
without a separate layer gate. Source fidelity uses all256 existing states
per layer and original BF16-weight/FP32 FFNs, not BF16 intermediate
rounding or whole-model quality. Numerical parity uses384 comparisons.

**Decision: stop this fixed full-source row-Q8 operator before conditional
fitting.** No threshold/timing retry, learned bank, full-model evaluation,
new held-out prediction/generation/task or n-scaled route/LUT is opened.
The proposed542.050MB ideal whole selected ledger is not measured traffic
or a physical whole-model artifact; fidelity plus shape arithmetic does
not override cost failure or establish accepted token rate.

[Raw result](meth232_full_source_row_q8_native_result.json),SHA256
`5dad8dcb67d2b7eeae0c747d63b5cada6e2ce6bd6faed94bac32dd6052c94368`.
Local ignored fixture
`results/native_expert_scaling/meth232_full_source_row_q8_fixture.bin`,
314,892,312bytes,SHA256
`85e5fb45efba1d398eee95132974a53d9f7d0a0d8040c348e3395829963a8bbd`.
C output SHA256
`4425933904056b541c6730adcf0823efe859ccfcecf2d8facfd551c765bb1633`;
executable SHA256
`d444773f113eb7539d71e75bdbf61da223a9cd4a7acd356cfed8befd36c013d6`.
Raw record includes source/runner/executable hashes and every segment,
layer fidelity and comparison row. Runtime22.485s after imports,
ending RSS1.624GB,GPU peak134.101MB; local3060,no T4.

Next: [METH-233 phase-cost diagnosis](METH_233_ROW_Q8_PHASE_DIAGNOSTIC_PROTOCOL_20261001.md),
bitwise-identical arithmetic with phase timers. It may decide whether to
change matrix products or activation computation. It cannot reopen the
METH-232 cost gate. No assumption that fusion or integer activations will
retain fidelity, resolve the deficit or transfer to sparse GigaChat.
