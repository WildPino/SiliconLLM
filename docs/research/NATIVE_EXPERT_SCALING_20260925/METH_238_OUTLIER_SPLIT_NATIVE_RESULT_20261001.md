# METH-238: fixed32 BF16 output escapes pass source/native prerequisite

Protocol/sources frozen at `694278e`; session87532 completes,exit0.
All source gate/up/table/bias controls are byte-identical to METH-234.
The output remainder zeros32 selected columns/row,stores their indexed
BF16 values,and recomputes row-Q8 scales. All source selection/value/codec
edge checks,217 segment readbacks and24 distinct matrices/organ pass.

| Frozen check | Measurement | Limit | Result |
| --- | ---: | ---: | --- |
| C/oracle median relative output L2 |4.161112e-7|<=1e-4|Pass|
| C/oracle worst relative output L2 |9.405341e-7|<=5e-4|Pass|
| Pooled source-function SSE/energy |.0001389773 (0.01390%)|<=.01|Pass|
| Six-thread24-layer component median |**8.727370ms/token**|<=10ms|**Pass**|

Three fixed passes8.754209/8.717465/8.727370ms;all pass. Indexed reads,
BF16 decode and separate exception reduction are included in actual C
operator. No ideal locality/traffic assumption. Worst source layer error
.0004203694 (0.04204%),reported without a separate layer gate. Source
screen uses256 existing states/layer;numeric screen384 outputs against
the declared separate row-scale/exception GPU oracle. Not actual BF16
whole-model intermediate arithmetic or held-out prediction quality.

**Decision: mixed operator prerequisite passes; qualify source priors
with this actual codec next.** Original-source BF16 escapes are exact;
FP32 projected/learned coefficients may round and must still pass the
unchanged stored1% derivative gate. No fitted/routed bank,n-scaled RAM/
LUT/DRAM,complete accepted rate or sparse-Giga variant follows from this
source fixture. METH-234/238 timings are separate fixed runs,not a paired
statistical speed comparison. Whole544.805MB ideal ledger remains arithmetic.

[Raw result](meth238_outlier_split_native_result.json),SHA256
`61b496f66455e764039877ddaf17cedaa3aab49c287f09178e085afd56dbe684`.
Local ignored fixture
`results/native_expert_scaling/meth238_outlier_split_fixture.bin`,
317,646,876bytes,SHA256
`42ce44c30e57e089ac9cdc8b986599dffa8bf99e47aa12ddd71cb49be1e85184`.
C output SHA256
`1f3847eb9d6a513afd16266a3b02d259ecf1e71426a13592a4e2ca6cd75a2661`;
executable SHA256
`78895dad1f227c8dc3e3e286ea5869666faced1c90200e57cc22b3844bf28eb4`.
All source/script/segment/row/layer hashes retained. Runtime24.391s after
imports,RSS1.912GB,GPU peak248.900MB,local3060,no T4.

Next: [METH-239](METH_239_MIXED_OUTPUT_PRIOR_PROTOCOL_20261001.md),all16
output-only source priors using the fixed32 mixed codec,no extra METH-237
cycles or changed slope threshold,then separately frozen E16/E160 fitting.
