# METH458 matched whole costs: admitted, expert matrices are a minority

5 October2026. ONE session28023 terminated exit0; source/protocol freeze1eb56f9,
first execution resumptioncd51633. ALL8 apparatus gates and BOTH sources' two
profiler admission gates PASS. Goal ACTIVE/INCOMPLETE; no new model/quality claim.

First457 admission fault remains immutable/ec85910: KeyError in historical363
runtime schema,224.109s, ZERO native commands.458 corrects only that assumption,
adds first-failure binding and namespace; substantive science unchanged.
[First fault](METH_457_FIRST_ADMISSION_FAULT_20261005.md),
[458 protocol](METH_458_SWITCH_MATCHED_WHOLE_COST_PROTOCOL_20261005.md).

## Measured matched costs

ALL96 natural cases for EACH original source, BOTH profile0/1 warm1/measured3.
384 sequential commands,1536 full generation binaries (384warm/1152measured),
all complete encoder/decoder/logits/routes SHA and IDs exact original quality.
128 CPU3[0,2,4],256 CPU6[0,2,4,6,8,10]; every actual worker/process mask exact.
Different original cores/workers/context values: these are within-source timer
contrasts, NOT a causal E-only scaling test.

| Observable | Original128 | Original256 |
| --- | ---: | ---: |
| Healthy cases / all timed cases |96/96|81/96|
| Accepted ordinary / prose IDs |1142/662|895/490|
| profile0 ALL-case mean full seconds |11.0547647|11.2002461|
| profile1 ALL-case mean full seconds |11.0343680|11.2790246|
| profile1/profile0 mean ratio |.998154946|1.007033640|
| Min..max book mean ratios |.974377..1.027628|.952945..1.078517|
| profile1 ordinary mean IDs/s |103.494826|79.350833|
| profile1 prose mean IDs/s |59.994374|43.443473|

Aggregate admission[.90,1.10] AND EACH book[.85,1.15]. No overhead subtraction,
outlier removal, adaptive weights or retry. All15 rejected256 cases' time remains
charged. Current means are a repeated consumed original diagnostic; their higher
rates than historical391/376 do NOT establish improvement by a new method.
Historical median/confidence estimates and current case medians retained separately
in raw; never substituted into the mean-cost equations. No fresh uncertainty
interval was specified for current diagnostic rates.

| profile1 fraction of ALL-case mean whole time |128|256|
| --- | ---: | ---: |
| Dense/control matrices |44.078843%|40.013019%|
| Sparse expert WI+WO matrices |24.858489%|27.094388%|
| Router score matrices |1.445771%|2.278304%|
| Head matrix |6.333439%|5.973404%|
| Residual |23.283458%|24.640886%|

Kind0 combines attention/control and DENSE FFN matrices; no attribution to a
specific organ within it is measured. Residual includes ReLU/normalization/
attention softmax/route mass/probability combination/allocator/greedy and overhead
outside matrix timers. It is not an unexplained hardware bottleneck measurement.
Router score fractions exclude softmax/mass and routing control. Matrix wall
timers include A16/OpenMP; not per-worker CPU seconds or hardwareDRAM traffic.

| profile1 whole/component mean seconds |128|256|
| --- | ---: | ---: |
| Encoder whole |5.5626240|5.7029122|
| CrossKV whole |.8575506|.7023699|
| Decode/head/greedy/stop whole |4.6141934|4.8737424|
| Encoder+crossKV expert matrices |1.9295433|2.1745426|
| Decoder expert matrices |.8134338|.8814400|
| Encoder+crossKV dense matrices |2.5366979|2.0672116|
| Decoder dense matrices |2.3271239|2.4458666|
| Encoder+crossKV residual |1.8616954|2.0399437|
| Decoder residual |.7074871|.7393079|

Phase0 MATRIX counter includes encoder AND crossKV. Separate whole phases do not
allow a separate crossKV matrix assignment. All phase matrix sums fit their whole
times; full phase sum serialization tolerance1e-7s. Admission allows a diagnostic
share estimate, not proof that profiler0 share is exactly profiler1 share.

## Conditional algebra and decision

For SAME nonoverlapped stages/context, f=expert-matrix/whole:

    T_new/T=(1-f)+f*r
    R_new/R=1/((1-f)+f*r)
    r_required=(R/target-(1-f))/f

Hypothetical expert-matrix elimination speed ceilings:128 **1.330822315x**,
256 **1.371636513x**. These assume the entire measured expert matrix stage can
vanish without new work/routes/quality change. They are upper bounds, not feasible
implementations.456 .801649 FULL-FFN local ratio has NOT been inserted for r.

| Conditional target (same profile1 mean R) |128 max r|256 max r|
| --- | ---: | ---: |
| Ordinary50/s |5.303948|3.166562|
| Ordinary100/s |1.140589|.237880|
| Prose50/s |1.804101|.516023|
| Prose100/s |-.609335|-1.087389|

Negative r is impossible for expert matrices alone; r>=1 means that diagnostic
target already holds.256 prose50 would require >=48.3977% matched expert-matrix
cost reduction if every other term is unchanged. Even ideal elimination ceilings
for prose are79.84185/s128 and59.58865/s256. Keep ordinary/prose targets separate.

Both f<.5 => frozen rule selects **dominant other whole components before another
WI layout**. This does not mandate a new general donor runtime or abandon useful-n.
The compact shared component costs more than expert matrices in these cases;
expert compression alone has bounded whole leverage. Next inquiry must link real
transfer/capacity to this whole budget; no456 rerun/tile sweep/all-bank promotion.
[Whole reassessment and next inquiry](WHOLE_TRANSFER_REASSESSMENT_AFTER_458_20261005.md).

## Retention, resources and remaining goal

Raw10,274,971B SHA256
`3762fd4e6c86a6ea7e64f54d4350cc2bd91761f2f856a06bb0a0a442f12d285f`.
[Raw](meth458_switch_matched_whole_cost_result.json),
[fresh metadata retention](RETENTION_458_20261005.json).
Metadata-only audit session34349 exit0: ALL7 retention gates PASS,28.344s,
30,335,727,235B freshly hashed, no scientific rerun/import. Retention SHA256
`65ec6408bc1de79d0a3bad869b57de7298285901c5c91ca502330250ea9ea8b0`.
ALL2304 new files4,951,786,960B retained,1165 reused files inventoried; no files
trimmed.523.234s=33.047admission+490.187native/aggregation. Parent peak84,107,264B,
native peak1,292,668,928B, conservative separate-peak sum1,376,776,192B.
ALL384 terminal Windows memory queries succeeded and bounded polled peaks.
This is working-set memory, not DRAM traffic; payload mappings are not residency.
Bytes hashed30,325,452,264 before final raw hash. No compile/kernel/engine edit/
new rounding/model download/fit/GPU or competing model process; exact daemon
allowance maintained. Unrelated working-tree modifications preserved.

Consumed original contexts establish component constraints only. Full new
transfer artifact + fresh donor-relative prediction/generation/task quality AND
SAME accepted50, useful growing-n, router winner+mass certification, actualDRAM,
more families/~100B remain required. No generality/quality/capacity conclusion
is supplied by the profiler or by these conditional ceilings.
