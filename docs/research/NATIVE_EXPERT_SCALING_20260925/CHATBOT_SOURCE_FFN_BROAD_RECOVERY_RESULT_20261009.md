# Broader fixed local recovery: COMPLETE/absolute function fidelity FAIL

9 October 2026. New FIT coverage across12 domains, fixed256 updates/site0 then23.
Original Student AST/STE/normalized total loss/optimizer/native arithmetic and
active4608 rows unchanged. No source calls, new replies, native, T4 or RESERVED.
Freeze8bb6af569888a0bebfa3d47fd231d0b74642b28f;320-input
bindingb995543146cf1d44bb529870e6ed8cb34db14ced40c923b5a8c7d245e9b4d380.

## Actual inputs, updates and retained artifacts

[Broad capture](CHATBOT_SOURCE_FFN_BROAD_CAPTURE_RESULT_20261009.md) supplies
48/8808 actual source x/y cases,24FIT/4422 and24DEV/4386. Fresh selected original
calibrated sectors at both sites; not the earlier two-prefix trained states.
Every initial pair matches its retained sector. All44 NEW initial inference/STE
responses per site have equal BF16 bits; four old initial responses are reused.
Their F64 aggregate errors match old reports within1e-12. All48 final outputs
per site saved. DEV never selects or fits a checkpoint.

All512 updates complete with six positive finite gradients, finite parameters/
moments and exact Adam counters.256/site; no fault, discarded work or repair.
Bound deterministic domain/case/row schedule and actual exposure match exactly.
Per site:13,965 training rows,all4,422 unique FIT positions;21-22 updates/domain,
10-11/case. Earlier two-prefix256 used10,496 rows/274 unique positions per site.
These comparisons hold update count, not row count, elapsed time or compute.

Four actual model/moment/RNG/history states retained off-repo; final site0
339,929,096B/SHAa260302562f5ac40fef1d5043989f7e867b4a1026c468cc13b9e90279f0b2590;
site23 samebytes/SHAb45e8d13340b12394a2c29144fd5922bfcaab52155606031c82be852d32d33ad.
Final packed trits/scales roundtrip every pair. Original source files unchanged.

## Case-balanced function errors

| Site / split | Whole L2 before -> after | Centered L2 before -> after |
|---|---:|---:|
|0 / FIT|.562957 -> .474764|.560572 -> .476520|
|0 / DEV|.559996 -> .519384|.557605 -> .520806|
|23 / FIT|.470861 -> .162547|.521842 -> .311164|
|23 / DEV|.488416 -> .195581|.555766 -> .372511|

Site0 DEV ratios whole .927478/centered .934006 (~7.25%/6.60% improvement);
site23 .400438/.670266 (~59.96%/32.97%). Every DEV case remains within1.05 of
its own before whole error. Site0 fails the10% aggregate improvement gate;
site23 passes it. Neither meets all-case whole error<=.10/cosine>=.99 or NEW
prospective centered error<=.10. SOURCE_FFN_BROAD_RECOVERY_FAIL.
Site0 maximum DEV whole/centered .536961/.536771, minimum cosine .846761;
site23 .595038/.969717/minimum cosine .803782. Aggregate gain masks large tails.

| DEV domain | Site0 whole after | Site0 centered after | Site23 whole after | Site23 centered after |
|---|---:|---:|---:|---:|
|apigen-80k|.450927|.453595|.125734|.200227|
|everyday-conversations|.528916|.529246|.374150|.614224|
|explore-instruct-rewriting|.523675|.524276|.172551|.440613|
|metamathqa-50k|.513987|.517085|.174605|.303299|
|numina-cot-100k|.522742|.529696|.143371|.265291|
|openhermes-100k|.535254|.535050|.316739|.608550|
|self-oss-instruct|.511043|.512051|.197545|.349807|
|smol-constraints|.526733|.527438|.166835|.352926|
|smol-magpie-ultra|.533638|.533782|.190772|.400078|
|smol-rewrite|.526716|.527360|.158330|.333203|
|smol-summarize|.525327|.525375|.163878|.340654|
|systemchats-30k|.533649|.534719|.162457|.261254|

Each domain has two DEV cases. Metrics are actual local functions at the same
source operands, not source-quality KL, a missing-knowledge fraction, unseen
chat behavior or a model-capacity ceiling. F64 mean/varying decompositions pass
1e-12; the earlier91-94% mean-gain result concerned the earlier four-case model
and is not copied as a claim about this new broader model.

## Effective representation changes and cost

| Site / matrix | Changed trits /9,437,184 | Master relative L2 | Scale relative L2 |
|---|---:|---:|---:|
|0 / gate|203,162|.027157|.041289|
|0 / up|175,648|.020034|.039320|
|0 / down|255,135|.021067|.058153|
|23 / gate|142,578|.022051|.049294|
|23 / up|277,854|.007089|.011256|
|23 / down|39,119|.023787|.012744|

Total changed trits2.24%/site0,1.62%/site23. This is not an error bound or proof
that further optimization helps; it identifies the size of the actual move.
FIT remains inaccurate, especially site0. Failed256 scope cannot distinguish
insufficient optimization dose from representation/surrogate limitations.

Family122.625s; worker result111.062s/event completion111.187s. Held OS peak
1,704,747,008B; GPU allocated1,141,823,488B/reserved1,342,177,280B. Sum measured
optimizer-update wall intervals9.166s/site0,8.974s/site23; these exclude guards,
initial/final evaluation, load/import/checkpoints/hash/verification. Do not use
them as end-to-end conversion price. Fixed600s/60reserve/OS8GiB/GPU10/11GiB/
output2GiB.204 namespace files1,528,461,446B. Resource/input PASS/exit0;session24779
terminal/closed. No other owned model/compiler job is running.

Result981679c917f5e7f39d868dfc2a5c6e7a34ba214fbb104084ef9d871ce1161c3f.
[Worker](../../../benchmarks/native_expert_scaling/chatbot_source_ffn_broad_recovery.py),
[protocol](CHATBOT_SOURCE_FFN_BROAD_RECOVERY_PROTOCOL_20261009.md), binding/result/
terminal JSON/log beside this record contain exact command/PIDs/AST/inputs/outputs.
Large tensors stay inresults/native_expert_scaling/chatbot_source_ffn_broad_recovery_20261009.

## Decision

Broader coverage helps in this fixed budget but does not establish faithful
functions. Next isolate optimization dose at the hardest site0 by restoring
actual model/Adam/RNG/history256 and extending a prospectively fixed schedule,
without new initial-response/source/capture replay. Keep loss/Student/precision/
active rows unchanged. This is NEW dose, not a blanket restart or general claim.
Loss weighting/common-residual representation are subsequent separate variables
if this bounded dose remains insufficient. See current recovery NEXT/INDEX.
All-site/whole-source/compact selection/state and useful same-artifact C quality+
>=50,useful n/CPU winners+mass/DRAM/family variants remain open.
