# Complete source-width FFN arithmetic control

9 October2026. COMPLETE/execution-resource PASS,quality-preservation criteria
FAIL. Original source geometry/organs/FFN row coverage retained;only the
ternary/AQ63/F32-internal/final-BF16 FFN package changed. This identifies an
actual conversion problem before row selection or core compression. It does
not independently attribute loss to trits,AQ63 or changed arithmetic boundaries.

## Bound input and actual execution

Falcon-H1-1.5B-Instruct revision80ebc50d7799a440b96c93bb6686a3924a09b0cb,
D2048/L24/intermediate4608/V65537/bothEOS11,228. Decode the actual preflight
sector9aa6f319/340,819,968B,re-encode EVERY pair withfull equality,0 new weight
calibrations.339 non-FFN parameter objects/875,386,240 elements retain identity.
The converted679,477,248 FFN coefficients allremain active;this is offline
source-width arithmetic,not an affordable final engine runtime.

ALL48 adopted original cached forced trajectories/8808 labels/12 domains:
reuse completed preflight2/274 rows,46 new/8534 forwards. No original source
reply regenerated. ALL16 known screen prompts and4 actual generated-history
followups completed,20 new counterfactual generations/951 emitted IDs. No
training/native/T4 or RESERVED query. Original source14/16 screen reused.

## Prefix results

| Split | Cases/labels | Case-mean KL_F64 | Label-mean KL_F64 | Differing greedy IDs |
|---|---:|---:|---:|---:|
| FIT |24/4422|1.906616820|1.851801933|1747/4422=39.5070%|
| DEV |24/4386|2.099672799|1.975474201|1884/4386=42.9549%|

| DEV domain | Case KL | Differing/labels |
|---|---:|---:|
| apigen-80k |1.633421|59/211|
| everyday-conversations |2.622750|118/225|
| explore-instruct-rewriting |2.974978|165/268|
| metamathqa-50k |1.331774|141/447|
| numina-cot-100k |.707975|99/512|
| openhermes-100k |2.779346|292/512|
| self-oss-instruct |1.017835|139/512|
| smol-constraints |2.893766|84/143|
| smol-magpie-ultra |2.400855|276/512|
| smol-rewrite |2.362865|215/400|
| smol-summarize |2.138453|65/132|
| systemchats-30k |2.332054|231/512|

Eight of12 domain KLs exceed2;the same eight exceed.35 greedy disagreement.
Original EOS strata24 cases/2664 labels/1240 disagreements;partial strata24/
6144/2391. Partial replies remain prefixes,not complete-answer quality.
These consumed development distributions do not certify unseen usefulness.
Compact broad24 DEVKL6.990792/87.4829% is a different retained intervention on
the same source packets. KL differences are not additive attribution of
compression/routing loss;do not infer percentages of knowledge preserved.

## Actual generation

Exact normalized requested-answer score1/16,original14/16. Arithmetic0/4,
extraction1/4,instruction0/4,history0/4. Only extraction `blue` passes.
Four actual generated-prior-history followups0/4 underthe same exact rubric.
Several failures retain factual fragments butbreak the requested format:
addition mentions12 in extra text;inventory says `Three.` where3 is requested.
Others have substantive errors:subtraction answers7 for15-6,multiplication
answers4 for3*4,division produces inconsistent5/10 for20/5. Repetitive and
damaged text also occurs. Therefore the failure is not solely normalization,
nor does0/4 exact history scoring prove every stored fact disappeared.

ALL951 F32 score rows finite/argmax IDs consistent.20 outputs nonblank/no
non-EOS special leaks/valid stops:6 EOS,14 caps of64. Four followups append
the actual generated answer toits supplied prior-history prompt;limited fact
probes,not broad fresh chat or a native cache test. Original source followups
were not regenerated. Allscore bytes/IDs/text/messages retained.

Fixed [protocol](CHATBOT_SOURCE_FFN_CONTROL_PROTOCOL_20261009.md):DEVKL<=1,
disagreement<=.20,everydomainKL<=2/disagreement<=.35,screen>=13/everycategory
>=2,allfour followups correct. Seven of10 gates FAIL;blank/leak/stop PASS.
Decision SOURCE_FFN_CONTROL_FAIL. No threshold or selection changed.

## Resources and reproducibility

Worker1662.578s/family1674.125s,below2100s cap/60s reserve. Held OS peak
3,741,171,712B,GPUallocated4,901,416,960B/reserved11,158,945,792B (under10/11GiB).
136 outputs1,368,299,640B,under2GiB. All bound input/resource gates PASS,
exit0/session25136closed;launcher20256/worker17344. No firstfault or overlap.
GPU allocator statistics are not physical VRAM residency. Selected runtime
files are bound,not a full DLL-tree certificate. Fullsource active cost remains.

Freeze d4f7f0f80ec468be2e4b0055c2a0cf88e3be7d50;
98-input binding3d86ba6235577bf28369125bcd1f3fa3eef7557c286376fed81e06a746a46430.
ResultSHA453d4ea94c9c27cb57c303bab443e34b600936c244ce9269b835f0b66db94e51.
[Binding](chatbot_source_ffn_control_binding_20261009.json),
[result](chatbot_source_ffn_control_result_20261009.json),
[terminal command/alloutput hashes](chatbot_source_ffn_control_result_20261009.terminal.json),
[worker log](chatbot_source_ffn_control_result_20261009.worker.log).
Tool `benchmarks/native_expert_scaling/chatbot_source_ffn_control.py`.

## Decision

Do not add core/selection compression tothis unrecovered arithmetic stage or
assign its failure to n/CPU routing:neither was changed. First execute the
[short F32 positive control](CHATBOT_SOURCE_FFN_CAST_CONTROL_PROTOCOL_20261009.md)
to isolate the changed internal arithmetic from the trit/AQ package. Then
price activation-aware local FFN calibration/recovery,carrying original kernel
andactive-cost constraints. Original LUT/ternary/compact SSM/SWA deployment,
fresh source-relative quality+50 onthe same artifact,useful n/DRAM/families
remain the goal;this result supplies no native admission or general impossibility.
