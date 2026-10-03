# METH-332: stable all-projection accumulation, original probability FAIL

Freeze10de72f. [Raw record](meth332_switch_full_source_result.json) SHA256
9e4b9c53c619b0124feb66aa02d3f1ee938ad2470bf05f540b9f0ccb779b6409.
[Protocol](METH_332_SWITCH_F64_MV_PROTOCOL_20261003.md), new C332 matrix-vector
F64 products/sum with F32 output; prior330 normalization, all other operations
unchanged. Exact source difference checked before observations. Original
328/329/330/331 sources/binaries/failed observations retained.

Every6392 actual mapped tensor SHA EXACT327; same original14.664B/12x256
source banks/config/tokenizer/forced controls/official reference. Both original
328 Tiny weight hashes exact and all state/route/probability/greedy guards PASS.

| Actual source case | Encoder pooledL2 | Decoder pooledL2 | Full logit pooledL2 | Probability maxabs |
| --- | ---: | ---: | ---: | ---: |
|0 France|1.16517e-6|2.35297e-7|2.17499e-7|**1.49011612e-6 FAIL**|
|1 experiment|3.26559e-6|4.84312e-7|1.88803e-7|7.74860382e-7 PASS|

All source/C initial/block/final states<=1e-4, choices/capacity/greedy exact,
both zero-head faults detected. Original probability1e-6 STILL failscase0.
Full higher-accuracy projection reduces pooled decoder/logit differences but
does not close qualifier; unchanged332 is NOT promoted under329/330 rules.
Do not call this a source model quality rejection or complete artifact pass.

133.000s main excluding imports, checked combinedRSS6,566,907,904B, end
controller4,599,721,984B. Exec75827 exit0. All outputs/native all-tensor audit/
Tiny fixtures/compiler/runtime logs retained. No accepted-rate or quality data.

Stop another blind summation/precision sweep on these consumed contexts.
331 demonstrates actual source-backend rounding and upstream numerical effects.
Next explicit numerical-policy proposal separates implementation correctness
for a prescribed precision recipe from donor-relative quality; original tight
CPU1-backend probability failures remain. See
[next policy proposal](SWITCH_NATIVE_ARITHMETIC_POLICY_NEXT_20261003.md).
That proposal is not an implementation or waiver of final quality/rate goals.
