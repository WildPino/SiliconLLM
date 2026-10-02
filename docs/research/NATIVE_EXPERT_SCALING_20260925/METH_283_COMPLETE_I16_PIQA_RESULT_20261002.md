# METH-283: complete I16/private128 passes full PIQA regression

Frozen protocol/source `d893f01`,script SHA
`e54edbb39bff39271b5c7f3c543a176a35fe9fa6b44b2fe3f50ea07230057241`.
Session97267 exits0,1101.906s after imports (18.365minutes),without repair,
candidate changes,data filtering or gate changes. Original donor,BF16
E1280 and stored276 candidate each score all1838 cached validation items
anew,using21's full BF16 tied head/no-cache suffix-NLL arithmetic.

| Arm | Correct/1838 | Accuracy |
| --- | ---: | ---: |
| BF16 donor | 1291 | 70.239391% |
| BF16 learned E1280 | 1291 | 70.239391% |
| Stored I16/private128 E1280 | 1289 | 70.130577% |

Candidate delta is two questions,-.108814 percentage points versus BOTH
controls. Against donor:16 losses/14 gains,paired lower fifth percentile
-.598477points. Against E1280:12 losses/10 gains,lower-.544070points.
All four unchanged accuracy>=-2point/paired-lower>=-5point gates pass,
bootstrap seed212121/20,000 draws. Mean suffix NLL is primary;total NLL
remains descriptive,exact ties select option0. No claimed task improvement.

All newly recomputed control rows (choices AND every option NLL/token count)
are exactly equal to266's corresponding control rows. Data/token/label
binding and task format/metric are also exact. This consistency is an
additional observation,not a changed gate or reuse of prior scores.
Repeated full regression dataset,not fresh independent task/general ability.
The new candidate has one fewer correct answer than old259 on this reused
task;that observation does not reopen259's semantic stop or negate the
separately scoped current prospective pass.

Raw [result](meth283_complete_i16_piqa_result.json),3,434,644bytes,SHA
`2ed1e54df9722c32287445eeb2b8d40590742b1893656c374cff6490912a7725`.
Contains all5514 scored rows/all option NLLs,1838 source/token/label bindings,
paired summaries,gates,load record and helper hashes. Actual unchanged
276 artifact SHA `4f9b9c7a76475d9b6e241947ee590884ea268d4bf7f02097df57ad4b2ac23fe9`;
same archive as277/280/281/282. Candidate loader consumes all725 fields,
no source/checkpoint-weight fallback,no KV cache. All76 archived helpers
remain byte exact;Git filtered checkout preserves frozen source/verdict/score
hashes. EndRSS3,429,310,464bytes,peakCUDA3,139,373,568bytes. LocalRTX3060/
six CPU threads,within45min/20GiB/10.5GiB/96MiB budgets,no download/T4 or
CPU performance overlap. No active job remains.

## Decision and concrete resumption

Current276 passes all frozen consumed/new-source prediction,generation
health/K64,anonymous semantics and PIQA gates in its Python reference.
It remains diagnostic-only/native-promotion=false. This licenses prospective
actual C binding/arithmetic/route-alias/cache qualification,then complete
native quality and accepted rate on the SAME artifact. The implementation
does not yet exist;[source-based plan](NATIVE_COMPLETE_I16_IMPLEMENTATION_PLAN_20261002.md)
records concrete reusable operators and required BF16 boundaries.

Next implement the phase60 complete-archive entry/loader without weight
fallback and freeze its qualification apparatus before native observations.
Qualify source FFNs and isolated conditional routes/maps on6144 original
states,then actual whole prefixes/cache and quality. Stop before speed on
a failed guard. Whole batch1 lowerCI95>=50 accepted tok/s,real DRAM/LUT/
routing cost,useful RAM-scale n and multiple-family10B/100B transfer remain
open. Retain259 semantic/264 cached/274-275 fixed cost stops. No historical
timing sum or old native artifact can establish the current whole rate.
