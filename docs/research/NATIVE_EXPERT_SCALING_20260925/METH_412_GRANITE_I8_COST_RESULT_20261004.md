# M412 result: correct complete I8 fixture, cost/repeatability profile FAIL

Frozen004411c, FIRST compile/native outcome, wrapper terminal exit0.
[Protocol](METH_412_GRANITE_I8_COST_PROTOCOL_20261004.md),
[controller](../../../benchmarks/native_expert_scaling/meth412_granite_i8_cost.py),
[raw](meth412_granite_i8_cost_result.json).
Raw SHA cb02f6413dfde4748872de1fefaf4df7d0462e03f41d2283451cbf3fcc6fe7bb.
Binary SHA1cfda07925069b82535d69448f015e95903a9f537b1b2805684f7e150f4d86ab.
ALL8 apparatus/math/whole-output gates PASS. Both cost/repeatability profiles FAIL.

## Decision

Reject this exact complete-row-I8 cost profile before actual Granite values,
reference import, export or quality evaluation. No14ms/range/10% threshold
relaxation or repeated unchanged run. This is NOT a general row-I8/source
impossibility: unchanged388 row dot redundantly loads the same A16 input once
per output row. ONE separately numbered, prospectively frozen four-row SIMD
variant could reduce those loads and expose integer accumulation parallelism,
with exact412 full-output identity and unchanged balanced profiles/gates.
That is the next concrete variable, not another donor or precision sweep.
If it fails, stop this Granite I8 kernel route before source-value acquisition.

The small1.335B/source32 case remains an another-family transfer prerequisite,
not useful>256/10x/real~100B/RAM-scale capacity. Original qualified Switch
artifacts/quality/rates stay intact. No final-goal promotion follows this screen.

## Full source-sized work and numerical evidence

Actual411 source0 geometry L24/D1024/FF512/32banks/top8; full Q/K/V/O,
selected8 packed gate/up and down, all32-score F32 routers, norms/A16,
SiLU/mixture/residuals, BF16 lookup*12, residual*.22, full49152x1024 I8 head/6.
ALL427819008 I8 coefficients and2064384B row scales executed every input;
complete conservative descriptor433231872B reconciled, stored descriptor
1444581376B, actual dynamic payload+outputs1443299328B. Static structs and
runtime/workspace overhead charged in RSS. Descriptors are not physical traffic.

Exact original388 A16/I64 kernel source bridge PASS.3848 sampled active integer
rows exact independent scalarI64; scaled relativeL2 2.4580692885963728e-8,
additional64 head rows3.262856583145958e-8.3266 stored bank/code/scale edge
checks,2805 mixed all-weight/representative-A16/SIMD-tail cases, globalI32
overflow, zero/ties/NaN/Inf/badbank/code/scale/headbit negatives all PASS.
All fixture activation codes independently rounded; ALL24 scalar router logits,
selected-softmax/sorted bank bindings, ties, independent norm/SiLU/residual and
BF16 lookup controls PASS. Scale negative checks the declared row formula;
code/head negatives exercise native dot. No stronger original-model parity claim.

ALL180 complete executed SHA match; ALL60 full3515716B archives directly equal
across3/6 profiles and processes, independently parsed through final head,
no missing/trailing bytes. Same complete archive schema S412OUT1 includes all
norm/residual/context/nonlinear/matrix outputs, routes/scores/gates and A16
inputs/scales, not only fingerprints. All outputs finite.
Exact worker masks/event/end identity pass; injected binding fault exits2 as
required. All positive subprocess terminal exits0 consumed before dependencies.
Selected synthetic union27..32 per layer does not prove useful real32 experts.

## Unchanged prospective cost gates

Balanced processes3,6,6,3,3,6, three repetitions x10 inputs each;0/1 warmups.
All9 rep medians and24 fixed-input medians per profile charged.

| Physical workers | ALL9 repetition medians, ms | max/min | Worst fixed-input median, ms | Joint gate |
| --- | --- | ---: | ---: | --- |
|3|16.31720,15.04950,14.34320,15.47755,13.60430,13.74575,13.85510,13.53765,13.49850|1.208815794392296|16.29560|FAIL14ms and1.10|
|6|31.51365,19.81980,16.02615,17.28045,16.11945,15.59270,15.05705,13.98980,14.05800|2.252616191680631|51.91690|FAIL14ms and1.10|

Some medians <=14ms do not qualify a profile. No post-outcome deletion of first
repetitions/processes or warmup expansion. Source timing variation alone does
not identify its cause. Native peakRSS about1.455GB. MAIN18.500s/max checked
combinedRSS1496752128B/output212553277B, within600s/120s/6GiB/2GiB envelope.
No network/source values/reference model/training/GPU/T4 or new quality data.

## Scope and reproduction

All tensor values synthetic. Independent layer/context fixtures: attention
causality/scale.015625, RoPE/actualGQA, KV/cache, layer composition/prefill,
tokenization/generation/task/source-reference backend parity are OMITTED.
Norm reduction is declared F64, tie rule lowerID. No source quality, accepted
tokens/s, real DRAM or useful-n scaling established. No16B Ling/Giga port resumed.

Reproduce protocol command from frozen004411c with fresh result/output locations
and retained411/374/388 inputs. Controller is HEAD/physical-byte-bound; later
engine prefixes require the frozen checkout. Compiler21.1.8 and OpenMP SHA in
raw, compile exit0/no warnings. Full output files under
results/native_expert_scaling/meth412_granite_i8_cost. Default0ff9705 and prior
3f87d37 engine paths source-exact; qualified Switch binaries untouched.

Next four-row proposal must freeze all changes before compilation. Exact same
fixtures/archive bytes/math/worker profiles, independent new tile extrema and
scalar controls; no rate/quality/data acquisition unless joint cost gate passes.
