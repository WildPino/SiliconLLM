# Original full-vocabulary native width envelope: COMPLETE, raw cost PASS

9 October2026. Goal ACTIVE/INCOMPLETE. Actual original kernels/LUT/AQ63/dReLU,
sameD256/N96/L6/n1152/k8/fullV65537/Adam25 bank. DN512/DT16 versus DN1024/DT48.
The wider artifact duplicates existing recurrent channels with zero new read
columns; it adds no knowledge. Both models retain the known poor-quality Adam25
function. No source/optimizer/GPU/RESERVED/T4 calls. No useful chatbot admission.

[Original protocol](ORIGINAL_NATIVE_ENVELOPE_PROTOCOL_20261009.md),
[missing-only completion](ORIGINAL_NATIVE_ENVELOPE_FINISH_PROTOCOL_20261009.md),
[worker](../../../benchmarks/native_expert_scaling/original_native_envelope.py),
[completion worker](../../../benchmarks/native_expert_scaling/original_native_envelope_finish.py),
[native consumer](../../../benchmarks/native_expert_scaling/original_native_envelope_main.c),
[original binding](original_native_envelope_binding_20261009.json),
[completion binding](original_native_envelope_finish_binding_20261009.json),
[raw result](original_native_envelope_finish_result_20261009.json),
[held terminal](original_native_envelope_finish_result_20261009.terminal.json),
[stored adjudication](original_native_envelope_stored_adjudication_20261009.json).

Original freeze4e42d479/binding93f38c80;completion freeze
`88850ef28ddc3d95dd1f38d60212d08494c93310`,39 inputs,binding SHA
`e7f1c7117a6b1f84d3fdad6acd7c874f3a1df0e46f6d71653715d5476b3537b5`.
Raw SHA `858552d6e71e594595a0eb0cdedf859e8e4eb5279a44e2d01a2e815df3f46bb8`.

## Actual construction and endpoint

New widened packed artifact520,029,440B, SHA
`a06e2619fb5bbe2a96690ed5dba03bc2345bc03799d48485b134630f56daf423`,in
`results/native_expert_scaling/original_native_envelope_20261009/duplicate_dn1024_dt48.packed`.
Original507,505,920B SHA98835350... unchanged.45 SSM fields changed,65 fields
byte-identical,all110 independently verified after export. All bank codes/scales,
router,SWA,embedding/full head,norms retained. Core payload grows12,523,520B;
this is not doubling total stored model size or increasing expert n.

Each source x/gate/conv/decay/delta channel duplicated. Existing x_proj first512
columns and16 delta coordinates retained;new columns/extra32 delta coordinates
zero,B/C rows shifted after48. dt_proj duplicate output rows,zero new columns;
out_proj old512 columns retained,new512 zero. The real-arithmetic function is
identical. This pre-observation choice superseded planned half-weight duplicate
sums to preserve floating operation order. No post-observation criterion change.

All1024 native scan channels execute;dense matrix kernels have no zero-weight
shortcut. New coefficients are runtime data and not compile-time constants.
Nevertheless new read columns are zero and state duplicates;this fixture proves
cost/preservation,not useful learned states or future trained routing/locality.

20 original computational bodies extracted unchanged from engine.c,one existing
ID/mass memcpy observer in mlp_moe;no additional computational-body diff. New
wrapper changes only DN/DTR macro guards and consumer. Two real compiled
executables,one thread/CPU0,O3/AVX2/FMA/znver2,exp_fast1/byte-pair LUT,no OpenMP.
Core/head/router/expert reference F32 copies are not secretly allocated for the
bank;code payload339,738,624B in both artifacts,expert reference bytes0.

## Preservation qualification

Three FULL fixed histories:DEV self_oss_instruct_036/357,FIT magpie_ultra_022/1507,
DEV magpie_ultra_036/1488,total3352;medianDEV/longestFIT/longestDEV frozen selection.
Model loaded once per native process,state reset per request,batch1 forced IDs.
Two native parity passes produce full65537 head/actual top8 IDs/masses eachposition.

**All219,680,024 full-head F32 values per arm,all routes/masses and all18432 LUT
integer coordinates per arm are byte-identical.** Independent stored raw SHA
comparison confirms identity. All3352 row/history relative RMS0,actual ID/mass
delta0,argmax disagreement0. Mass defect max1.9371509552001953e-7<=1e-6.
Both integer witnesses equal the independently bound Adam25 expected witness.
The consumer supports genuine runtime-n/full-vocabulary packed original math.

This native-to-native result does not requalify historical C/GPU/source-quality
failures. Tested function preserves poor Adam25 DEVKL9.01572/disagreement96.5741%.
No new source-relative quality/generation/task observation in this cost family.

## Raw complete batch1 latency

Three unprofiled passes each arm,AB/BA/AB order;one separate profile pass each,
BA order. Timing excludes coldload and full-logit file IO. It includes complete
native forward/full65537 head,argmax/finiteness and route support/mass consumer.
No useful own-history generation;these are raw forced-history rates.

| Fixed history | Tokens | Base case median IDs/s | Wide case median IDs/s |
|---|---:|---:|---:|
| DEV self_oss_instruct_036 | 357 | 121.0020803 | 99.3673722 |
| FIT magpie_ultra_022 | 1507 | 130.4001294 | 117.4912929 |
| DEV magpie_ultra_036 | 1488 | 142.3385174 | 132.3355141 |

| Fixed history/arm | Unprofiled seconds,passes0/1/2 |
|---|---|
| 357/base | 2.671963100 / 3.485097100 / 2.950362500 |
| 357/wide | 3.592728600 / 4.252045200 / 3.302412000 |
| 1507/base | 11.382301000 / 11.556737000 / 11.649529300 |
| 1507/wide | 12.779311000 / 13.060509000 / 12.826482400 |
| 1488/base | 10.453951800 / 9.322291000 / 11.801787300 |
| 1488/wide | 11.244147200 / 12.421276100 / 10.659630300 |

Median aggregate rate=3352/median(sum request seconds per pass),not arithmetic
average of case rates:base136.7704615/wide121.3780898 raw IDs/s. Aggregate slowdown
1.1268134291,about12.68%more elapsed time. Frozen raw affordability gate PASS:
wide minimum-case median99.367>=50 andslowdown1.127<=1.5. No accepted-speed admission.

Times vary noticeably,especially shorter request and DEV1488. No extra sweep was
run after declared repetitions. This is a measured feasible envelope under these
weights/contexts,not a precise universal latency law or large-n DRAM guarantee.

## Component evidence,not an inferred matrix-count speed

One profile perarm. Long1507 FIT profile:

| Exclusive component | Base seconds | Wide seconds |
|---|---:|---:|
| Recurrent scan | .3873642015 | .7404046003 |
| SSM projections/conv/delta/gate/output | 1.5792330992 | 3.4234438988 |
| SWA | .2130107000 | .2052650003 |
| Flat router | 1.1067751009 | 1.0190656028 |
| MoE excluding router | 1.6808908958 | 1.5628670942 |
| Full vocabulary head | 6.3011914014 | 5.8509364992 |
| Norms | .0103018014 | .0101102023 |
| Profile request wall | 11.4303671000 | 12.9277902000 |

Tacc's MoE total includes router;table subtracts it once. Main consumer and
uncounted residual/counter costs explain remaining wall time. Head is about55%
of base profile wall and45%of wide. Wider SSM scan+other is about32%of wide wall.
Unchanged head/router times differ between passes:timing variation prevents
interpreting those differences as improvements. Counted matrix products grow
26,067,200->28,934,400(+11%),scan capacity doubles;actual component times provide
the relevant cost evidence. No scalar-A hoist/new scan kernel was used.

500000 timer reads plus volatile accumulation:profile base .0161268000s/wide
.0145163000s (~32.25/~29.03ns per calibration iteration). Profile wall minus raw
median per357/1507/1488:base+.586849200/-.126369900/-1.148134200s;wide+.498265000/
+.101307800/+.139194800s. This difference includes variation and is not isolated
counter overhead. Coldload profile base .430576400s/wide .372770700s;all other
load/calibration/component observations remain in raw records.

Native state/router allocation1,500,160->2,728,960B. Tracked loader/model/state
heap509,006,080->522,758,400B;consumer mallocs not included in that heap counter,
but held process OS includes them. Physical DRAM bytes **unavailable/null**.
Logical selected pair-code payload2,359,296B/token does not measure physical DRAM.
No cache residency,bandwidth or useful growing n conclusion follows this fixture.

## First fault and missing-only completion

First family:launcher32628/worker13136,creation1791564412.069604,exit1/session92576
CLOSED. AFTER complete wide export/independent110-field verification/base compile
exit0,metadata event raised duplicate seconds keyword. No parity/speed/profile
then. Held42.03100000001723s/worker40.718s;held worker OS1,075,974,144B plus direct
compiler4,780,032B/launcher29,569,024B. Original fault/log/launcherfailure/namespace
kept. Compiler child33084,creation1791564444.9257038,exit0,8.078s. Nested compiler/
linker separate peaks not held. No missing source-runtime fields retroqualified.

Completion adopts actual fixture/extracted bytes/base executable/base receipt;
fixes event name to child_seconds. No export or base compilation replay. New
one wide compile and10 native children=11;one inherited compile,total12. Native
33520 input positions total:6704parity+20112speed+6704profile. No extra dose/test.

Completion launcher27420/worker22436,creation1791564617.31805,exit0/session7845
CLOSED. Held335.719000000041s/worker330.578s;held OSworker1,799,647,232B+max direct
child528,060,416B+launcher29,696,000B=conservative2,357,403,648B. GPU0. All own
scope caps1800s/reserve90/OS4GiB/output3GiB/log4MiB PASS. Nested linker allowed
and observed,its separate memory peak unknown. Both held family times total
**377.7500000000582s**,excluding bind/preparation/commits/stored adjudication.

Original9files520,342,189B,completion46files1,760,785,838B,total55files
2,281,128,027B;wide fixture remains at original path. Actual parity F32 payload
1,757,440,192B. Pre/post all39 inputs/hash/extents;completion output hashes checked.
Independent stored PowerShell validates46new+9old extents/hashes,bit identity,
12child cardinality,20body evidence/timing medians(maxdelta0). No separate held
stored-audit timing/resource qualification. Principal binaries bound,not DLL tree.

## Decision toward useful transfer

Original-style native width expansion is executable and has a measured raw cost
margin. Implement the corresponding **trainable width/optimizer transport** next,
with actual donor-informed Adam25 masters/moments,not quantized-bank dequantization
or random checkpoint substitution. Verify the initial export equals this packed
fixture, unchanged bank masters/moments and correct transformed organ fields.

One actual longest-FIT whole forward/backward/update/export/native comparison must
establish active new channels,actual changed coefficients/finite adjoints and
residency before a larger joint learned-state pilot. Old GPU/native numerical
failures remain;GPU is a training surrogate and native output is the quality
endpoint. New width does not automatically preserve donor information. A finite
quality pilot must test actual source-relative native losses and fresh own-history
tasks,not interpret raw cost PASS as learning success or a reason for endless dose.

No T4 allocation now. Its month+ authorization requires measured training memory/
throughput, concrete recovery hypothesis,finite budget/checkpoints/plateau stops.
Full converter/useful chatbot+same-artifact accepted>=50,useful RAM-driven n/
structured CPU IDs AND normalized mass/physical DRAM/additional donor families
and10B/100B demonstrations remain unfinished. [Resumption](ORIGINAL_ENGINE_TRANSFER_NEXT_20261009.md).
