# METH-393: exact router full scores, oracle shortlist normalization rejected

Frozen6e84efc, physical committed-input/source/default preflight PASS.
Session92450 exit0 fully consumed;292.891s/max checked combined RSS1,408,495,616B.
ALL8 apparatus gates PASS. New opt-in capture C reverses EXACT388; earlier
engine paths exact37af5e7/default0ff9705. Both complete payload/spec hashes fresh.
Actual physical workers[0,2,4]/ACTIVE/infinite, live placement and negative fault
checks PASS. Instrumentation copies normalized input/full F32 scores AFTER
unchanged router mv; no model arithmetic or source weights changed.

ALL384 measured cohort teacher/natural complete states/logits/routes hashes AND
natural IDs EXACT363/387/389; engineering profile/long controls also exact.
Every full-score selected ID exactly matches native output and independently
reconstructed selected F32 probability has relative error EXACT0. Invalid trace
index and NaN score controls detected. All actual trace inputs/scores finite,
sequence/phase/length/bank order legal. No original quality cohort rescored.

| Original n / mode | Router queries | Minimum oracle m for <=1% scale: median /95th /max |
| --- | ---: | ---: |
|256 / teacher |24,768 |140 /191 /224 |
|256 / own-natural |23,094 |141 /191 /224 |
|128 / teacher |24,768 |71 /98 /115 |
|128 / own-natural |23,556 |71 /97 /114 |

Oracle ranking already knows ALL full scores; this is an ideal lower obstacle,
not an accelerated lookup. At256 teacher/natural, top64-only normalization
inflates the selected multiplier by medians11.68%/11.95%,95th60.51%/57.16%.
24,012/24,768 and22,519/23,094 queries exceed1% scaling. Even top128 still
exceeds1% on15,644/24,768 and14,967/23,094 queries. At128 top64 medians1.43%/1.39%,
95th5.53%/4.98%. These are local algebraic changes, not measured whole-model
harm. Different pretrained sources/cohorts do not establish a causal n curve.

Raw `meth393_switch_router_audit_result.json`, SHA256
`bea3d2e1db10f03d5d9de3173d71a455142166ae71229bbb847959baf12ce07d`.
Retained local complete outputs/trace/log/binary bytes1,748,802,650 (<6GiB).
30min/16GiB/disk guards held; no concurrent model/timing/download/GPU/T4.
Diagnostic binary SHA67cc62872ea51748f2f92818567dbfcc54c94a152606dfde674ad65a5f3b4afd.
Capture I/O is charged inside its timers; no accepted-rate or native-cost
qualification from these instrumented times. Qualified374/389 binaries remain
separate and their376/391 accepted rates remain authoritative.

Decision: frozen criterion95th minimum oracle m>64 holds for BOTH source/modes;
reject unchanged top64-only normalization under the1% diagnostic scaling budget.
Preserving original top1 identity is insufficient to inherit quality. Next
screen a compact full-score representation with candidate exact refinement
and normalization over all approximated scores, using these actual inputs.
Only a prospective NEW whole-quality cohort and SAMEartifact rate can promote
an approximate router. No useful>256/hierarchy speed/LUT/DRAM/other-family/~100B
claim follows. Full final goal incomplete.
