# METH-271: output-aware rows pass fidelity, fail native cost

Selector protocol frozen in `8f7ee2e`, apparatus in `96a5629` before
observation. Session5390 exited0; no repair. All6144 consumed states complete.
Even128 states/layer select96 new rows; odd128 remain reserve controls.
Old32 rows,all289 nonprivate fields and all6144 old269 outputs stay exact.

| Metric | Result | Frozen gate |
| --- | --- | --- |
| All-state mean relative squared error | .0001530664808342408; ratio .7758459197787797 | <=.85 original32: pass |
| Odd-reserve mean relative squared error | .00015131114999652104; ratio .7812542646384542 | <=.85 paired32: pass |
| Maximum layer energy error | .00018638461187947541 | <=.00030542892636731267: pass |
| Odd-reserve maximum layer energy error | .00018702991656027734 | <=.00030748543213121593 paired32: pass |
| Native relativeL2 median/max,384 vectors | 5.361424812707872e-7 /1.2371250588431008e-6 | <=1e-4 /5e-4: pass |
| Native24-layer component median | 11.397642ms | <=10ms: **fail** |

Native passes11.397642/10.925507/11.753063ms; six threads on Ryzen5 3600X,
clang21.1.8,original253 flags. GPU scoring finishes/synchronizes before
native execution; no concurrent model job. Component timings are not a
whole-model rate. Preserve the cost stop; no relaxed budget or retiming.
The selector's improvement is22.4154% all-state,21.8746% reserve. These are
consumed component controls,not independent generation/semantic quality.

Fixture337,596,452bytes,8,262,144bytes above252; SHA256
`ece28ea3344d9ecef674d30b4c43b067ee4290379cf9130c97c2eafee1e344d8`.
Native output SHA256
`669cb3d98f175ce575fd8eeb6af8225001d5d647f819f9fe6390c5b0a195ef22`.
Selection/scoring/export22.531s,end RSS2,384,982,016bytes,peak allocated
GPU213,963,264bytes. Native compile/check/timing12.907s,peak working
set354,263,040bytes. No active job remains.

Raw [result](meth271_output_aware_rows_result.json), SHA256
`638594a650189ab45efa62c08be04a3e42b21f7eff00c52d372d5991077120c3`.
All4864 per-layer selector scores,selected/source IDs,fit/reserve controls,
361 segments,native384 errors,compiler/commands/hashes are retained.
Command and exact selector in [protocol](METH_271_OUTPUT_AWARE_ROWS_PROTOCOL_20261002.md).

No complete archive/promotion.259 stays closed by267. A changed execution
kernel can test whether the same actual functions fit the CPU budget;
measure a contemporaneous unchanged kernel control to distinguish observed
operator improvement from historical timing variation. This does not
reclassify271's cost failure or assume a particular cause. Useful learned
n,RAM route/LUT/DRAM,whole quality/accepted>=50 and family transfer remain due.
