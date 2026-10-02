# METH-274: I16 shared projections pass full CPU fidelity, stop at cost

Frozen `80835f4`,session26089 exits0; no apparatus repair. All immutable
source/archive/helpers/271 fixture/361 segments and old32/source-row guards
pass. All6144 old269/271 GPU baselines reproduce exactly.

| Check | Result |
| --- | --- |
| All5,505,024 GPU/CPU I16 codes and12,288 scale parameters | Zero discrepancies |
| All59,768,832 AVX2/scalar-I64 integer dots | Zero discrepancies |
| Zero/ties-even/positive-negative I64 extreme reductions | Pass |
| GPU actual-source BF16 mean error | .0001530848670275494 |
| CPU actual-source BF16 mean error | .00015308550458333534 |
| CPU maximum layer energy error | .000186288874829188 |
| Original384 GPU/native relativeL2 median/max |4.7129301464975655e-7 /1.042141004647989e-6|
| All6144 GPU/native relativeL2 median/max |4.6715903465679644e-7 /1.042141004647989e-6|
| Native24-layer median |10.550804ms: **fails<=10ms** |

All original all-state/reserve source and native numeric gates pass both
GPU and actual CPU; source comparisons include candidate BF16 output cast.
The source error differs slightly from unchanged271's.0001530664808342408;
do not call input quantization exact arithmetic. Integer accumulation is
exact for the quantized inputs. CPU proofs precede timing. Timing384 outputs
are bitwise equal the full qualifier's prefix. Passes10.550804/11.348031/
10.175923ms preserve the fixed cost failure; no retiming or budget relaxation.

GPU scoring5.609s after imports,end RSS2,361,421,824bytes,peak allocated
GPU155,440,640bytes. Native compile/qualify/timing18.906s,integer qualification
2.0576851s; peak native365,400,064bytes qualifier/364,748,800bytes timed.
Local3060/six threads,Ryzen5 3600X,clang21.1.8,original flags; GPU sync
before all native work,no active job. Qualification is validation overhead,
not a per-token rate. Actual337,596,452byte physical source fixture unchanged.

Raw [result](meth274_i16_shared_input_result.json),SHA256
`aa2acf6452031d621568cb4712c1a976587755c398af2dc03274ac844cd451cc`.
Quantizer check fixture11,059,220bytes SHA256
`ae70d33d3945c6ad98467eae6196931aa9da302b26d4b18377742fe2091eb70d`.
All6144 native outputs SHA256
`0336e15c46e80f8d4d45816525e16cce8f34de44e635c04fac584f12257ca5ba`.
Timing384 check SHA256
`e85072b8606d899c32fb2a5f731084db02c02a3d97524c17b2a12aadbebb3472`.
Command/equation/gates in [protocol](METH_274_I16_SHARED_INPUT_PROTOCOL_20261002.md).

No complete archive/promotion;267 semantic and271/272 cost stops remain.
The generated shared kernel performs separate gate/up row streams at
different code-array addresses. Their tile-interleaving/paired integer
execution is a new physical-layout hypothesis,not a claimed cost cause or
already faster implementation. Price it with a contemporaneous unchanged
control and bitwise full-output conservation before further composition.
Useful n,RAM route/LUT/real DRAM,whole accepted>=50 and family/10B/100B
applicability remain open.
