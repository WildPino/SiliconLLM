# METH-275: fixed-function paired16 Q8 input layout/execution

## Question and changed variable

274 passes actual6144 CPU source/reserve/numerical and all integer/quantizer
guards but costs10.550804ms,above10ms.273 identifies shared/readout matrix
execution as dominant. Test paired shared-input physical layout: each row
stores16 gate Q8 codes then16 up Q8 codes,repeated across896 columns.
Execute their exact I16 dots in one tile loop,reusing each input vector.
No quantizer/precision/row choices/private features/LUT/readout/rank/bank/
router/alias change; no additional kernel or tuning sweep after failure.
This is physical layout and its matched paired reader,not new useful n.

Bind274 raw SHA
`aa2acf6452031d621568cb4712c1a976587755c398af2dc03274ac844cd451cc`,
source271 fixture SHA
`ece28ea3344d9ecef674d30b4c43b067ee4290379cf9130c97c2eafee1e344d8`,
all361 old segments,source/quantizer/check/include/executable hashes and
125vectors. Original274 code/CPU outputs are frozen inputs. No275 code,
packed fixture,executable or observations exist at this protocol freeze.

## Physical transformation and fidelity

New header `<8s6I>` M275PK01,24,896,4864,32,32,128. All shared gate/up code
bytes remain exact and reversible: tensor[4864,56,2,16],tile-major within
each feature row. Follow each paired-code layer field with original gate
and up FP32 row-scale arrays,then byte-identical original down/private/
residual/bias fields. Preserve original table. Add28 zero padding bytes after
the table: first paired-code offset2112,all layer sizes divisible64. Load
weights into an explicitly64-byte-aligned buffer,so two paired16 tiles occupy
one cache line; do not assume malloc alignment. Total337,596,480bytes (+28
versus271); remove two old code fields and add one pair field per layer plus
padding:338 segments. Every old code matrix must unpack byte exact; all313
nonpaired segments retain original byte hashes,header/EOF/all338 new segments
read back exactly. This format/alignment refinement is fixed before any275
observation or packing execution.
Do not retain duplicate code arrays in the timed native process.

Paired reader uses two I32 vector accumulators,signed I16 madd products and
I64 final lane sums,same proven overflow bound/F32 scale application order.
All6144 quantizer/integer guards and entire all6144 native output including
original M274ALL1 header must be bitwise equal274. This permits inheritance
of274 source/reserve/GPU-native fidelity on consumed states; no fresh model
quality or general arithmetic certificate. Preserve a scalar paired-layout
I64 oracle rather than comparing only two AVX implementations.

## Cost and decision

CPU only after verified no model job. Compile with original flags/sixthreads.
One contemporaneous unchanged274 timing invocation,then one changed275
qualifier,then one changed275 timing invocation. Each timed invocation has
the original384-output warm check then exactly three256-token24-layer
sweeps; all384 outputs and all256-state checksum equal274. Require changed
median<=10ms AND at least5% faster than contemporaneous unchanged control.
No reclassification/retiming of274,relative-only substitute,best pass,
thread/affinity/threshold change or alternate pack widths after failure.

LocalCPU10min packing/binding/compile/qualification/timing,20GiB RSS,
1GiB new output,individualnative timeout180s,no GPU/T4/download. Preserve
all paired timings,CPU/compiler/commands,segments/unpack/source/output/hash
and failure evidence. Freeze apparatus before observation. Pass licenses a
separately versioned complete archive with this exact arithmetic/physical
layout,consumed whole-model regression and newly excluded-source prediction/
generation/task/anonymous checks before native promotion.267 remains closed;
new learned RAM-scale n,dynamic route/LUT/real DRAM/accepted>=50 and family/
10B/100B transfer remain mandatory.

## Apparatus freeze

`meth275_paired_i16.py` binds274 code/executable/quantizer/full outputs and
271 segments,exports the reversible pack with28 zero alignment bytes and
reads all338 segments/header/EOF. `meth275_paired_i16_cpu.c` reads one64-byte-
aligned weight buffer,paired codes/scales,original down/private/residual
fields,then uses the paired integer reader and scalar-I64 paired oracle.
No duplicate gate/up arrays in native memory. All6144 qualifier output
retains M274ALL1 so its entire hash must match274; timing M274OUT1 likewise.
Both sources and this protocol are committed before any275 observation.
No packed artifact/executable/result at this apparatus freeze.

```powershell
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth275_paired_i16.py --binary results/native_expert_scaling/meth275_paired_i16_fixture.bin --exe benchmarks/native_expert_scaling/meth275_paired_i16_cpu.exe --check results/native_expert_scaling/meth275_paired_i16_all.check.bin --control-check results/native_expert_scaling/meth275_unchanged274.check.bin --timing-check results/native_expert_scaling/meth275_paired_i16_timing.check.bin --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth275_paired_i16_result.json
```
