# METH-400: joint blocked code layout and SIMD integer LUT builder

Prospective, before observations.399 retained32802d2 resolves both398 bottlenecks:
builder52-70ms and rows110-149ms EACH exceed14ms in allnine repetition medians.
A one-component change is insufficient. Test a joint NEW exact algorithm/layout
on locally bound GigaChat source-sized geometry before learned fitting/large n.
This is not a repeat of398/319 or wait/affinity tuning. Final goal still requires
real useful learned capacity/original-relative whole quality and complete rate.

## Exact transformation and new algorithm

SAME318/319 synthetic palettes/codes/F32 scales/activation nearest-even[-63,63],
fullD1536/26layers/64parents/top4/MLA32/shared/dense0/head/router/norm/lookup.
No coefficient truncation/precision/rank/output-width or parent-count change.
Reindex original row-major twoU8-code groups into per-bank column-major code
planes: original(bank,row,group,book) -> same(bank,group,book,row). Initialization
generates the SAME original seeded code bytes into new locations; not extra
stored capacity. Every16960 sampled row reconstructs all its codes and checks
against independent original-generator indexing; bank-edge checks retained.
ALL90 full64-bit output/route fingerprints EXACT319, transitive398/399;
NOT archived/exhaustive whole-output byte comparisons. Controls remain separate
from the existence of real pretrained knowledge in these synthetic codebooks.

SIMD LUT builder: process four palette vectors8-wide together. Palette I8[-63,63]
shift+64 -> unsigned[1,127], multiply by unchanged signed activations[-63,63]
using maddubs/madd, combine four dot sums and subtract64*sum(input8). Adjacent
products<=16002 so I16 saturation impossible. Each book dot<=31752. Same output
I32 LUT values; all262144 index-pair/group cases/MAX_INPUT positive/negative
extrema/mutated-table/old scalar controls retained. Unsupported saturation test
still concerns old decoder's fullI8 activation negative control, not new support.

Per matrix, parallel jobs = selected-bank x input64-column block. Each job
builds eight-group/two-book512-entry I32 tables on its own16KiB stack storage,
then consumes them for ALL output rows in eight-row vectors from contiguous
column code planes. Partial I32 output vectors saved per job. After the region,
reduce all partials and apply SAME F32 scaling sequence. Builder/reduction/
OpenMP dispatch ALL inside timed apply. Inputs remain independently separate
for32 MLA heads and four routed-down activations. No implicit table reuse.

Per block max64*126*63=508032; full row max8960*126*63=71124480: all I32
accumulation exact/no order-sensitive floating sum. Original palette input data
read; no assumed cache residency/bandwidth benefit. Reused global partial-output
workspace max from ALL283 source-shaped selected operators, computed in source
and independently from317 catalogue. Record actual allocation/RSS separately.
Logical table writes/gather count unchanged398:340000768B and357154816 I32
values/token; layout/algorithm changes actual execution, not their descriptor.

## Bindings, gates and stopping rule

Exact committed newC/controller/protocol/engine/helpers/317/318 failure/319/
398 source/399 raw before compile.399 rawSHA
0749edfb40ec44a5ddecf0c943cc9d2df9821e2787c486e23170d19de40e5314.
Spec/fresh actual source controls/BF16 embedding exact318; compiler/libomp319.
Default engine tail exact0ff9705 and qualified old prefixes preserved. Source
archive whole hashes remain prior evidence, not newly claimed full rehashes.

Three fresh processes,10 fixed inputs/two warmups/three repeats each. Six
OpenMP/PASSIVE/DYNAMIC false/KMP_AFFINITY none, PROC_BIND absent/sanitized inherited
thread variables; no profile tuning or physical-mask claim. ALL90 fingerprints
exact319 and independent source/integer/LUT/negative/bank/capacity tests pass.
Original four-row decoder tests remain retained controls; new row traversal8
declared separately. Actual new ready row_tile8/allocation independently verified.

Cost gates unchanged319/398: EACH nine repetition medians and24 fixed-input
medians<=14ms; within and pooled max/min<=1.10. Any apparatus failure retained
before repair; failed cost closes this exact kernel before codebook training or
640-bank allocation. PASS licenses only separately frozen source-aware fitting/
larger-pool cost, not useful new capacity or whole-model quality/50tokens/s.

MAIN<=20min/process RSS<=12GiB/compile<=120s; expected<=2min/3.3GiB. No model
job overlaps native timing, no new source-weight acquisition/training/GPU/T4.
Independent per-layer/context fixtures exclude causal attention/RoPE/KV/full
layer composition; actual head/router/source controls do not supply donor
knowledge to synthetic matrix codes. Source-specific original quality remains
required for any learned transfer. No physical DRAM measured by this test.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth400_blocked_integer_lut_preflight.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth400_blocked_integer_lut_result.json
```
