# METH-400: joint blocked integer LUT is exact but remains too slow

Freeze26ff048, execution exit0.399's two independent bottlenecks motivated a
joint change: column-major code planes, four-palette SIMD builder, thread-local
8-group I32 tables and per-block partial integer reduction. SAME source-sized
GigaChat318 synthetic coefficients/books/scales/precision/full operator geometry
plus actual source head/router/norm/BF16 embedding. No donor knowledge fitted.

## Gates and decision

| Fresh process | Repetition medians, ms | Max/min | Peak RSS, B |
| --- | --- | ---: | ---: |
|1|120.704500 /121.135250 /120.503400|1.005243|3,229,372,416|
|2|118.696600 /119.165250 /124.517900|1.049044|3,217,182,720|
|3|120.407700 /121.725550 /119.576850|1.017969|3,217,190,912|

ALL90 full64-bit mixed output/route fingerprints EXACT319 (also398/399);
independent integer/scaled/source/head/router/lookup, LUT controls and code-bank
edge tests PASS. Every16960 sampled row additionally reconstructs all its codes
from column planes and checks the independent original seeded byte indexing.
These tests/fingerprints are not exhaustive archived full output-byte comparison
or learned original-relative whole quality. Capacity bytes unchanged.

Pooled repeat1.049043528 and within ratios PASS1.10. All nine medians and24
fixed-input medians exceed14ms: cost FAIL despite stability. Median range
118.6966-124.5179ms. Previous398/399 runs were not interleaved paired comparisons;
no precise causal speedup ratio or hardware attribution claimed.400 component
attribution/physical traffic was not measured.

**Decision:** close this exact joint full-width kernel before codebook fitting
or640-bank work. No accepted-token/s or whole-model quality promotion. Further
synthetic full-width polishing needs a new identified decision/uncertainty;
return to real pretrained capacity and a stated transfer mechanism.

## Exact math, workspace and scope

SIMD builder shifts palette[-63,63]+64 to unsigned[1,127], exact paired products
<=16002/I16, subtracts64*sum(input8). All262144 exact two-index/group cases,
maximum-width positive/negative dots and mutated table detected. Original
activation quantizer[-63,63]/nearest-even and F32 scaling unchanged. Per64-column
partial max508032/full row max71124480, exact I32 accumulation order independent.
Each worker's job builds16KiB table for its input/bank block and consumes ALL
rows8 at a time from column planes; all jobs and final reduction timed. Tables
for32 MLA/four down inputs remain independent. No assumed cache residency.

Derived/reused global partial workspace860,160B; per-thread table16,384B.
Dynamic allocation3,187,806,208B.
Logical table writes340000768B/token and357154816 I32 gathered values/token
remain separate from physical DRAM. Complete descriptor542987008B<560MB;
synthetic stored/active coefficient positions10275979264/1428619264 unchanged.
Source-sized64parent/top4/26-layer/D1536/fullMLA32/fullFF1280/dense08960.
Input-ready per-layer residual/context fixtures exclude causal attention/RoPE/
KV/full residual composition and do not prove useful additional expert capacity.

Three fresh six-worker OpenMP/PASSIVE/DYNAMIC false/KMP_AFFINITY none processes,
PROC_BIND absent/inherited settings sanitized. No placement/wait tuning, actual
physical masks unmeasured. Initialization/fixture prep/source binding/selftests/
fingerprint/serialization outside timers, all matrix/control/head work inside.

## Resources and reproduction

MAIN49.141s, maximum process RSS3229372416B;20min/12GiB/compile120s limits met.
No model/native timing overlap/new weights/training/GPU/T4. Fresh source control
and BF16 embedding hashes/spec exact318/319, whole old archives prior hash only.
Opt-in engine prefix/default0ff9705 and prior qualified Switch branches retained.
Raw `meth400_blocked_integer_lut_result.json`, SHA
`2ab887259f1320cc3ec73c0476d6f860cc6e59aa081d6e62eafb7de4eacde0cb`.

- cpu_source_sha256: `a16607e3328e20c0b8097039ea8ecb751a4f7cb841938801fc9e47b3917dce71`.
- controller_sha256: `3dcf808410c5169cb44481e18f77685a6e9cf435ad4705cbae79cc173929751e`.
- protocol_sha256: `9fdf6ab9b38f0efce76374c005ab3463598ec4d1f17c4d5c42d46275f01c5e5b`.
- engine_sha256: `e67a31fa08efe49630e1f0b2d5ba449d921d1e76e6383deba77a021aea968273`.
- executable_sha256: `48b900d20a1653c4a09b0554e58a5be0e54a91ca10525ac4db2b11bacc27c3b5`.

Spec/exe/logs: `results/native_expert_scaling/meth400_additive_i8`.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth400_blocked_integer_lut_preflight.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth400_blocked_integer_lut_result.json
```

Qualified pretrained Switch128/256 complete artifacts/rates remain authoritative;
GigaChat full-width learned transfer/final generality/large useful n still open.
