# METH-296: actual native generation health/cache/state capture PASS

Freeze `cf84f90`, session97155 exits0; all12 fixed apparatus/health gates
pass. SAME276 archive/725 fields/original285 profile, SAME24 new292 prompts,
all72 actual unpenalized greedy128-cap full-head continuations retained.
Original BF16 donor/E1280 use full-prefix recomputation in a separate GPU
process; it exits before actual phase60 native cached inference. Parent
never initializes CUDA. No source/checkpoint fallback or oracle injection.

| Arm | EOS/24 | Early non-EOS under16 | Repeated8gram3x |
| --- | ---: | ---: | ---: |
| BF16 donor | 21 | 0 | 0 |
| BF16 E1280 | 22 | 0 | 1 |
| Actual native | 23 | 0 | 0 |

All pooled/category health limits pass. Independent stdlib recomputation
from saved continuation IDs reproduces every EOS/early/repetition count
exactly. Labeled continuation texts have NOT been inspected; this is health,
not semantic preservation.

All4,264 native fixed-prompt top1 IDs match actual295 exactly; all24 first
full-head rows are byte exact. The first generated choice/state equals the
last prompt choice/state. At source indices0/8/16, first three actually
generated states each: all9 independently rebuilt whole prefixes give
BYTE-exact normalized hidden and full-head logits to real cached execution.
All3 actual erased-history negative controls change both arrays and detect
the fault. This is scoped cache correctness, not universal-family parity.

All4,264 prompt and1,672 actual generated normalized BF16 states are saved
with real full-head choices/logits/position metadata. Native binary
25,277,748bytes is retained for subsequent actual CPU K64 inclusion/rerank
checks on these states. No GPU proposal pass or CPU K64 result is inferred.

Total500.719s after imports: GPU controls272.984s/endRSS2,402,447,360bytes/
peak allocated3,081,579,520bytes; actual CPU205.109s/peak childRSS
1,450,131,456bytes. Budgets pass, no T4/downloads. Whole assay cost includes
all-prompt head/state capture and independent prefix/negative controls;
it is NOT accepted decode-rate measurement.

[Raw result](meth296_native_generation_result.json) SHA
`87a2b7988eef513271186095a96167217ef9826ccab5444993c8be75ed9c0bb7`;
[protocol](METH_296_NATIVE_GENERATION_PROTOCOL_20261002.md). Raw binds all
actual output/source/entry/executable/control/binary hashes, compile command,
loader report, source/prompt IDs and resource readings. Generated token
IDs/texts are preserved for separately frozen arm-anonymous review.

Decision: freeze anonymous semantic panel/review on these SAME72 outputs,
all findings committed before map unblinding, unchanged strict six pooled
unsupported/severe/missing-detail counts<=BOTH controls. Then full1838 PIQA,
CPU K64 and accepted>=50 on SAME artifact remain. Original numerical5%
failures retained under292's explicit revision; diagnostic/unpromoted.
No useful tenfold new capacity/large-RAM/LUT/DRAM/family/10B/100B proof.
