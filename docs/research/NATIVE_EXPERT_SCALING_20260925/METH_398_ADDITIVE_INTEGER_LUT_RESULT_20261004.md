# METH-398: exact integer activation LUT fails full-width cost

Prospective freeze6127962, strict physical preflight PASS, execution exit0.
Reuses original GigaChat source controls/317-319 additive representation.
NEW I32 activation tables/AVX2 gathers; SAME input quantizer, full widths,
synthetic books/codes/scales, source head/router/norm/embedding and all outputs.
No waiting/affinity tuning, original donor-adaptation generic runtime paused.

## Result and decision

| Fresh process | Repetition medians, ms | Max/min | Peak RSS, B |
| --- | --- | ---: | ---: |
|1|164.229400 / 164.652600 / 146.102250|1.126968|3,232,075,776|
|2|162.451150 / 137.868250 / 137.688750|1.179843|3,219,894,272|
|3|163.777700 / 162.276600 / 161.533000|1.013896|3,219,881,984|

All90 full operator output/route hashes EXACT319; all numerical/source/capacity
controls PASS. All nine repetition medians and24 fixed-input medians exceed
14ms; pooled max/min1.195831903>1.10. Descriptor542,987,008B still passes560MB.
Thus exact output preservation and compact codebytes do not establish fast
execution. This new realization fails the same cost gate as prior319; its
measured137.689-164.653ms medians also exceed319's38.124-55.526ms range. The
runs were not interleaved paired timing, so no precise causal speed ratio or
hardware attribution is claimed.

**Decision:** close this exact full-width integer activation-LUT kernel before
source-aware codebook fitting or640-bank work. This is not rejection of all
LUT layouts/algorithms or all large expert pools. Table versus row-gather cost,
cache misses, actual physical placement and DRAM traffic were not measured.
No inference throughput/accepted token rate or whole-model quality claim.

## Exact arithmetic and cost boundary

Each two-book dot uses all original8 coefficient positions and exact I32 sums;
integer bounds31752 per book /71124480 at maximum row width. Table builder is
inside EACH timed apply, separately for each active bank and input, including
32 different MLA inputs and four routed-down inputs. A global73,400,320B
workspace is reused; dynamic allocation3,260,346,368B. Newly built logical
I32 table bytes340,000,768/token,357,154,816 gathered values/token; these are
addressed/counts, not physical traffic or a RAM/cache bandwidth floor.

Three fresh six-worker OpenMP processes, PASSIVE wait/KMP_AFFINITY none,
OMP_DYNAMIC FALSE, OMP_PROC_BIND absent, inherited routing/thread variables
sanitized. Actual worker masks were not checked and not claimed. Each has10
fixed inputs/two warmups/three repetitions, all90 recorded outputs. Input
fixture preparation, initialization/loading/hash verification and selftests
outside timers; activation table construction/all283 selected matrix operators/
source controls/SwiGLU/mixtures/lookup/head inside. Input-ready independent
layer/context fixtures exclude causal attention/RoPE/KV and complete residual
composition; codebooks are synthetic and not donor knowledge.

All three old318 scalar/control tests remain:16960 integer/scaled rows,
13240 code/palette bank-edge checks,32131 exhaustive decoded/input cases,
128524 tile lanes/14641 mixed adjacent pairs,25 scalar original routes,
64 source Q6 head rows,1536 exact BF16 lookup values and negative controls.
NEW all262144 index-pair/group cases (256x256 across four groups),
maximum-width positive/negative integer bounds and mutated table detected.
Output arithmetic L2 remains3.578e-8 to independent high precision; all
full output hash comparisons require exact bytes, not that tolerance.

## Bindings and reproduction

Raw `meth398_additive_integer_lut_result.json`, SHA
`44647e766b0918504ff3d3209f88bd6254d452c979bce9079296d81aeb7beb48`.
- cpu_source_sha256: `eba749b10a4634ab4302b862ee6cf67e2f9a496bb8d1ea28811e5f0b855ee893`.
- controller_sha256: `93d7b69a394ee274c9152d1f1d85d1e598fb9213d237e7dca4033750ddb2b3b7`.
- protocol_sha256: `dab1d2faa7f3e612f19a23bd3a6b3e66c7a27ef5a75e079bb3e9bba3c842d6fe`.
- engine_sha256: `01259deb80904cbca842e512e0af4b4122c7ff2f4ae4d78fae423d267121440c`.
- executable_sha256: `af8dc214ea0967943682f07e30b514165f9277c420efaaeff63a667a12f7d8dd`.

Previous319 raw SHA61f5cd62e35cd90e96b75ce74704d9c845741509f669304103a0e1ffc0a93f97;
exact original spec SHA986e8f80abf9ff235158b6202ba57e03ece398636b1bf8554d4e3bf68da075e6.
Fresh Q4 control segments and BF16 embedding hashes/spec match318. Whole old
source-file hashes remain prior records, not newly claimed full rehashes.
Compiler/libomp exact319 and default engine tail exact0ff9705; new opt-in
prefix only. Logs/executable/spec under
`results/native_expert_scaling/meth398_additive_i8`.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth398_additive_integer_lut_preflight.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth398_additive_integer_lut_result.json
```

MAIN44.281s, maximum sampled process RSS3,232,075,776B;20min/12GiB limits met.
No model/native timing overlap, new weight acquisition/training/GPU/T4.
Physical source-bank usefulness/new n and learned codebook quality remain
unproved; independent Switch128/256 qualified artifacts remain authoritative.
