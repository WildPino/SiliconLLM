# METH-398: exact additive representation via integer activation LUT

Prospective, before observations. Resolve whether an exact new arithmetic
realization of318's compact full-width representation can meet319's native
14ms operator budget, before codebook fitting or larger useful-bank work.
Reuses donor-adaptation's bound GigaChat source controls and317-319 evidence.
Generic donor port remains paused; no new training/acquisition/T4.

## Changed variable and fixed inputs

SAME318 synthetic code/palette/scales and full1536/26/64-parent/top4 geometry;
actual Q6_K head/F32 router/norm/BF16 lookup source. No reduced MLA channels,
rank, expert width, precision, output dimensions, extra labels or changed math.
NEW per-activation I32 tables: for every8 input values, both256-entry I8 books
have exact8-product integer sums. AVX2 gathers two table sums per8 weights,
then same integer accumulation and F32 scale sequence. No decoded I8-weight
reconstruction during row dots. Tables separately built for each selected
bank and each32 MLA/four routed-down input; no implicit cross-input sharing.
The builder is INSIDE every timed apply. Full prior output hashes must match.

Unlike300's two-coefficient U8/F32 palette and301's four-coefficient one-book
format, this has two indices per8 coefficients, two I8 books, integer dot
bounds and AVX2 table gathers. SAME317 addressed-weight descriptor542,987,008B;
NEW340,000,768B logical table writes/token and357,154,816 gathered I32 values.
These do not measure physical traffic or a bandwidth floor. Workspace bound
32*8960*256=73,400,320B, reused between matrices. Actual new allocation/RSS
recorded separately. Stored conditional capacity is unchanged and synthetic.

Each palette dot<=8*63*63=31,752, full accumulated row<=8960*126*63=71,124,480;
I32 exact bounds, no saturation. Activation quantizer unchanged nearest-even
[-63,63], not SwitchA16. Exact full output identity is required, not an L2
allowance for replacing integer arithmetic.

Bind committed newC/controller/protocol/engine and previous318/319/303/300/
reference helpers,317 raw,318 first failure and319 raw.319 raw SHA
61f5cd62e35cd90e96b75ce74704d9c845741509f669304103a0e1ffc0a93f97;
original spec SHA986e8f80abf9ff235158b6202ba57e03ece398636b1bf8554d4e3bf68da075e6.
Fresh source segment and embedding hashes/spec exactly318. Whole old source
archive SHA is prior evidence, not falsely claimed freshly rechecked.
Engine new opt-in prefix only; default tail exact0ff9705. Clang/libomp exact319.

## Controls, decision and resources

Three fresh processes; fixed10 inputs/two warmups,3 repeats/process; all90
full operator output/route hashes EXACT319. Old16960 scalar output samples,
all bank/palette offsets, scalar original router/head/embedding and old
quantizer/extrema/exhaustive controls remain. NEW262144 exact index-pair/group
cases (ALL256x256 pairs, ALLfour groups), maximum-width integer positive/
negative extrema and mutated-LUT fault detection. Invalid bank fails before
building tables. Any apparatus/numeric/hash failure stops and is retained.

Same declared319 execution profile: six OpenMP workers, PASSIVE wait,
OMP_DYNAMIC FALSE, KMP_AFFINITY none, OMP_PROC_BIND absent. Sanitize inherited
OMP/KMP/GOMP/worker-binding variables. No placement/wait tuning, no inherited
Switch rate claim. No model job overlaps native timing. Exact actual six
physical placements are not established by this profile and not claimed.

Cost criteria unchanged319: EACH nine repetition medians and24 fixed-input
medians<=14ms; within and pooled max/min<=1.10. Failure closes this exact
kernel/geometry before learned codebook/640-bank work. PASS licenses only
separately frozen source-aware fitting/large-bank cost; synthetic weights and
input-ready independent layer fixtures do not establish a causal model,
quality, useful n or accepted50 tokens/s.

MAIN<=20min, process RSS<=12GiB; compile<=120s; expected<=2min/3.3GiB.
No source-weight acquisition/GPU/T4. Compile/execution records and first
failure retained before any repair. Scientific source/protocol immutable.

```powershell
.venv\Scripts\python.exe benchmarks\native_expert_scaling\meth398_additive_integer_lut_preflight.py --out docs\research\NATIVE_EXPERT_SCALING_20260925\meth398_additive_integer_lut_result.json
```
