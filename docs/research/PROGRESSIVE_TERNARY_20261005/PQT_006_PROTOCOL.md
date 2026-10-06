# PQT-006: same-artifact packed CPU execution and bounded cost

Prospective, 5 October 2026. The previous goal turn made empirical progress:
PQT-005-R1 completed original expert-function comparison, independent replay,
first-failure retention and exact repair reproduction. No tested ternary arm
preserves the function. The full research goal remains active and incomplete.

## Question, artifact and interpretation

Can the exact PQT-005-R1 native pair files execute directly on the available
CPU with correct outputs, and what storage/consultation cost does this
prototype achieve relative to its own original F32 and I4/I8 controls?
Do not repack, refit or introduce a continuous matrix cache. Test both original
Switch experts 8 and 105, source plus D/DR/PR-B/FF/I4/I8, all fourteen files.
Authoritative identities are PQT-005-R1's fetch/native audit and representation
records (scientific source `3215ca3`, original source revision unchanged).
All representations already failed joint quality/storage preservation. Native
speed cannot promote a representation that failed function gates.

Implement original F32, packed two-bit ternary, packed signed I4 and row-I8
matrix/vector paths in C11 AVX2/FMA, exact existing `<8s14I` format, dimensions
768/3072, explicit counted biases/scales. One CPU thread, F32 input/activation
and accumulation, ReLU, no AMP/TF32/GPU or original-source modifications.
Packed codes are expanded eight lanes transiently in SIMD registers; apply
group-64 or row scales. Resident representation is the file bytes plus small
loader metadata, hidden F32 scratch (3072 values), input/output buffers.
No permanent dequantized float weights. Loader rejects reserved codes, wrong
shape/tag/header/length/flags, nonfinite coefficients and negative scales.
Runtime rejects nonfinite inputs, intermediates and outputs.

Compile with local clang 21.1.8, target x86_64-w64-windows-gnu, flags
`-std=c11 -O3 -mavx2 -mfma -fno-strict-aliasing -Wall -Wextra -Werror -shared`.
Explicit FMA and ordinary reductions, no fast-math. Freeze source/protocol
before original-artifact execution; record actual compiler text, command,
source/Git bytes, DLL bytes/hash, CPU identity and artifact hashes. Native
CPU model observed for planning: Ryzen 5 3600X, six cores/twelve logical
processors, reported L3 32 MiB. Observations are not timing reservations.

## Correctness qualification, before timing

Lightweight one-thread check, <=60 seconds wall-clock including imports,
owned scientific CPU environment torch 2.6.0+cpu / NumPy 2.1.3 / Python 3.12.10.
This is arithmetic/codec validation, not a latency benchmark. Preserve its
elapsed time as a safety observation only. No timing ratios from qualification.

Before real artifact use, independent sparse scalar goldens in original fixed
dimensions exercise all four formats, group/row scales, negative codes and
biases. An explicit two-layer case produces known outputs [9,-1.5] in the
first two coordinates and zeros elsewhere; an additional exact sparse case
crosses columns 63/64 and the final Wi/Wo columns, expecting [10.875,-2.25].
Test zero/homogeneity when unbiased
and an independently calculated biased case. Malformed synthetic files check
reserved ternary=11, I4=-8, I8=-128, nonfinite/negative scales, wrong header,
truncation/trailing bytes. Nonfinite input must be rejected without a result.
Use exclusive synthetic files; no alteration of retained real artifacts.

For each real pair: use exactly the first consumed raw-development row for
that expert and the already retained zero input (two rows, twenty-eight calls).
An independent Python decoder reads native bytes directly, uses separate
explicit bit extraction/segment parsing and CPU FP64 full-function reference.
It imports no fitting code. Verify the file hash against the retained GPU
audit, exact decoded arrays against separately retained packed/scales/bias
arrays, source F32 against admitted original tensors, and first-row identity
against retained inputs. Native F32 versus CPU FP64 relative RMS <=1e-5,
maximum absolute difference reported. For exact zero reference, require exact
zero output; biased zero references use the ordinary relative RMS check.
Also compare with the retained GPU F32 outputs under <=1e-5. Save native
outputs, reference values, per-case errors and DLL/source/input identities.
Any rejection/nonfinite/tolerance failure stops timing and must be retained.
This small check verifies the selected arithmetic paths, not all possible
input values or complete-model language behavior.

## Prospective timing, only after resource coordination

Do not launch timing based on an idle process/GPU sample. Prepare qualified
artifacts, DLL and bounded driver first, then obtain an uncontended CPU window
from the current owner. Preserve the actual reservation authority/window;
do not fabricate it. Revalidate at execution, reject relevant competing
benchmark/training processes, pin process to logical CPU 2 (affinity mask 4),
single thread. No process interruption or main-environment writes.

Wall-clock stop <=180s; memory pressure buffer 128 MiB, loaded files ~64 MiB
plus metadata/buffers. Measure consultation only, excluding hash/file loading,
allocation, compilation and eviction traversal; record loading/storage
separately. FFI/timer overhead remains included and is measured separately.
These are prototype single-expert FFN calls, not project router/model latency
or a comparison against the best existing native kernel implementation.

Use the same first raw development row, no selection by latency or error.
Two regimes, all fourteen expert/arm combinations per block, 31 blocks:
(1) repeated resident calls: four warm-up calls, then eight timed calls,
divide whole interval by eight; (2) cache-pressure: traverse an explicit
128-MiB buffer immediately before one timed call, traversal excluded.
Do not claim this proves cold DRAM behavior; report it as cache pressure.
Order randomized per block with Python Random seed 20261011 (resident) and
20261012 (pressure), independent of numerical outcomes. All raw nanosecond
intervals/order retained, no outlier removal. Check output bytes against the
qualified native output after each timed group, rejecting mutation/drift.

Report medians, p10/p90 and per-block ratios versus the same expert's F32
control. Paired bootstrap of median ratio, 10,000 resamples, CPU Random seed
20261013 per fixed expert/arm/regime order, 95% percentile interval. Native
speed signal requires upper interval <=1.0 in both regimes for both experts;
it is separate from the inherited quality/storage gates. A joint usefulness
claim requires all gates, so current ternary artifacts cannot pass it.
Baseline accuracy itself is checked against original FP64 function.

## Evidence and next method dependency

Own paths only; PQT-006 `qualification_001` and later `timing_001` exclusive
namespaces. Keep complete small logs/JSON and array/DLL hashes, first failures,
source/compile/qualification/reservation/timing identities, no source or raw
record overwrite. Document English results and resumable state in
TERNARY_INDEX.md. Do not install into the main environment or launch a local
GPU job. A worktree does not isolate CPU/RAM/disk.

Method research remains necessary: the original function experiment showed
that sparse calibration and hard-update stability must be distinguished.
Current role-1/probe outcomes are consumed; a new prospective function-fit
study needs new fixed training/evaluation identities and actual acceptance
checks before it can support stronger feasibility claims. Complete-model
predictive/generative preservation remains independently unproven.
