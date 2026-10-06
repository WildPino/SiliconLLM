# PQT-006: direct packed CPU arithmetic qualified, timing pending

5 October 2026. [Protocol](PQT_006_PROTOCOL.md), metadata-only
[R1](PQT_006_R1_PROTOCOL.md)/[R2](PQT_006_R2_PROTOCOL.md) repairs,
[passed qualification](PQT_006_R2_QUALIFICATION.json),
[raw retention](PQT_006_RETENTION.json).

## Established result

The exact fourteen original/converted native pair files from PQT-005-R1 now
execute directly in a one-thread AVX2/FMA C prototype. Packed weights are
expanded eight values transiently in SIMD registers, with explicit group/row
scales and biases; no persistent dequantized float matrix is created by the
packed runtime. The independent verifier uses temporary dense FP64 matrices
only for reference arithmetic. Original F32 control uses its actual F32 file.

Both experts and every arm pass unchanged <=1e-5 numerical tolerances on the
first consumed raw development row and zero row: twenty-eight real cases.
The largest deviation from independent CPU FP64 reference is 1.6799701e-7;
largest deviation from retained GPU F32 output is 3.6754165e-7. These establish
sampled implementation correctness, not preservation of the original expert
by the lossy representation. PQT-005-R1 quality/storage failures still apply.

| Arm | Max native/FP64 RMS | Max native/GPU F32 RMS | Pair file bytes | File plus C loader bytes |
| --- | ---: | ---: | ---: | ---: |
| Original F32 | 1.391e-7 | 3.629e-7 | 18,874,432 | 18,874,496 |
| D | 1.256e-7 | 2.984e-7 | 1,474,624 | 1,474,688 |
| DR | 1.058e-7 | 3.342e-7 | 1,474,624 | 1,474,688 |
| PR-B | 1.266e-7 | 3.648e-7 | 1,489,984 | 1,490,048 |
| FF | 1.333e-7 | 3.675e-7 | 1,489,984 | 1,490,048 |
| I4 | 1.383e-7 | 3.274e-7 | 2,654,272 | 2,654,336 |
| I8 | 1.680e-7 | 3.546e-7 | 4,734,016 | 4,734,080 |

The loader adds 64 bytes per pair to the already counted 64-byte file header.
Hidden scratch is 12,288 bytes; each F32 input/output is 3,072 bytes. This
does not count shared DLL/CRT/process overhead, allocator bookkeeping or
reference-verifier memory as representation storage. DLL is 74,240 bytes.
GPU arithmetic quality was verified on larger groups in PQT-005-R1; this
native check does not cover every possible input or a complete language model.

Independent sparse scalar cases pass for all four formats: exact [9,-1.5]
outputs, zero/homogeneity, biased cases, and group/lane/final-column boundary
case [10.875,-2.25]. Malformed dimensions/header/lengths, reserved ternary/I4/I8
codes, nonfinite original coefficients or scales, negative scales and nonfinite
inputs are rejected. Every native segment is compared with retained NPY bytes;
all 276 prior terminal files and original dataset identities are rehashed.

## Execution and first admission failures

Frozen successful source `5d27bc5`, clang 21.1.8 x86_64-w64-windows-gnu, original
frozen flags, DLL SHA-256 `e755d2237b7274f4e83eeeb457ada05257b98f643644920a0361201895657d83`.
Qualification completed in 3.891s including imports, below its 60-second safety
stop. This elapsed observation is **not a consultation-time measurement**.
Owned CPU scientific runtime 3.12.10 / torch 2.6.0+cpu / NumPy 2.1.3, one thread.
No GPU job, main-checkout/environment mutation or process interruption.

First source `e07a85b` compiled, but process inventory returned exit 1 for
absent requested process names. R1 `efafccf` corrected enumeration but
incorrectly classified its own venv launcher as competition. Both stopped
before arithmetic controls or real source-function access. The bounded
launcher probe and R2 live snapshot confirm immediate parent and executable
identities; only that verified owned launcher and caller are excluded, with
other relevant processes still blocking execution. Neither exclusion nor an
empty inventory proves authority for an uncontended timing window.

Original `qualification_001/002` records/DLL/logs are retained. Their C and
native decoder/fixture source bytes equal the successful third build; repairs
changed resource/record handling only. The first executor snapshotted external
stdout before its final print, so terminal stdout differs from that internal
snapshot; preserve both identities and bind final bytes in retention. Subsequent
snapshots exclude open external logs. No numeric tolerances or cases were relaxed.

## Remaining work and resumption

Actual native consultation timing is prepared but **not executed**. The driver
requires a live owner-provided uncontended CPU window; no unanswered question,
idle sample or elapsed wait is reservation authority. Owner coordination has
been requested for five minutes; benchmark stop 180s, one logical processor,
128-MiB cache-pressure buffer and ~64 MiB loaded representations. Two fixed
regimes/31 blocks/randomized orders, raw intervals and paired-bootstrap ratio
criteria are prospectively frozen in the protocol. No outlier removal.

Run after actual coordination using the owned scientific Python:
`benchmarks/progressive_ternary/time_native.py --qualification
results/progressive_ternary/PQT-006/qualification_003 --reservation <actual record>`.
Keep stdout/stderr outside `timing_001` to bind terminal evidence independently
of executor-open log streams. A stale reservation must be renewed, never
inferred. Current source/protocol hashes must still match the qualified build.
No positive useful-capacity claim is possible for current failed-quality arms.

The next GPU scope should address complete-model conversion/prediction and
changed-state generation rather than extend only one-projection evidence.
Preserve a new independent-of-fitting evaluation corpus and unchanged final
gates before source access. Keep original expert-function and native evidence
distinct from whole-model language quality. Goal active and incomplete.
