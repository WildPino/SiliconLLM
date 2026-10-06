# PQT-010-R1 frozen-output audit repair, preregistered

6 October 2026. Repair only the failed independent audit of PQT-010. Preserve
the original private kernel137260621/version1/ref
`sirwildpino/pqt-010-shared-parameter-ablation-20261006-002`, first HTTP dispatch
failure, all original raw outputs and the first audit failure. Do not repeat
archive construction, original evaluation, calibration, selection or fitting.
All results remain consumed development diagnosis; no method promotion.

## Observation and question

Terminal ERROR observed03:03:24.963314 UTC. Single retrieval contains168 files /
3,714,350,318 bytes, all size/SHA verified; actual21 embedded scientific source
files equal Git `91145c8d5cd99096974cf94d0529769dcbf59055`. Installation16.065978 s
and evaluation360.047353 s returned0; audit phase6.233696 s returned1, with its
own failure after3.598783 s: RuntimeError / Invalid device argument.
No numerical outcome interpretation or independent qualification yet.

The first audit invokes `pqt_memory.reset()` before explicit CUDA initialization.
Its source directly resets each allocator's peak counter. The pinned
[PyTorch2.11 implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/cuda/memory.py)
calls the native reset after device-index resolution without initializing CUDA.
This is a causal hypothesis to test in a fresh-process startup control, not yet
an independently reproduced cause. The evaluation initialized CUDA before that
same reset; its success does not validate the audit's cold startup.

## Frozen intervention and verification

Change only telemetry reset to call `torch.cuda.init()` before the original loop.
Every other original scientific module remains exact, including the independent
audit's criteria and all six-arm evaluation/payload modules. No numerical budget
or threshold relaxation. Qualification compares the whole corrected telemetry
file against the exact original file plus that single inserted line.

A separate fresh-process control first records CUDA uninitialized state, executes
the exact original telemetry module (raw length/SHA bound), and retains its
reset outcome. Then corrected reset must initialize CUDA and successfully reset
both physical T4 allocator counters. Retain startup/error-type/fixed-message
classification, exact original/corrected code identities, loaded runtime and
post-reset process-memory telemetry. No original weights/function/corpus access.
If the original cold reset does not reproduce the expected error, stop the repair
for adjudication instead of claiming the cause. Corrected reset/control must pass
before the long audit. Synthetic ordering/unchanged-source and input identity/
copy/missing/ambiguous/mutation guards must pass before dispatch.

## Immutable outputs and transport

Bind all required original outputs from the qualified terminal retrieval and
all21 actual original-source identities before dispatch. Use the original kernel
as a private `kernel_sources` input, an interface supported by
[official metadata documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).
Availability/byte preservation of ERROR outputs through this mount is unproven:
stop before original function access if missing, ambiguous or modified. Never
assume latest input content is immutable solely from its reference. Unique mount
resolution and streaming raw SHA/size verification are mandatory.

Create an exclusive copied audit view containing every entry of the original
scientific artifact manifest and the manifest itself, all byte exact. The
original failed audit and bootstrap/source/failure logs remain separately bound
and preserved. Reconstruct only the exact original HF cache paths/files by
download of the immutable checkpoint/tokenizer, verify their declared identities;
do not change original inputs.json. No new corpus access. Separate audit process
imports no writer/driver/fitting modules. Before the unchanged full audit,
independently reverify the complete copied view against the frozen input manifest,
old failure and all original scientific module identities/single-line telemetry
change. Replay all source parameters/tie, untouched168 matrices/vectors, F32/I8
replacement bytes, all states/choices/F32/F64 points, both consumed domains,
exact R1 baseline identities, capacity and every original criterion.

## Resources, retention and decision

Fresh owner/quota/all-addressable-terminal admission; only the frozen historical
zero-ID2010 marker exception is permitted. Same pinned image/runtime, SDPA/F32/
determinism/seed20261005/TF32false. Private acct3 fresh
`sirwildpino/pqt-010-r1-frozen-audit-20261006-001`, installation900 s,
startup control60 s, preparation900 s, audit1800 s, server4500 s.
Retain first missing-mount/startup/numerical failures in exclusive records; do not
restart or silently replace. Delegate status-only long-job monitoring and wait.
Fetch once on terminal and verify actual repair source against committed Git.
Charge original installation/evaluation/failed audit and repair separately;
original fitting1809.718723 s charged once, new optimizer updates0. Corrected
audit peaks are new measurements; failed audit memory remains unknown.

Only a complete passed independent replay permits subsequent standard-library
three-way reaggregation and scoped scientific interpretation. A repaired audit
does not satisfy native timing/integration, expert natural-context transfer,
broad capabilities or the whole research goal.
