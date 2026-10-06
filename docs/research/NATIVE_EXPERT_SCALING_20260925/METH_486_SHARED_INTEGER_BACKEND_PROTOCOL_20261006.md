# METH486: one complete shared integer GPU inquiry

6 October 2026. Prospective fixed protocol, frozen before compilation, CUDA
loading, runtime initialization, numerical controls or model execution.
Goal ACTIVE/INCOMPLETE. Existing RTX3060 only; no SDK, package, download or T4.

## Decision and new variable

484 admitted ALL original descriptors, 169 eligible shared matrices, exactly
166232064B per artifact, and a 3.75MiB maximum legal GPU operator scratch.
458 admitted matched source cost; 387/363 admitted original donor-relative
quality on ALL24 books x4 cases for original distinct Switch n128/n256.
The new variable is a complete exact integer GPU backend in engine.c.
CPU0/GPU1 use the SAME newly compiled486 binary. This scientifically necessary
new-variable counterfactual checks temporal machine effects and integration;
it does not invoke any completed458/374/388 executable, main, audit or controls.

An exact faster shared core is an execution enabler. It does not establish
convenient pretrained capacity transfer, causal useful-n, sublinear CPU LUT
winner AND normalization mass, cold physical DRAM or another family/100B.
All those joint requirements remain open regardless of the result.

## One algebra and layout

Original CPU RNE quantization/clamp [-32767,32767] remains unchanged:
F32 maximum/32767, F32 division, lrintf, clamp. For signed16 code q, unsigned
representation u and sign s=(u>=32768):

```
q0=u&127; q1=(u>>7)&127; q2=(u>>14)-4*s
q=q0+128*q1+16384*q2
Wq=P0+128*P1+16384*P2, Pj=Wqj
```

q0/q1 in [0,127], q2 in [-2,1]. For W in [-128,127] and D<=4096,
partial bounds16256D/256D fit I32; weighted intermediates<=6291328D fit I64.
Source-valid AVX I32 lane bound512*128*32767=2147418112<INT32_MAX.
Final dot bound128*32767*4096=17179344896<2^34, exactly representable F64.
The AVX bound is NOT extended to q=-32768 combined with W=-128 at D4096.
The existing484 all-signed16 identity is reused; it is not replayed.
CPU result uses unchanged order `(float)(((double)dot*row_scale)*query_scale)`.

One fixed layout: pad M,D to8; original row-major W interpreted column-major
[Dpad,Mpad], opA=T; column-major B[Dpad,4Q] contains q0/q1/q2/zero for each
query; opB=N; C[Mpad,4Q] I32. Fixed cublasGemmEx COMPUTE_32I=72,
CUDA_R_8I=3, CUDA_R_32I=10, alphaI32=1,betaI32=0, GEMM_DEFAULT=-1.
Default means one library heuristic policy, not a claimed fixed internal
kernel. No algorithm, tile, shape or repeated performance sweep.
Host pointer mode, default math, null stream read back; explicit8MiB workspace.
Blocking cudaMemcpy and explicit synchronization before CPU reconstruction.

Primary API contracts: [cuBLAS12.4.1](https://docs.nvidia.com/cuda/archive/12.4.1/cublas/index.html),
[runtime memory12.4.1](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__MEMORY.html),
[NVIDIA compute enum](https://docs.nvidia.com/cuda/nvmath-python/0.5.0/bindings/generated/nvmath.bindings.cublas.ComputeType.html),
[NVIDIA data types](https://github.com/NVIDIA/nvmath-python/blob/main/nvmath/_utils.py).
Minimal explicit Win64 ABI uses dynamic symbol loading; no CUDA headers are
installed. Static size checks plus actual statuses/modes and ALL controls
qualify this ABI/layout, not DLL file presence or guessed version strings.

## Complete apparatus before observation

Freeze486 C/backend/entry and generated model/cost/generation copies, reversible
derivation, source generator, main/checks, independent auditor, binding builder,
both Windows terminal scripts, finalizer and this protocol in one source commit.
Then freeze metadata/runtime/input binding in a second commit before sole main.
Pin compiler/libomp, cuBLAS/cudart/driver and installed driver support DLLs,
Python executable/base DLLs/stdlib/NumPy/psutil/Windows runtime, source dependencies,
484/458/387/363/cohort records and original reference physical paths/sizes/digests.
Main refreshes FULL22.36GB payload digest and ALL used inherited references
before CUDA loading. No Torch/SciPy/model/tokenizer import or new data fitting.

This is the numbered complete repair of admitted485 first ABI fault, not a
replay of that attempted namespace. Bind `cublasSetWorkspace_v2`, verified in
the actual pinned Win64 PE exports. Before main/CUDA, metadata binding verifies
ALL25 requested exports independently of DLL loading. Source generator reverses
original388 bytes exactly;486 engine prefix reverses to qualified485 engine.
Aware ISO instants are compared as datetimes in the auditor, preserving its
PID/create/time bounds and all numerical/resource criteria. Parent main/control/
original audit/audit-R1 namespaces remain immutable and MUST NOT be invoked.
[Repair provenance](METH_486_NUMBERED_BACKEND_REPAIR_20261006.json).

## Ordered gates and controls

1. ALL frozen sources/runtime/full payload/reference digests; no concurrent
   science (except the exact authorized publisher daemon), one compilation.
2. Actual single RTX3060 CC8.6, runtime12040 and library version/status/module
   paths, integer ABI and host/math/stream/workspace. No fallback backend.
3. ALL eleven fixed controls before whole inference:
   (M,D,Q,pattern)=(1,1,1,0),(17,15,1,1),(8,16,1,2),(9,17,1,3),
   (8,4096,1,4),(768,768,1,0),(768,768,29,1),(768,768,256,3),
   (3072,768,1,4),(768,3072,1,2),(32128,768,1,3).
   Patterns zero, signed extremes/cancellation, RNE half integers,
   deterministic F32 inputs and scales, W=-128 full-width extreme.
   Main checks ALL input/quantizer/digit/GPU partial/pad/zero/reconstructedI64/
   original scalar+AVX/finalF32 bytes. Independent auditor rederives with signed
   Euclidean divmod and full I64 products; never imports main/checks or CUDA.
4. Register only169 shared attention/denseFFN/head matrices. Original CPU F32
   routers/F64 dots, I8/A16 experts, norms/ReLU/cache/acceptance/ties remain.
   Exact reversal of generated source and preceding485 engine prefix required.
5. BOTH artifacts ALL96 cases each, teacherCPU0/GPU1 (warm0/reps1), natural
   CPU0/GPU1 x profile0/1 (warm1/reps3). Fixed book/case order; backend and
   profile order alternate by case index. Native execution uses OWN previous
   states. ALL3456 whole outputs (2teacher+16natural per case) match qualified
   original source bytes including state/logits/routeIDs/acceptance/probabilities.
   ALL counters, stop rules, generated IDs, healthy counts and worker masks
   match bound source references. Any missing/mismatched case fails.

## Quality and accepted-rate definitions

Quality uses exact new whole teacher/ownstate natural source bytes, verified
against physical archived outputs and ALL original donor NPZ digests. This
transitively preserves the admitted387/363 donor comparisons on SAME heldout
cases; no claim of new books, generalization, new donor execution or capacity.
Fresh teacher NLL/choices rederived against stored native metrics; fresh greedy
IDs/healthy acceptance and prose counts rederived. All18 original quality gates
remain explicit: all/masked NLL upper95<=.05; known-token lower95>=-.02;
field lower95>=-.05; all/masked original-top1>=.95; original field signal>=.10;
original/native complete fraction>=.80; generation known-token>=-.02,
field>=-.05,LCS>=-.02,health>=-.05; trigram upper95<=.05, prose-edit<=.10,
plus original bridges/native bridges/ALL96 tasks. No approximate quality waiver.

Healthy source tasks close five ordered sentinels, four nonempty fields, final
32095, no malformed positions, within-field repeated trigrams<=.5. StopsEOS1,
closing32095 or cap64. ALL rejected request TIME stays in denominator.
Ordinary accepted IDs include sentinels; prose IDs satisfy1<id<32000.
Source totals n128=(96healthy,1142ordinary,662prose), n256=(81,895,490).

Primary complete computation rate is SUM accepted IDs / SUM ALL96 case means
of three profile0 timed repetitions, including encoder, cross-KV, complete
ownstate decode/greedy/stop and every pack/copy/launch/sync/reconstruct/scale.
Four digit columns are ONE request, not request batch4. Diagnostic whole-state
file writes happen after the source timer, identical to original scope; actual
process wall/bytes are additionally retained. Bootstrap10000 draws/seed485485,
resampling24 books with4cases together; one-sided5%/95% quantiles. Strong
>=50 qualification requires BOTH ordinary AND prose lower95>=50 on BOTH actual
artifacts plus benefit over fresh matchedCPU.100prose is a reported stretch.
Evaluate speed only after both complete artifacts; no early winner selection.

Profile1 decomposition is admitted only if its aggregate/time ratio lies
[.90,1.10] and ALL24 book ratios [.85,1.15], separately for each backend/source.
Failure excludes that decomposition; complete profile0 rates stay reportable.
Do not make a global arithmetic conclusion from timing variability.

Load time includes context/library init,169 uploads and staging. Explicit
first-request rate charges load+untimed-warmup request (repetition-1); report
load-amortized request conditions1/10/100 separately. Warmup compute/actual
process wall is retained; no free initialization claim. A process-cold request
does not establish cold OS page cache or physical streamed DRAM bandwidth.
Code copies/scales/controls buffers and transfers stay visible.

## Wire, resources, failure and sole execution

Controls: magicM485CTL1(8),U32count11; each7U32 M,D,Q,Mpad,Dpad,pattern,reserved0;
W[M,D]I8,row scales[M]F32,x[Q,D]F32,codes[Q,D]I16,query scales[Q]F32,
partials[Q,4,Mpad]I32,reconstructed[Q,M]I64,y[Q,M]F32,scalar oracle[Q,M]F32;
all little endian/exactEOF. Write inputs before CUDA and partials before assertion.
Whole output reuses SWR32O01 exact geometry/layout; no changed acceptance wire.

Ceiling main45min, parent1GiB, host conservative parent+largest native peak24GiB,
eachcommand180s/output256MiB, alloutputs12GiB. Binding refines prospective
output bound from ALL archived file lengths BEFORE observations. Own device
weights+IO+workspace<=2GiB; actual complete-model allocation178552832B.
CUDA context/library memory is NOT a measured WDDM process peak. Global
cudaMemGetInfo before/after setup retained with explicit limitation.
Independent audit15min/512MiB, all output/reference/payload/runtime bytes.

Stop immediately at first mandatory runtime/arithmetic/source gate failure;
retain exception/API return/status/stdout/stderr, raw input/partial bytes and
actual child/controller PID/create time and Windows Application1000 records.
Performance gate failure closes THIS fixed layout, not all GPU/fusion methods.
No retry, altered enum/layout/kernel or overwritten output namespace.
Guard may stop only its own live child and retains cause/partials.
Record actual first main/audit handles, actual tool exit codes and digests in
separate registrations, then Windows terminal, independent audit, auditWindows
and frozen metadata finalizer. No main/control/audit replay.
While science runs: reobserve SAME returned handle; light PS reads and concise
commentary only. No concurrent edits/hashes/Git/other benchmark.
