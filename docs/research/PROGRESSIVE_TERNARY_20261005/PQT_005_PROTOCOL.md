# PQT-005: original pretrained expert-function conversion

Prospective, 5 October 2026. No PQT-005 original-function reference, fitting or
evaluation has occurred. PQT-004 showed a coverage benefit without preservation;
this experiment tests actual original expert wi/wo values and prepares the
same artifacts for a separately coordinated CPU-native cost test.

## Exact question and source

Can behavior-aware ternarization preserve an entire original pretrained expert
FFN, including its ReLU, at a counted storage budget? Original Switch decoder
block 11 experts 8 and 105, immutable revision and F32 tensor hashes admitted
by PQT-SRC-001. Source f(x)=Wo*ReLU(Wi*x), Wi [3072,768], Wo [768,3072], no
bias; original dropout disabled in inference. Config ReLU/shapes checked and
hashed in the immutable input manifest. Never use parent's I8 function outputs
as the original float reference. Compute original-weight references anew.

Input dataset is private acct2 `giggio253/pqt-005-switch-original-20261005`.
Package admitted weights and all four PQT-SRC-002 input arrays, preserving
byte hashes and provenance, plus original config and input manifest. Creation
is authorized within private research operations, fresh namespace only, no
update/version overwrite or public release. Verify live owner/quota/owned jobs,
record creation intent before the network mutation; on unknown outcome inspect
the same ref rather than create again. Exact server metadata must confirm
private ownership, successful processing and version 1 before kernel dispatch.
Runtime rechecks every mounted file against the committed input manifest;
recursively locate it rather than assume a mount path.

## Data roles and fresh function probes

Use raw pre-input-quantization F32 inputs from the inherited I8 core. Expert 8:
87 role-0 calibration rows, 41 role-1 rows; expert 105: 122 calibration rows,
54 role-1 rows. All inherited rows, including role-1, are **consumed development
observations**, not fresh natural validation or original float-model contexts.
Keep original row indices/mode/book/role/acceptance/probability metadata. Fit
only role-0, no weighting/selection by old errors or router probability.
Report role-1 raw-input function fidelity as a development transfer measure.
Report effective quantized role-1 input results separately; no fitting on them.

After all final expert/arm representations are frozen, generate independent
synthetic robustness probes per expert using CPU torch.Generator seed
20261006+expert, pinned runtime. From role-0 raw inputs compute FP64 coordinate
mean and population standard deviation; global std=sqrt(mean coordinate
variance). Gaussian: 64 samples mean + randn*max(coordinate_std,.1*global_std),
cast F32. Perturbed: first 64 calibration inputs in original order +
.05*global_std*the next 64x768 independent random normals, cast F32. The CPU
random normals use FP64. Also one all-zero probe. Retain all sample bytes,
means/stds/seeds for independent reproduction. These are fresh function probes
conditional on calibration statistics, not natural-language capacity evidence.

## Fixed arms and fitting

Six arms for each expert, all reported: D, DR, PR-B, FF, I4, I8.
D: original-weight group-64 L2-optimal scales and direct ternary codes for both
matrices. DR: D then two coordinate passes for Wi on original calibration
preactivations; two Wo passes on resulting ReLU activations targeting original
full-function outputs. No bias.
PR-B: progressive curvature compensation plus two Wi coordinate passes;
calibration mean preactivation offset (3072 F32 values), then ReLU. Original Wo
progressive compensation on this candidate hidden input plus two coordinate
passes targeting original function outputs; mean output offset (768 F32).
Original matrix scales remain fixed during these discrete fits. Count offsets.

FF: start from PR-B, optimize the full function's normalized calibration MSE
with exact hard ternary forward weights. Original source weights never receive
gradients. Train latent codes, log group scales and both offsets, six parameter
tensors. Same straight-through identity derivative, Adam latent lr .02,
log scales/bias lr .001, beta(.9,.999), eps1e-8, no weight decay/AMP, gradient
norm clip 1.0, latent clamp [-1.5,1.5], scale factors [.25,4] around initial
max(scale,1e-8). Loss denominator is original full-calibration output mean
squared energy. 256 actually applied updates per expert; each batch is first
32 indices of a fresh CPU randperm(calibration_count), generator seed
20261005+expert. Save every batch index/update/loss; verify each Adam state=256
and all values finite. Only final checkpoint is evaluated.

I4: PQT-002 original-weight direct symmetric -7..+7 group-64 baseline.
I8: original-weight row scale=max(abs(row))/127, nearest-even codes clipped
[-127,127], signed int8; zero row scale gives zero codes. Neither fits inputs
or uses offsets. These are simple precision controls, not optimal PTQ/QAT.
All final arms for both experts frozen before role-1/probe evaluation. Report
final calibration error as fitted-set evidence, not as a promotion test.

## Metrics and frozen decisions

Report full-function normalized SSE and relative RMS for every input condition,
per-row squared error/reference energy and cosine, source/candidate outputs,
hidden ReLU agreement, expert-specific and pooled aggregates. Cosine is
diagnostic; account for zero-energy rows explicitly rather than hide them.
Function preservation gate per expert: relative RMS <=.01 on each of consumed
role-1 raw inputs, fresh Gaussian and fresh perturbed groups; zero-probe output
norm / calibration source output RMS norm <=1e-4. This is an expert-function
gate, not token/generation/semantic preservation. Both experts must pass.
Capacity: payload <=35% of the pair's 9,437,184 FP16 bytes. Include scales,
offsets and actual deployed-file header separately. Total fitting <=2700s;
FF each <=600s. Report quality/storage comparisons with I4/I8 and direct D
without promoting a failed arm based only on smaller size or calibration fit.

## Representation and independent verification

Ternary two-bit code mapping/ordering and group scales remain PQT-001's format.
I4 codes retain the reserved -8 rejection; I8 -128 reserved. Export each
matrix's packed codes, positive F32 scales and optional biases, plus a native
pair file with a 64-byte little-endian header. Header `<8s14I`, magic
`PQTFFN1\0`; integers: version=1, format (0 original F32, 1 ternary, 2 I4,
3 row-I8), D=768, F=3072, group (64, or 0 for F32/I8), bias flags (Wi bit0,
Wo bit1), six segment lengths (Wi codes/weights, scales, bias; Wo likewise),
total file bytes, reserved=0. Body in that order, C-order raw bytes. No
continuous shadow/optimizer values are deployed. Original F32 control also
exported in the same pair envelope, without bias/scale segments. File and
matrix identities remain linked to quality results.

Before source-function access, existing independent math/scale/STE controls
plus scalar two-layer ReLU goldens, zero and positive-homogeneity controls.
Independent cuda:1 process imports no fitting module. Rehash inputs/artifacts,
independently decode every packed matrix and native pair file, check exact
agreement and accounting, regenerate all probe/schedule bytes, and compute
all original/candidate full-function outputs in FP64. Main F32 versus audit
F64 relative RMS <=1e-5; recompute every error/energy aggregate and frozen gate.
Failures/nonfinite values prohibit adjudication; preserve first evidence and
number repairs, never silently relax tolerances. Native timing is not part of
this GPU test. The subsequent CPU implementation must execute these same
files directly and pass scalar/F64 correctness before any timing claim.

## Resources and retained scope

acct2, fresh private `giggio253/pqt-005-original-experts-20261005-001`, qualified
pinned T4x2 image and numerical runtime, deterministic seed, TF32 disabled.
cuda:0 fit/evaluate, cuda:1 separate audit. Install <=900s, experiment <=3300s,
audit <=900s, server <=5400s, total fit <=2700s. No local GPU/native timing,
main environment mutation or process interruption. Commit protocol/source,
prepare hash-bound private input dataset, verify server readiness/privacy,
generate committed-byte bundle and commit it before live job admission/push.
All data/code/logs/first failures retained under PQT namespaces; retrieve once
on terminal state, large arrays outside Git, small raw evidence/hash manifests
committed, update TERNARY_INDEX.md. Earlier goal turn made empirical progress
(PQT-004 plus activation admission); the full research goal remains incomplete.
No expert-function result can substitute for complete-model predictive and
generative quality or useful coordinated native cost.
