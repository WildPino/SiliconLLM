# M435 protocol: one shared nonlinear final-bank input mapper

Freeze math/controller/THIS protocol before FIRST compile/import/numerical
observation.434 rules out sole selector rescue for frozen431 functions;433
shows the output objective is feasible.403/404/406/414/416 maps linear or
orthogonal;425..431 per-ID rank8 learned through prediction loss. NEW variable:
one nonlinear input transport shared across all expert IDs, receiving ALL1008
development positions rather than sparse private-ID exposure. Does this small
shared geometry meet the previous input eligibility bounds? Resolve BEFORE
more selector/readout/token-loss fitting. No old checkpoint tuning or rescue.

## Provenance and data

Committed418/420/434 raw records and exact recorded helper identities. Fresh
ALL1936 capture inventory and384 native-forward archives, original374/389
binaries.434 complete source hashes inherited explicitly, not reread: only
captured input states used, source weights unchanged. Parent434 apparatus
must pass and both potential gate sets fail.405/418 paired book/case/source/
decoder IDs/split/input byte identities exact; keys exact434. Natural cohorts
qualification-only and never used for supervision/eligibility. One final
decoder sparse bank11 normalized FF input256->input128.18books1008development,
6books336validation, all4cases/14positions. This validation is consumed, no
fresh whole-model quality claim. No core/norm/head/original-function change.

## Exact fixed construction and accounting

Development-only F64 population means/std per coordinate, roundedF32;
std floor1e-6 BEFORE F32 conversion. Four768F32 fixed vectors mx/sx/my/sy.

    z=(input256-mx)/sx
    raw=NativeFloat(z,W1)+b1
    hidden=NativeReLU(raw)
    unit=NativeFloat(hidden,W2)+b2
    prediction=my+sy*unit

W1[128,768], W2[768,128], b1[128], b2[768]. NativeFloat: immutable422
F64 dot of F32 coefficients/input roundedF32. Other elementwise operations
F32; ReLU preserves negative zero, positive derivative1/nonpositive0.
197504learned coefficients790016B, fixed3072coefficients12288B; total802304B.
Values/gradients/two Adam moments3160064B +buffers3172352B. Torch step scalar,
allocator/workspace overhead separately bounded by measured peak. Coefficient
access802304B is nominal, not actual DRAM. Same mapper all IDs, constant versus
n; no measured native speed/coupled cost inferred. ONE seed435: W1 normal/
sqrt768 F32, b1/W2/b2 zero. Initially prediction byte-exact development mean,
W2/b2 derivatives live and W1/b1/input zero. No per-ID parameters/routing fit.

## Composed derivative qualification before ANY updates

Tiny D7/H5 seed435 nonzero parameter fixture; x/target fixed normal seed435,
normal means*.1 and std linspaces[.7,1.3]/[.8,1.4]. Both ReLU signs and full
five-field W1/b1/W2/b2/input derivatives live. ALL coordinates finite difference
h1e-4. Real first development input D768/H128 zero-output initialization,
five fixed unit Rademacher directions seed435+D, h1e-4, projection floor1e-8.
Independent literal NumPy native nodes byte-exact custom autograd nodes.
Capture constant offsets at each F32 native parent; ordinary F64 Torch
continuation and separate literal NumPy continuation endpoints relative<=1e-12
ANDabsolute<=1e-9. All native/reference gradients relative<=1e-3; F64
reference FD<=1e-5/native FD<=1e-3. Fixed residual loss-difference expression
mean((2*(anchor-target)*delta+delta^2)/sy^2), avoids subtracting large losses.
No ReLU crossings, offsets remain byte-fixed. Tiny omitted ReLU must differ
>1e-5; dropping final offset must violate endpoint limits. Fixed statistics
not differentiated. Full nodes/offsets/parameters/input/target/gradients/FDs/
directions saved. This qualifies local approximate derivative, not rounding.

## ONE fit and final evaluation

16fixed passes; shuffle development indices default_rng435+epoch. Batch32,
last16 =>32updates/pass, exactly512Adam updates/16128sample evaluations.
Each native-valued row forward/backward accumulated with batch averaging;
loss mean768 coordinate squared normalized error using F64 prediction/target/
output std. AdamLR.001, betas.9/.999, eps1e-8, weight_decay0, foreachFalse;
joint gradient norm clipped1.0 with error_if_nonfinite. No validation selection,
early quality stop, seed/lr/width/depth/pass/batch sweep. Final checkpoint only.
Keep every update order/loss/unclipped norm, dev and validation input metrics,
full dev predictions and val native/F64 smooth predictions, final parameters
and buffers. Do not duplicate paired input corpus; retained parents+hashes
and keys reconstruct inputs. No token-loss/gate/readout/function supervision.

Input eligibility SAME404/406: validation total squared error<=.5times
development-mean baseline, median row relativeL2<=.25,95th<=.50; finite and
global native/F64 smooth predictions relativeL2<=1e-5. Per-book metrics/cosine/
norms/errors retained, no selective IDs. If ALL pass, ONLY new coupled function
output/readout/prediction controls licensed, no capacity or model export. If
FAIL close THIS fixed shared mapper before width/depth/optimizer tuning.
All9 capacity demands and original-relative whole quality/SAMEartifact>=50
acceptedbatch1tokens/s/useful RAM-scale n/LUT/realDRAM remain required.

## Resource/first-failure contract

<=720s including bindings/2GiB RSS or Windows peak/16MiB outputs, diskreserve
2GiB. CPU0/Torch1/BLAS1/Torch2.6.0+cu124/NumPy2.4.6; noGPU/T4/network/new
corpus/C engine modification. No overlapping modeljob; preserve exact publisher
daemons. Stop FIRST binding/qualification/nonfinite/resource failure, retain
before NEW numbered repair. Scientific source/protocol immutable after outcome.

After freeze:
`.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth435_switch_shared_input_transport.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth435_switch_shared_input_result.json`
