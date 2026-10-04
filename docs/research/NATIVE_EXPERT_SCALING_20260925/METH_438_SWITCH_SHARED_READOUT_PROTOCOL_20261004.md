# M438: one shared output readout versus matched own-input control

Freeze NEW controller/THIS protocol before FIRST compile/import/forward/
gradient/fit.435 shared nonlinear input regression fails3/4 criteria;436 correct
native function inputs with frozen431 C/D remain insufficient. New hypothesis:
shared output supervision over ALL1008dev positions can expose source-function
utility better than sparse private-ID output factors. This is an oracle
feasibility screen BEFORE available compact inputs or selector fitting.

## Fixed data, sources and matched information

Exact committed418/420/431/433/434/435/436 records/helpers. Fresh unique parent
output inventories,384 native forward archives, full128/256 source payloads/
manifests/descriptors and source stat identities, original374/389 binaries and
actualTorchCPU DLL/extension hashes exact431. Same405/418 paired1344 keys/native
teacher probability streams exact431.18books1008dev/6books336val/all4cases,
final decoder sparse bank11 only. Validation consumed, no fresh generalization.
Natural cohorts qualification-only, never fit/utility supervision.

Real arm feature=captured original source128 WI/ReLU/WO output at its teacherID.
Matched arm feature=captured source128 normalized input before that SAME function.
Both see SAME unavailable source128 core/context information and IDs. Readout
parameters do not see IDs; records use them for consultation/useful-ID counts.
Control asks whether pretrained function improves upon its own input information.
Source functions/core/head untouched, original256 base/probability/finalnorm/head
retained. Do not reuse fitted435/431 values or fit a classifier/gate/input bridge.

## Shared construction, fixed precision and costs

Each arm mean/std of its dev feature coordinates: F64 population, std floor1e-6,
then F32 buffers. One C[768,32]/D[32,768], fresh SAME seed437 each arm. D normal/
sqrt768 F32, Czero; no learned bias/private-ID factors. For one position:

    normalized=(feature-mean)/std
    low=NativeFloat(normalized,D)
    correction=NativeFloat(low,C)
    probability=original256 native selected probability
    post=original256_post+probability*correction
    final=original256 native RMS(post)
    logits=original256 native I8/A16 head(final*F32(1/sqrt768))

Immutable422 NativeFloat F64dot->F32, other elementwise F32, original selected
probability and RMS/head precision retained.49152learned F32coeff196608B plus
1536fixed coeff6144B =>202752B stored/nominal coefficient accesses per arm.
Values/gradients/two Adam moments+buffers792576B per arm, overhead measured by
RSS/peak. Shared cost constant versus n; no actual DRAM/native latency claim.
Original PLUS extra final-bank function when ON, not old single-function rate.

## Independent composed qualification BEFORE any updates

NEW helper includes normalization, both factors, full coupled probability,
weighted residual, full coupled RMS and complete head. Frozen source feature
treated as independent boundary input, no gradient through captured WI/WO.
Tiny D7/rank2/V13/N5, SAME fixed seed437. Nonzero C/D, random feature, mean*.1/
stdlinspace.7..1.3, base.2.. .8, finalnorm.8..1.2, scores-.7.. .9; head integers
uniform-9..9/scales.04.. .07, target normalized random logits. All47 coordinates
of C/D/feature/base/scores independent FD h1e-4, all gradients live.

Two actual first-development D768/rank32 initial-zero contracts, one per arm.
Numeric probe ONLY430 fixed onehot (argmax(original)+16064)%32128 avoids weak
nearly stationary teacher-mixture derivative; NOT training objective. C/base live
norm>1e-10, D/feature/scores gradients exactly0; post/full head exact418/420.
Five fixed unit Rademacher directions seed437+N, h1e-4, projection floor1e-8.

Literal NumPy native ALL9 nodes byte-exact custom forward. Detached constant
offsets at each native parent; ordinary F64 Torch continuation versus separate
NumPy continuation endpoints relative<=1e-12 ANDabsolute<=1e-9. ALLfive fields
native/reference relative<=1e-3; F64reference FD<=1e-5/native FD<=1e-3. Stable
relative-CE loss difference log1p(sum(anchorProbability*expm1(centeredDelta)))
-dot(target,centeredDelta), immutable426 approach. Offsets stay byte-fixed;
negative dropped-head-offset test in all fixtures, omitted coupled probability/
RMS derivative detected in tiny. Fixed statistics not differentiated. Full
native/continuation/offset/parameter/gradient/FD/direction archives retained.

ALL1344 positions per arm initial post/final/head_input byte-exact native418,
two first actual full heads exact. Other full-head identity INHERITED420 through
SAME head operator and exact head inputs, explicitly no new exhaustive full-
head replay. No unfounded all-new full-logit-run claim. Stream hashes retained.

## ONE fixed fit, final-state oracle utility

3passes per arm; same default_rng437+epoch development permutation. Batch32,
last16=>32updates/pass=>96Adam updates each/192total/6048sample evaluations.
Rows forward/backward accumulated with batch averaging, then one update. Loss
equal-mixture CE of native logits against actual p256/p128 at SAME original256
teacher-forced prefixes, F64 stable422 loss, no preservation penalty. AdamLR.001,
betas.9/.999,eps1e-8,weight_decay0,foreachFalse, joint clip1,error_if_nonfinite.
All orders/losses/gradient norms and optimizer-state finite checks retained.
Final state only; no validation selection, early quality stop, seed/rank/lr/
pass/batch/gate sweep. No extra final-development forward cohort; online epoch
loss is not a final-checkpoint training score. Save both readouts+normalizers.

Each final arm forced336 validation full-head matrix. Fixed target-aware mask
SAME431/434: added_mix+.01<=old_mix ANDadded256<=old256+.02. Each arm own SAME
rule, no threshold fit. Real permutation source128(ID+1)%128 with SAME source128
input, SAME shared readout/devstats and SAME REAL mask. Removal restores original
post/head. Three full336x32128F32 matrices plus qualifications/checkpoint retained,
full losses/masks/IDs/book metrics/per-ID gains in raw. No oracle deploy/export.

ALL9 original potential requirements:>=1%mixture AND128 relative gain, mean256
increase<=.02/everybook<=.05,>=34consultations,>=8useful IDs (>=2positions ANDmean
mixture gain>=.01), real beats SAME-budget own-input control by.01, SAME-mask
primitive permutation ANDremoval harm>=.01. ALL must pass before any NEW task-aware
available-input/routing controls. If FAIL close THIS shared rank32/readout recipe
before rank/lr/pass/seed tuning. Oracle input/teacherIDs/targets are unavailable
inference information; even allPASS not deployable capacity/whole quality/rate.

## Resources and first-failure stop

<=720sec INCLUDING bindings/4GiB RSS or Windows peak/320MiB outputs/reserve2GiB.
CPU0/Torch1/BLAS1/Torch2.6.0+cu124/NumPy2.4.6. NoGPU/T4/network/new corpus/native
C edit/benchmark or concurrent modeljob. Preserve exact publisher daemons. Stop
FIRST binding/qualification/nonfinite/resource failure, retain before NEW numbered
repair. Scientific source/protocol immutable after outcome. Original goal useful
RAM-n/LUT/realDRAM/whole quality/SAMEartifact50/another-family/~100B stays active.

## Procedural retry boundary, scientific configuration unchanged

437 concurrency gate stopped before any numeric fixture/qualification/gradient/
update. First failure retained911c42f, raw SHA256
b2ae21e98f04f14dd5e2e764818bff61bc1e44bd59b4c1b30448b15b8d69b0f4.
Win32 follow-up confirmed transient publisher worker25520/29476 absent; exact
original daemons preserved. No process killed or allowlist widened. Bind437
failure/helpers, assert bindings stop and0updates. Controller changes only
experiment/output/protocol identity and retained-failure check; immutable437
math and ALL data/seed437/rank32/objective/updates/qualification/thresholds/
resource/concurrency policy unchanged. Do not restart437 or reuse its output.
First438 failure still stops and must be retained before any numbered repair.

After freeze:
`.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth438_switch_shared_readout_pilot.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth438_switch_shared_readout_result.json`
