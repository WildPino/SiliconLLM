# M419 protocol: integer-native forward and explicitly approximate backward

Freeze both NEW419 Python sources/protocol before FIRST forward/gradient/FD
outcome. Native417/418 sources/engine/qualified originals remain immutable.
418 ALL9 apparatus gates PASS, raw SHA
4ebb37ed40d788eb85168d028b3e0b538cd7d65e009293cd32d5fb0eb117d829.
417 first schema failure retained f2a57bd. Current HEAD17c0b87 before freeze.
No fitting/optimizer/update/native model changes in this stage.

## Question and next decision

Can an independently controlled final-bank differentiable prototype preserve
native forward values EXACTLY and give a usable, explicitly approximate backward
for small function-specific corrections? Failure stops THIS numerical contract,
retained before any repair; PASS licenses only a separately frozen fit/selector/
objective/resource/causal-function-benefit protocol. Useful added capacity, new
excluded whole quality and SAMEartifact accepted50 remain required later.
This replaces failed state-vector matching with prediction-aware functions;
it does not resume lexical/state-map or closed donor-cost tuning.

## Fixed data, source bindings and exact-forward scope

ALL384 retained418 observations:192 shared405 teacher cases and192 original393
natural cases,4895 final-decoder block11 states. Fresh hash ALL1936 prior files,
all prior committed helpers, full22.36GB source target payloads/manifests and
all manifest lookups. Tokenizer identity inherited through immutable fresh418
raw/provenance/entire capture inventory. Independent source exports338/380 bound.
No new corpus/reference/model invocation; source128 native teacher remains the
qualified quantized approximation. Natural data qualification-only, excluded
from future paired fitting; teacher only18dev/6val books/all4 cases.

Prototype computes Switch FF norm from preFF, ALL selected3072 WI rows, ReLU,
ALL selected768 WO rows, selected probability over full captured original router
scores, F32 residual, finalnorm/head scaling, ALL32128 head logits per position.
All observed endpoint arrays/probability and all three A16 scales/codes byte-exact
against418 and its old native outputs. Full-state/head archives retained per case.
Source SIMD router scores are inherited from qualified exact capture, not newly
recomputed; no new router-score implementation parity claim. Hard argmax fixed
from original captured selected ID, tie lower ID explicitly controlled.

I8 forward: F32 absmax/32767/F32 divide/nearest-even+clamp A16; original signed
I8 weights/row F32 scales, exact I64 sums then two F64 products->F32. No -128
weights or -32768 activations. RMS forward: F32 squares, F64 sum, F32 mean,
epsilon1e-6/F32 sqrt/inverse and two F32 multiplies (Switch, not Granite).
ReLU preserves native negative-zero behavior (only x<0 replaced by +0).
Full-score probability F32 differences/F64 exp->F32/F64 sum->F32, residual two
separate F32 ops, final/sqrt(D) F32. Torch eager CPU operators, no fusion/compiler.

## Explicit surrogate and single correction parameterization

Native-values custom autograd functions carry APPROXIMATE backward:
I8/A16 uses fixed F64 dequantized W transpose and identity STE for activation
quantization, ignoring quantizer/rounding derivatives. RMS backward is smooth
F64 square/epsilon formula including its mean-coupling term, evaluated at saved
F32 primal input; dtype conversion/rounding ignored. ReLU derivative positive1,
negative0/zero0; no differentiability claim at kink. Full-score probability
backward uses F64 softmax selected-row Jacobian on saved F32 scores; argmax held
fixed, no derivative through discrete selection. F32 adapter forward accumulates
dot in F64->F32, smooth F64 outer-product/transpose backward->F32.
Native rounding/top1 do NOT have the derivative this STE assigns.

ONE prospective rank8 input correction x'=x+A(Bx), output correction
y'=y+C(Dy), frozen original WI/WO and source core. Four per-function factors
A,C[768,8], B,D[8,768], F32 parameters. For two real controls, fixed seed419+n,
A/C exactly zero, B/D Gaussian std1/sqrt(768), so initial output must remain
byte-exact while gradients A/C are live; gradients B/D must be exactly zero.
No all-zero-four-factor training initialization or rank/grid/seed sweep.
Selector/objective/full-fit budgets NOT yet fixed here, no uncharged inference
claim or permission to fit. Rank8 is a bounded numerical candidate, not an
effective capacity result or promise to alter old256 functions.

## Independent gradient/FD gates fixed before observations

Tiny seed419: D7/FF11/V13/router5/rank2, explicit deterministic positive WI/pre
to avoid ReLU boundary, nonzero small factors. F64 smooth twin uses dequantized
weights and no activation rounding. Independent literal NumPy F64 reference
and central FD epsilon1e-4 over ALL pre/scores/A/B/C/D coordinates. RelativeL2
smooth autograd vs FD<=1e-5; native saved-primal STE vs smooth FD<=1e-3. Tiny
smooth logits vs independent F64 reference<=1e-12. Denominator max(norm,1e-12).
Individual RMS/full probability FD<=1e-5; ReLU signed-zero/value/gradient and
equal-score lower-ID selection controlled. Never FD native rounding to infer
a true quantized gradient, or silently replace exact forward by dequantization.

Two real fixed controls: ONLY first position teacher.book0.case0 on each source,
own original selected function, full32128 logits and rank8 zero corrections.
Prediction cross-entropy target from opposite source's SAME shared-context
captured native logits, T1, detached; diagnostic only, no updates/benefit scoring.
ALL baseline states/head logits exact; A/C gradient norm>1e-10, B/D zero.
Native STE vs F64 smooth twin relativeL2<=1e-3 for A/B/C/D/pre/full score vector;
native vs smooth full-logit relativeL2<=1e-3, separately labelled approximation.
For each factor ONE predeclared random Rademacher unit direction seed419+n,
central NumPy F64 smooth directional FD epsilon1e-4 vs autograd projection,
relative error<=1e-5 with denominator max(abs(projection),1e-8). No coordinate
selection by observed gradient or post-outcome numerical threshold relaxation.
All gradients/losses/arrays finite, complete factors/gradient/logits retained.

Five negative controls: wrong native head coefficient bit at maximal nonzero
head activation column must change original logit; missing selected probability
must change whole head; missing RMS mean-coupling term and missing probability
tail derivatives must disagree with FD>1e-3; wrong tie-ID must reject. Original
ten418 malformed-capture controls reused through freshly bound immutable evidence.

## Resources and accounting before work

CPU ONLY local .venv Python/Torch2.6.0+cu124/NumPy2.4.6, Torch intra/inter-op1,
all BLAS1, main logical0. No native timing, model job overlap, GPU/T4/network.
Only ancestors and two exact pre-existing tiktok_publisher pythonw daemons allowed.
Maximum20min, process RSS/Windows peak working set<=4GiB, output<=2GiB. Errors
retain first failure; no retry or mutation of419. No compiler/new binary required.

Full head I8+scales24802816B; I64 forward workspace197394432B per source;
F64 dequant backward197394432B per source, only used for two diagnostic gradients.
Head row work ALL4895*32128*768 plus selected WI/WO; one selected expert cache
per source, no dense128-function optimizer. Complete arrays about0.9GB retained.
Prototype RAM includes enlarged integer/gradient workspaces; these are training
overhead and cannot be counted as compact native inference storage.

Prospective added128 in ONE bank:603979776 original coefficients,
605945856B I8+F32 scales. ALL128 rank8 corrections3145728 trainable coefficients,
12582912B F32 values, same gradients,25165824B two Adam moments (total50331648B
before optimizer bookkeeping). One selected correction addresses98304B F32
factor coefficients. Full32128 vocabulary head and future selector costs must
be charged. No full12-bank optimizer proposal or inherited top1-speed claim.

Record all source/helper/payload/prior/runtime hashes, FD and approximate
gradient errors, full baseline/archive inventories, process/resource costs,
limits/first failure and decision. If eligible, freeze a NEW bounded real-function
fit with no-added baseline, hard selector active budget and causal pretrained
function ablations BEFORE training or fresh excluded whole-quality documents.

After freeze:
`.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth419_switch_function_gradient_contract.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth419_switch_function_gradient_result.json`

Goal ACTIVE/INCOMPLETE. No useful384/10x/RAM-scale n/CPU LUT/physicalDRAM,
other-family quality or~100B conclusion from this numeric prerequisite.
