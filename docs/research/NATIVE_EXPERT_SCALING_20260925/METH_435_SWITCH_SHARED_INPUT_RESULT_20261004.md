# M435: shared nonlinear input transport remains ineligible

Frozen c6382f8 before import/numeric/fit. Completed once, terminal exit0.
[Protocol](METH_435_SWITCH_SHARED_INPUT_PROTOCOL_20261004.md), math/controller
`benchmarks/native_expert_scaling/meth435_switch_shared_input_{math,transport}.py`.
[Raw](meth435_switch_shared_input_result.json) SHA256
967676c9c6697bc75e5bd49391f1a6af2f745461afed75b2ce21ec3580bf0f51.

## Apparatus and fixed fit

ALL5 apparatus gates PASS. Fresh1936 captures/384 forward archives and original
374/389 binaries exact; paired source/decoder/key/input identities exact434.
434 complete source hashes inherited explicitly, payloads not reread. Natural
cohorts not used for training or eligibility. Development-only population
normalization; frozen source functions/core/head untouched. Tiny D7/H5 all89
coordinates plus actual D768/H128 initialization five directions qualify
independent native endpoints, fixed-offset F64 reference gradients and FD.
No ReLU crossings; negative controls pass. Native/reference max4.6602e-8,
F64 FD max1.0135e-10. Real initial output mean byte-exact; intended zero first-
layer/input gradients and live second-layer gradients verified before updates.
This is a local approximate derivative, never a derivative through rounding.

ONE shared768->128->768 ReLU map, seed435,197504learned+3072fixed coefficients;
802304B stored/nominal coefficient accesses,3172352B learned values/gradients/
two Adam moments plus buffers. Cost constant against expert n, no physicalDRAM
measurement.16passes/512Adam updates/16128sample evaluations, batch32(last16),
LR.001/clip1, all update orders/losses/norms retained. Final checkpoint only;
no validation checkpoint/seed/width/depth/rate/pass selection or tuning.

## Input feasibility, SAME404/406 requirements

| Metric | Development | Consumed validation | Eligibility limit |
| --- | ---: | ---: | ---: |
| Total squared error / development-mean reference |.3092701321|.6091305275|<=.50 on validation |
| Median row relative L2 |.5977772788|.8767260897|<=.25 |
| 95th row relative L2 |.7259788841|.9910153734|<=.50 |
| Median cosine |.8029352556|.5110616040|Descriptive only |

Validation squared error21760935.72217 versus mean-only35724585.68633.
Three scientific input gates FAIL; numerical finite native/F64 smooth gate
PASS. All six validation book median errors.82965.. .92931; no isolated book
or selected-ID rescue. Input error falls versus the mean but remains too large.
The development/validation gap is an observation, not a proof of universal
nonlinear nonrecoverability. One bank/consumed split does not establish fresh
generalization, transferred predictions or whole-model quality.

## Resources, retained evidence and decision

28.843s including bindings, peak557035520B, freshhashed2801220192B,
five archives11846750B. CPU0/Torch1/BLAS1; noGPU/T4/network/new corpus or native
engine change. Tiny/real full qualification NPZs, checkpoint, full1008development
predictions and336validation native/F64 predictions retained with hashes in raw.
All512 updates complete, no apparatus/resource failure or repair.

**Close THIS fixed shared input mapper before width/depth/optimizer sweeps.**
No coupled-function fit/export licensed from its regression. The next useful
uncertainty is whether a perfect input interface alone could improve frozen
function utility: NEW436 prospective oracle substitutes captured source128
normalized input at the added WI entry, retains frozen431 output factors and
original256 residual/core/head. Exact source128 feature replay qualifies the
teacher-ID route. Actual classifier and teacher routes are diagnostic only;
preservation/benefit/permutation/removal remain required. This oracle is not a
deployable map or evidence of impossible future readouts. Freeze new controller/
protocol before outcomes; no new optimizer.431/426 recipes remain closed.

Useful RAM-scale n/CPU LUT/routing/realDRAM, another family/~100B and original-
relative whole quality/SAMEartifact>=50 acceptedbatch1tokens/s remain OPEN.
