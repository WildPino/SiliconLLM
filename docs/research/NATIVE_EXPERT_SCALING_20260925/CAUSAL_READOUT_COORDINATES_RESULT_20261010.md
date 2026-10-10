# Original C causal coordinates: complete recovery/audits, approximate admission

10 October2026. COMPLETE. 8808 actual27 normalized causal coordinates recovered
from stored native logits; APPROXIMATE numerical admission after exact dot-path
bound refinement. No native/source/history/GPU/optimizer call and no new model
quality observation. Goal incomplete.
[Recovery protocol](CAUSAL_READOUT_COORDINATES_PROTOCOL_20261010.md),
[dot-path refinement](CAUSAL_COORDINATE_DOT_BOUND_PROTOCOL_20261010.md).

## Authoritative evidence

- [Original recovery](causal_readout_coordinates_result_20261010.json) and
  [complete full audit](causal_readout_coordinates_stored_adjudication_20261010.json):
  first NOT_QUALIFIED record retained; only absolute feature-error upper fails.
- [Bound refinement](causal_coordinate_dot_bound_result_20261010.json) and
  [Decimal50 complete audit](causal_coordinate_dot_bound_stored_adjudication_20261010.json):
  CAUSAL_COORDINATES_QUALIFIED_APPROXIMATE, every numeric flag PASS.

Actual27 chosen because it already has all24 FIT native histories, not from a new
quality ranking. Use48 existing FIT/DEV cases/8808 output-label positions, with
three adopted offsets checked exactly. DEV teacher probabilities not consumed;
no quality-based DEV selection. These are normalized states from teacher-forced
C histories, not raw hidden state or free-running chatbot success.

## Measurements and algebra

| Measurement | Value |
|---|---:|
| Old head numerical rank |256/256|
| Singular extrema |20.9698944261 / .912987455226|
| Head condition |22.9684365388|
| QR relative error |1.24916771013e-15|
| Q orthogonality Frobenius error |3.45276838491e-14|
| SVD relative error |5.17335864377e-15|
| Singular-value lower including residuals |.912987455225670|
| Max native-logit reconstruction absolute residual |1.53139606454e-5|
| Recovered state norm range |253.117577288..271.046755674|
| Conservative native norm upper |342.576506687|

Old final norm has a large coordinate gain: the bound uses16 max|gamma|, not a
unit-gamma norm assumption. Inverting this head is sufficiently conditioned;
that does not mean the causal state contains all donor information.

For every reconstructed fhat:
error_L2 <= (||H fhat-z_native||_2 + native_dot_error_L2_upper)/sigma_H_lower.
The original gamma256 bound gives max absolute.24418192398, failing frozen.05;
relative.00096485215 still passes1%. Full recovery/audit completed before any
bound correction. The original gamma256 derivation remains valid but loose.

The unchanged exact dotf256 body has32 FMA steps per lane and7 sequential scalar
summations, no scalar tail. Longest rounding chain39. Using gamma39=2.32458654993e-6,
with the SAME norm upper/factors/states/residuals/thresholds, dot error L2 upper
is.0339222539410, returned max feature bound.0374432890195, relative bound
.000147186806128 = .0147187%. Both frozen.05/1% gates PASS. Neither an altered
threshold nor an inversion replay occurred. Old NOT_QUALIFIED data is preserved.

Bounds are conservative algebra under the stated unexceptional F32/FMA model,
evaluated F64. They are not directed-rounding interval proofs or bitwise state
identity. Future head fitting must propagate this uncertainty; final new native
C histories, not reconstructed feature predictions, determine artifact quality.

## Full audits, counts and custody

Recovery freeze30f36689b4a03b0c25117d30493b707424abcc85,
bindingf8fcb9d9eb9ab2940cc5ef0fbfc3b3675712f5eeb10f70d8346ab6de31748db0:
70 unique input extents/6,465,252,109B, not unrelated87GB historical collection.
One QR/SVD,8808 state solves/full-head reconstruction rows; 102 output files
155,156,696B plus1,557,332B result within256MiB.48 feature files18,038,784B.
Features in results/native_expert_scaling/causal_readout_coordinates_20261010;
exact paths/extents/shapes in bound-refined result.

Recovery session68256 CLOSED/exit0; launcher19344 created14:51:39.282998+02:00,
worker13052 created14:51:44.768559+02:00; gone before audit. Held47.547s,
prewrite35.719s/print35.891s. OS709,226,496+32,559,104=741,785,600B<=3GiB.
Audit session88609 CLOSED/exit0; launcher18296 created14:53:16.451498+02:00,
worker8708 created14:53:21.996372+02:00; all gone. Held60.484s, prewrite48.047s/
print48.860s. OS578,035,712+30,908,416=608,944,128B<=3GiB.
All70 inputs/102 output hashes, full QR/SVD/orthogonality without refactorization,
all8808 normal equations/residuals/bounds/maxima,384 math.fsum head witnesses,
original four readout bodies, decisions/call/resource gates PASS.
Max normal-equation delta2.50111042988e-12; scalar-head arithmetic3.55271367880e-15;
metric comparison2.14583906200e-12. Audit8808 stored head rows,zero QR/SVD/replay.
Result e95ae8a8831ae3800ca1049c8e7fb26f95919b34311d4a53df082e85270ef36c;
audit0fa2456815bf93ffb5c1c03a4c191087c9553e3fa181e919f95bbdb1bdf02bfc.

Refinement freezea5856e7c23b987f2fcd876f1f061eab18ae8873d,
binding66a62c7102709d142b0b71045d3b25994b11afc2523d6ed4b2e7d556dbf5ef0d,
17 inputs/9,273,637B of scalar evidence/code; no feature/old full-score binary read.
Held.844s +.515s audit, worker prewrite.438s/audit.140s. OS35,659,776+31,068,160
=66,727,936B; audit32,092,160+29,470,720=61,562,880B<=512MiB.
Short processes terminal directly through receipts: capture launcher27064/worker
19120 created Unix1791637183.0938604, audit11564/26732 created1791637184.7654169.
All gone. Full exact39-path proof, input/output hashes, all8808 Decimal50 bounds
PASS; max comparison6.93889390391e-18. Zero new states/QR/SVD/head/native/source/
optimizer/GPU calls. Parent complete heavy audit adopted explicitly.
Result a48f33db4f850e3def589618ff1020519a0e595d85bcae22ba0844e4d599c3a8;
audit f3f2db3f27e8f13564b7496ae5e3586c12a53a6e734f815578e1e107137c0f3e.
Total held phases109.390s; initial binding preseal is separate preparation.

## Exact continuation

[Shared categorical native readout](CAUSAL_CATEGORICAL_READOUT_NEXT_20261010.md)
is the next implementation. Seal every consumed feature binary/teacher/phi,
FIT-only covariance/whitening/initialization and one bounded convex head fit;
then offline fuse the native head and evaluate the ACTUAL original C artifact.
No extra runtime operator/donor read or unchanged failed history dose. A shared
map may recover readout information, or expose missing current causal features;
finite optimizer failure is not an information theorem. Training-side inversion
has no inference-time cost. Original SSM/SWA/ternary/LUT functions remain the core.

No new useful-chatbot, accepted50 tokens/s, useful RAM-driven n, physical DRAM,
CPU IDs/normalized mass or family/scale admission. Old quality failures and GPU/
native arithmetic gap retained. Approximating a usable feature cohort is a
converter input qualification, not completion of the conversion pipeline.
