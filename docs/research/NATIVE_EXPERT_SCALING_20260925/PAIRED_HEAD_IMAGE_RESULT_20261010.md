# Paired head/domain probe: encoder room verified, not full quality

10 October2026. COMPLETE / NONLINEAR_ENCODER_ROOM; independent complete audit
PASS. Goal incomplete. [Frozen protocol](PAIRED_HEAD_IMAGE_PROTOCOL_20261010.md),
[binding](paired_head_image_binding_20261010.json),
[result](paired_head_image_result_20261010.json),
[audit](paired_head_image_stored_adjudication_20261010.json).
No first fault, no restart, no source generation, model training, DEV or T4 call.

## Changed uncertainty and result

One fixed NEW paired head/gain/domain, warm x=phi/a: metadata-only first/middle/
last labels of24 FIT cases,72 labels/12 domains. This is not the old Adam1 head
or scale probe replay. The previous paired codec's full-FIT quality FAIL remains.

| Measurement on these SAME72 labels | Initial linear code | Feasible optimized code |
|---|---:|---:|
| Mean categorical KL | .8715961713472219 | .0065297330512313735 |
| Argmax disagreements |17/72|0/72, real AND original C readout|
| KL upper / initial |1|.00749169542718206|
| Worst returned KL |7.316284324380215|.23083112300683872|

99.2508% reduction in mean KL demonstrates much better feasible coordinates
for THIS head. It does not establish that a nonlinear causal encoder is learnable,
that the full4422 labels pass, or that the chatbot is useful.

Frozen encoder-room gate (upper <=.25 initial) PASS. Mean selected KL<=.01 and
argmax<=1% PASS; selected maxKL<=.05 FAIL. Therefore sampled quality FAIL.
All72 numerical tangent-ball lowers are zero. Whole-case/cohort/domain lowers
are also zero using actual case denominators and zero for omitted labels.
`head_domain_floor_present=False` means no obstructing lower was demonstrated;
it DOES NOT mean the optimum/floor is zero. Only15/72 gaps<=1e-5; maximum gap
.230831123. The iteration cap512 was reached, not global convergence.

Worst label is `broad_fit_smol_magpie_ultra_022`, index255/position1506:
initial KL2.1981932634 -> upper.2308311230, lower0, norm4.62901295.
It already had the teacher argmax initially: categorical mass remains a problem
that top1 agreement alone hides. Next worst is same case index0, KL.031193462.
The returned norms span1.34029316..8.01139440 in radius16; minimum carrier
radicand191.81755971 and zero rounded negative coordinates. The returned solution
does not approach this radius boundary, but that is not a global optimum claim.

## Actual original native readout

Same frozen F32 head and original four functions,72 final-readout rows in one
DLL invocation, no full causal engine call. All native numeric gates PASS:

- score max absolute discrepancy versus real uncentered head3.41557341486e-5;
- norm relative discrepancy4.62367063198e-8;
- real/native mean KL4.58963726916e-13;
- zero real/native argmax changes and zero native/teacher argmax disagreements.

Original engine.c unchanged. This qualifies final-readout arithmetic for these
codes; it does not qualify a causal state generator, routing, experts or speed.

## Freeze, complete audit, costs and exact counts

Freeze302f4fb46662ab0df2a20ddccfa6fda29b61829d;
binding916e2bf4e1a018bf04eda1c09733d0b20d8830f9d9eec4ab5a4d0a10b38017b7,
84 input extents/2,658,119,002B.
Result2352fc9a3027ec068101984bbb6b38efe2f00033007b1b776725342c14f47b5c;
audit aaed7399a838c8498096acae957686035f26c1486fd5378826acba230d252759.
Native ABI/AST preflight has zero head/descent/source calls.

Capture session70152 CLOSED/exit0, launcher19436 created14:27:09.239356+02:00,
worker25972 created14:27:11.993038+02:00; both gone before audit. Held166.062s,
worker prewrite157.219s/final print157.313s. Worker OS1,693,650,944B +launcher
31,154,176B=1,724,805,120B<=4GiB. GPUallocated357,857,280/reserved375,390,208B
<=2/3GiB.45 namespace files190,284,282B plus57,371B result=190,341,653B<=256MiB.
33 checkpoints,512 latent descent updates/36,864 latent row updates,
2,688 actual objective/gradient evaluations (including rejected backtracks),
193,536 objective-gradient label evaluations;432 extra score-row products.
Each fg has a score product and probability-gradient product; teacher moment
product once. These are latent variables, not pretrained weight updates.

Audit session68070 CLOSED/exit0, launcher28500 created14:30:16.197356+02:00,
worker20028 created14:30:18.708805+02:00; all gone. Held24.343s, prewrite18.969s/
print19.078s. Worker OS953,110,528B+launcher30,937,088B=984,047,616B<=2GiB.
All84 inputs/45 namespace outputs hashes/custody, exact selected teacher and
phi/gain, all72 initial/final upper/final lower codes and gradients/bounds,
all33 saved current objectives/gradients (2,376 rows), feasible/monotone envelopes,
actual native state/norm/head F64 reference, original four bodies, all metrics/
branches/call counts/resource gates independently checked. Complete audit PASS.
Max full-score reconstruction discrepancy4.26325641456e-14, gradient4.84234874421e-12,
metric9.50794998289e-13; native versus F64 on actual stored normalized native
state3.27829369553e-5.36 stored objective-gradient batches/2,736 stored head-row
reconstructions, zero optimizer updates/native binary/source/model replay.
Bounds are F64 convex supporting-plane calculations, not interval certificates.
No quality-triggered audit omission. Both held phases totaled190.405s.

## Decision toward the converter

The branch name does not prove nonlinearity is necessary: a different KL-trained
linear encoder remains possible. Do not extend unchanged latent descent merely
to shrink its weak lower bound.
The current pair exposes encoder room that the quadratic surrogate missed.
[Encoder sufficient statistics and exact finite-reference KL identity](PAIRED_HEAD_IMAGE_ENCODER_ALGEBRA_20261010.md)
connect this result to categorical training. Keep causal recovery in the original
SSM/SWA/ternary/LUT runtime as the next problem. Existing compact native states
and old learned head are in a different coordinate system; attaching this head
alone would not constitute a coherent initialization or architecture refutation.
[Next causal-coordinate qualification](CAUSAL_READOUT_COORDINATE_NEXT_20261010.md)
uses retained actual27 C logits to test state recovery and a fused categorical
readout path. Before a new joint campaign, qualify a low-cost readout/coordinate adaptation
from the existing causal state, using FIT only and unchanged held-out criteria.
No useful chatbot, accepted50 tokens/s, useful n, physical DRAM, structured
CPU IDs/mass or second-family admission has been demonstrated.
