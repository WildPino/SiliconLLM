# Selected next: preserve p6 weights with bounded integer dot arithmetic

13 October2026. **PROPOSED / UNIMPLEMENTED / UNBOUND.** Native five-trit F64
LUT is numerically qualified but4.2--4.6x slower than original F32 output call
on the declared samples/cache conditions. Do not repeat its unchanged cost
campaign. Preserve successful p6 source q/scales,replace expensive floating
gathers with an arithmetic primitive better matched to this CPU,then measure.

## Separate new approximation from exact implementation

Keep q_weight in[-364,364] exactly from qualified p6. Proposed per-input
F64 activation scale a=max(abs(Z))/32767 and nearest-even actual F64 ratio
r=RNE(Z/a) clipped[-32767,32767],int16;zero input explicit scale0/r0.
This IS new activation quantization,so it has a separate fidelity gate and
cannot inherit the previous BF16-preserving LUT proof. Scales depend on
available input,not DEV fitting/answer/router labels.

Reference dot=sum_i q_weight_i*r_i EXACT signed int64,then convert toF64,
multiply a then weight row scale in fixed order,F32/BF16 RNE. Compare both
quantized integer reference to original native Y and incremental activation
error against qualified unquantized-input p6. Existing per-case extra raw/
BF161% and original source local gates unchanged;FIT/DEV all3 sites/all48.
Field bounds retained:activation coordinate error<=a/2 under qualified
numerical policy;row output error<=weight_scale*a/2*sum_i|q_weight_i|.
No expectation of preserving quality beyond declared observed boundaries.

## Algebraic overflow proof before code

For AVX2 signed16x16 PMADDWD,one product magnitude<=364*32767=11,927,188;
pair<=23,854,376,block128 total<=1,526,680,064<INT32_MAX. Reduce each128-
coordinate block in int32,then accumulate all24 blocks in int64. Full3072
dot magnitude<=36,640,321,536,well belowINT64 and2^53. Weight never-32768,
activation never-32768,so special PMADDWD corner overflow excluded.
Freeze layout/lane reductions/exact bounds/mutation/extrema controls before
launch. EVERY native integer dot must equal independent exact reference.

Independent F64 matrix multiplication of these integers can be an exact
reference:each product and the bound on EVERY partial integer sum are below
2^53,so IEEE F64 represents them exactly. Validate actual runtime behavior/
all bounded operands and converted int64 equality;do not merely compare
relative score norms to hide integer overflow. Then independently verify
the scale/rounding/output/metrics and all actual inputs before cost admission.

## Native/storage scope and decision

Int16 q weights12,582,912B+F64 scales16,384B per site versus nativeF32 O
25,165,824B. This is more bytes than ternary-pattern p6 but fewer expensive
table operations;neither size nor instruction counts prove acceleration.
The q array was previously a diagnostic copy;deploying it changes the
representation and its actual bytes must be counted. The ternary codebook
remains a source representation/possible expert-bank route,not the int16
kernel's physical weight format. Shared-core primitive extension allowed by
goal,without discarding trained output dimensions or declaring useful n.

First fixed numerical preflight of activation change/all-source gates,then
native PMADDWD/int64 qualification,then isolated warm/source-flushed timings
on SAME accurate artifact. No unchanged F64-LUT reruns or premature large
bank/T4 campaign. Proposed producer900s/audit1200s/OS6GiB/new orderedE<=3GiB
if full integer dots/F64 scores/BF16 outputs archived;exact implementation/
numeric tolerances/storage/runtime/cold protocol UNBOUND until frozen code.

Ultimate source CHATBOT->compact causal core+useful selectively consulted
ternary functions->engine.c still needs generator/FFN/attention/head/tokenizer/
OWN histories/RAM-driven n/10B/100B/multiple donors/conversion memory/SAME
useful50. Fast shared readout would be enabling,not completion. No hours/T4.
