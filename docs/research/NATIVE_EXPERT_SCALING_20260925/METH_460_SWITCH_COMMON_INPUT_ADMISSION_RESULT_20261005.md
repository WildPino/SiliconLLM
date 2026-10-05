# METH460: exact common-input A16 sharing admitted, speed remains unmeasured

5 October2026. ONE model-free command exit0. Freezea467981/first-run656fd15.
ALL8 admission gates PASS. Goal ACTIVE/INCOMPLETE. No native compilation/run/
weight value read/fit/download/GPU or engine edit. Current source algebra and
manifest/counter metadata are the new evidence; no fresh model quality result.

First459 catalog physical-EOL stop remains immutable e0a98ab,1.078s/no native
observations/zero case counts.460 changed only namespace, first-fault binding
and catalogue identity; strict qualified source/engine/metadata bytes unchanged.
[First fault](METH_459_FIRST_CATALOG_FAULT_20261005.md),
[460 protocol](METH_460_SWITCH_COMMON_INPUT_ADMISSION_PROTOCOL_20261005.md).

## Established operation contract

Both original manifests have D768/F3072/12encoder/12decoder/6denseFFN layers
per stack.36 legal fanouts/source:12 encoder selfQKV,12 crossKV preparations,
12 decoder selfQKV. All96 selected I8 matrices/source have exact768x768 shape,
distinct original offsets and original row scales. Binary manifest hashes fresh;
full payload VALUE/hash evidence inherited458, only size/mtime refreshed here.

For the SAME input vector x and FE_TONEAREST, all branches produce the SAME
A16 absmax/scale/F32 division/RNE/clamped codes. Compute them once, retain every
original integer dot, every row scale, F64 product order and finalF32. Output
buffers must stay disjoint/nonaliasing, with original token-major/cache strides.
Self QKV can share after the same selfnorm. CrossKV share the final encoder x.
Cross-attention Q after another normalization/residual, output O, WI/WO nonlinear
boundaries and the head cannot be folded into these groups by this argument.

General jointQKV already exists in phase60 fp32 SWA. Layer-major/token4 kernels
share a different dimension (multiple query vectors). Qualified374/388 still
recompute each branch's A16 codes and open separate parallel regions. This is
an absent specialization under the qualified contract, not a novel algebra law
or a repository-wide semantic absence claim.117 tracked C/H files surveyed,
eight physical-EOL differences explicitly retained with identical canonical
HEAD text; no old source rewritten. Strict compiled/qualified bytes stay exact.

## Removable work, ALL cases charged once

For Le/Ld layers, S source positions and T actual generated positions:

    A16 calls removed=2Le*S+Ld*S+2Ld*T
    OpenMP regions removed=2Le+Ld+2Ld*T

mv_batch creates ONE parallel region per matrix but quantizes S query vectors.
Logical calls count query vectors. These are distinct observables.
ALL1536 inherited warm/measured dense counter rows rederived exactly from
manifest dimensions and dense-FFN layer placement; no generation rerun.
Aggregation below counts one deployment generation per case, not repeated warm
or measured copies. All15 nonaccepted256 cases and their170 generated IDs count
toward work; accepted quality numerator remains895 ordinary IDs.

| Count over ALL96 cases |128 source|256 source|
| --- | ---: | ---: |
| Actual generated IDs including rejected |1142|1065|
| Fanout A16 evaluations before / after |208152 /80520|205380 /79596|
| Removed A16 evaluations |127632|125784|
| ALL dense A16 evaluations before |329784|323316|
| Fraction of ALL dense A16 calls removed |38.701696%|38.904354%|
| Fanout parallel regions before / after |46872 /16008|44100 /15084|
| Removed parallel regions |30864|29016|
| ALL dense parallel regions before |103992|97524|
| Fraction of ALL dense regions removed |29.679206%|29.752676%|

These fractions are CALL counts. Other dense vectors can have width3072;
they are not fractions of scalar quantizer arithmetic, cycles or whole time.
Fanout integer products remain122,773,045,248128/121,138,053,120256 over all
cases. Logical weight-code bytes equal these I8 element counts and remain
unchanged. Matrix arithmetic/weight storage/core width/expert capacity reduction
is ZERO. No physicalDRAM/cache/locality or rate benefit inferred from this.

## Integer and precision bounds

Use the ORIGINAL nonsaturating AVX2 integer dot. Including I8-128 and A16±32767:

* Pair MADD absolute bound8,388,352 <2^31.
* I32 lane at768 columns402,640,896 <2^31.
* I32 lane at4096 columns2,147,418,112 <2^31.
* Final I64 dot at4096 columns17,179,344,896 <2^63.

Fanout does not change a sum length/bound. Final dot may exceedI32, so keepI64.
No new blocking/saturation/reassociation is licensed. Exact algebraic sharing
still needs native primal/output checks for its actual C implementation.

## Decision and scope

Admit ONE exact common-input A16 fanout WHOLE inquiry. A16/F32norm/route/head
precisions and all coefficient bytes unchanged. Native implementation will have
its own prospective fixed whole-cost/quality/resource gates; no step/time benefit
is inherited from38.7% removed calls or from458's40..44% dense fraction.
[Next bounded native proposal](SWITCH_COMMON_INPUT_FANOUT_COST_NEXT_20261005.md).

The inquiry must end in one decision, not an indefinite kernel sequence. Useful
n/conditional storage/common pretrained interfaces and fresh complete quality
remain the objective. This optimization can only change common-core execution
overhead; it creates no new expert information or reduction in weight bytes.

Raw634,016B SHA256
`6e1f921d05f08eba48e59c05d167420fc9160732ef3060617e23a4cdbf4646b0`.
[Raw](meth460_switch_common_input_admission_result.json),
[metadata retention](RETENTION_460_20261005.json).
Separate metadata-only audit exit0/ALL6 gates PASS,8.406s; retentionsha
`51b658380511c9fb654c7f30e1b493ba95288feec234d69c24ef4227128f1e36`.
9.594s/peak53,153,792B/13,682,199 metadata bytes hashed before final raw hash.
No native output directory or model values. Original payload size/mtime stable;
their complete hash/generation quality/rates are explicitly inherited458.
First failure/raw/source459 and every catalog byte difference preserved.
