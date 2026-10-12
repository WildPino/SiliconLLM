# Source-factor scalar-norm native readout: precision PASS, quality FAIL

12 October2026. Complete capture/replay and full independent audit CLOSED exit0.
An actual source-derived F32 readout export now executes original engine head
kernels. Goal ACTIVE/INCOMPLETE:no useful full-model/native throughput admission.

## Exact source-map transformation and cheap norm interface

[Tool](../../../benchmarks/native_expert_scaling/gauged_native_readout.py),
[protocol](GAUGED_NATIVE_READOUT_PROTOCOL_20261012.md). Chosen fully audited broad
FIT-only point alpha.0001. V=Q R with positive R diagonal gives orthonormal Q;
E=W R^T,c=Q^T mu,mu_perp=mu-Qc,b=c-E^T mu. Code t=Rz+c=E^T h+b and
decoded y=Qt+mu_perp preserve the exact same factor predictor in real arithmetic.
No source information recovery,extra training or basis quality improvement claimed.

Target norm uses epsilon_t=(||mu_perp||^2+2048 eps_source)/256. Actual epsilon
**.00013196362901973118** includes the required center contribution;not original
engine's1e-5 or a silently substituted8e-5. Source gamma/head multiplier folded
into A;export C=sqrt8 A Q,beta=sqrt8 A mu_perp. Head reads
`C*(t/r_t)+beta/r_t`. Original engine hsum256/dotf/matvec/rmsnorm bodies retained;
named scalar_rmsnorm changes only epsilon argument/returns inverse RMS. Original
engine.c bytes unchanged. Normalized bias is an explicit extension,not omitted.

QR reconstruction relative5.3243e-16,Q orthogonality max1.3323e-15. Source
reencoding E^T Q-I defect**.0700728204** remains:not an identity skip/projector.
Changing coordinates does not repair generic factor idempotence or causal dynamics.

## Actual full F32 execution and source-relative quality

All4422 FIT/all4386 DEV stored source label states,full65537 vocabulary;139 actual
native API calls (sequential per-row original head,OMP_PFOR empty). F64 t castF32;
exported67,109,888B head,262,148B bias,4B scalar epsilon. Actual normal/inverse
fields and2,308,999,584B full native score bits retained,not a sample-only check.

| Numeric comparison against original factor predictor | FIT | DEV |
|---|---:|---:|
| Largest absolute score error | 5.5018e-5 | 5.5970e-5 |
| Largest inverse-RMS relative error | 8.5142e-7 | 7.7247e-7 |
| Mean KL(original||native) | 7.9447e-13 | 1.1894e-12 |
| Changed original argmax | 0/4422 | 0/4386 |

ALL frozen gauge and native precision gates PASS. Full decoded-vector/code/radius
relations,F64 coefficient/source KL/argmax equivalence and F32 head/bias/epsilon
casts verified. This isolates normalization/kernel precision from representation
loss:the exported reader preserves the already poor source-factor model.

| Native source-relative metric | FIT | DEV |
|---|---:|---:|
| Case meanKL | .99714031 | 2.93424566 |
| Case mean ID disagreement | 22.6093% | 41.0883% |
| Label meanKL | 1.14831061 | 2.91412000 |
| Label disagreement | 24.8530% | 40.0365% |

ALL case/label/domain quality gates FAIL. No source model generation,actual
compact causal state acquisition or whole-engine CHATBOT run here;last native
chatbot remains0/16. Native calls total30.318s;NOT accepted tokens/s or a clean
head timing admission:teacher-injected states and external publisher retained.
Norm now needsO(256),bias addsO(65537),head still16,777,472 dense MAC/selected
67MB. Actual end-to-end/router/DRAM/core/ternary/useful-n costs remain unqualified.

## Independent complete proof

[Auditor](../../../benchmarks/native_expert_scaling/gauged_native_stored_audit.py),
[frozen protocol](GAUGED_NATIVE_STORED_AUDIT_PROTOCOL_20261012.md),
[scope erratum](GAUGED_NATIVE_AUDIT_SCOPE_ERRATUM_20261012.md).
All**577,249,896** F32 score words,**2,254,848** normalized state words and**8808**
inverse words EXACTLY match a separately compiled scalar reference.
[Reference C](../../../benchmarks/native_expert_scaling/gauged_native_scalar_reference.c)
uses explicit fmaf/eight logical lanes/scalar left-to-right sums/fused bias;
no original engine body or SIMD intrinsic copied,no producer DLL replay.
Vectorization/automatic contraction disabled;original producer disassembly
retained to confirm norm/dot/bias arithmetic. Both native computations are head
components;neither is a whole causal engine replay.

Every actual source head coefficient/gamma/scale,complete Q/R/E/b/center/epsilon
map,exact F32 casts/all source codes/all full source probabilities/case/domain
metrics/gates checked independently with2048 feature-width sums and logaddexp.
No second QR refit. Max relative2.8058e-12/scalar7.8050e-11;full input/output
before/after hashes PASS. No numerical/mechanical fault in completed replay/audit.

## Interruption and custody retained

Original session83696 lost during intentional turn interruption:no normal result
or terminal,handle absent and no owned Python process on recovery. Do NOT infer
exit0/1,after-hash closure,full DEV or peak from its partial files. FIT completion
had appeared in stdout;full-size DEV score mapping can still be incomplete.
[Interruption seal](gauged_native_capture_interruption_20261012.json) hashes every
original partial file. Original namespace remains untouched.

New [replay invocation](gauged_native_readout_launch_replay1_20261012.json)8ff74ce
uses EXACT same frozen code/protocol8318bfa/[binding](gauged_native_readout_binding_20261012.json)
SHAb8c2664da38a61305fc13b915fdb8620fe59bc4f74adaa22782bccd7b70da540,
same thresholds/inputs/budgets,new report/output namespace;prepare4.25s.
Producer47035 CLOSED0,241.218s/OS3,132,416,000B;9 files2,557,013,564B.
[Result](gauged_native_readout_result_replay1_20261012.json)
SHAcccfed401c0a4b102092ec220713573a30b0b4ce6071404de38494f175c8cca6;
terminal23bf2d756e794c7d08a5db5b50982ed2d61a0926713ae5920e249ed8686dca8b.

[Scalar reference compile preparation](gauged_native_reference_preparation_20261012.json)
83aed8d,closed0/.641s printed total,no numeric inputs. Audit source5a90956/
[invocation](gauged_native_stored_audit_launch_20261012.json)eebc767.
Audit81538 CLOSED0,406.891s/OS3,205,128,192B.
[Adjudication](gauged_native_stored_adjudication_20261012.json)
SHAead4d9ade71c446deb6d8519662c9c44b3906df52b93bf3586af08c9417c1e69.
All900s/6GiB/3GiB capture and900s/6GiB audit limits PASS. Principal source/runtime/
compiler/DLL/log/disassembly and complete dataset/output extents bound.
JSON keys epsilon_F32(extent) and epsilon_f32(number) are case-sensitive;PowerShell
inspection needs ConvertFrom-Json -AsHashtable,not case-insensitive PSObjects.

## Selected next toward useful conversion

The native readout precision/norm interface is now locally reproducible. Quality
and code generation still fail. Next [actual native head domain feasibility](NATIVE_HEAD_DOMAIN_FEASIBILITY_NEXT_20261012.md)
PROPOSED/UNIMPLEMENTED separates fixed head-image room from current linear and
future causal encoder limitations. It uses the just exported head/positive-sphere
domain and actual source references,not unchanged old paired255 latent descent.
Any better oracle codes still require causal SSM/SWA/ternary generation,useful
expert functions/IDs+mass/physical DRAM and same useful artifact>=50 across
declared families/scales. No new long epoch,T4,RESERVED/longalign or deployment.

Ordered D namespaces;original engine/foreign changes/publishers preserved.
Full [session closure](gauged_native_session_closure_20261012.json) records actual
PIDs/creation/hashes/closed handles. No owned run live;goal ACTIVE/INCOMPLETE.
