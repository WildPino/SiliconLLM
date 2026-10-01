# METH-240: frozen full-feature mixed-output E16/E160 function-transfer pair

## Question, prior evidence and changed geometry

Does increasing actual conditional output functions from16 to160 conserve
the pretrained layer function and yield useful routing-dependent accuracy?
METH-238 prices full4864 gate/up/LUT+fixed32 output escapes at8.727ms with
source fidelity; METH-239's actual mixed source priors pass unchanged1%
stored-derivative fidelity. METH-231's H2048 plus affine recipe stays
rejected. This variant uses all original nonlinear features,output-only
functions and the physically qualified mixed codec. No width/strength
retry of that closed geometry and no synthetic/copied-capacity claim.

Bind METH-239 result SHA256
`e6e72a440dc18e01411ab85ce79ec3f4e1bfe7e8aa9b1c80a3de43342a437a30`,
85.546MB source-prior snapshot; METH-238 source fixture/result; original
pinned source layer12 matrices; METH-222 actual captured BF16 inputs/outputs;
METH-227 fixed full-input16/10 parent/child keybanks. All fit counts,
parent/leaf label hashes and source tensor hashes must replay exactly.
Same512x128 fit /128x128 consumed validation raw windows; former METH-222
row split is unchanged. They are development data,not fresh independent
documents. Validation is accessed only after all fitting/prerequisites.

## Frozen training order and actual coefficient representation

1. Keep shared gate/up row-Q8 and FP32 SiLU table fixed. phi(x) has4864
   coordinates; no896x896 affine term,input concatenation or feature selection.
2. Fit16 parents from actual METH-239 stored priors. For each parent,
   FP64 variance v of phi on its fit states,floor=max(1e-4*mean(v),1e-12).
3. Derive **all160 child priors before child fitting**:use inherited
   actually stored fitted parent readout and source f(c),J(c) at frozen
   child center. Same one-shot output-only FP64 thin-QR projection and
   fixed32 mixed codec as METH-239. Condition/growth/raw/stored center/
   gradient gates unchanged. Full feature/autograd checks at children0,156.
   If any child prior fails,omit unfitted E160 tensors,save prerequisite-only
   snapshot/result,do not fit children or access validation.
4. If all child priors pass,fit160 children using their own actual priors
   and inherited parent v/floor. Every occupied cell is fit,including the
   known18-state/one-unique-input child156; source derivative information
   supplies slope knowledge that centered data alone cannot provide.

For parent/child,FP64 z=phi(x),Wp=decoded actual mixed coefficients,
bp=stored bias,R=y-(z*Wp^T+bp),where y is the **captured BF16 donor output**.
Center z/R,solve source-anchored ridge:

`(zc^T*zc + 1024*diag(v))*deltaW^T = zc^T*Rc`.

Use equivalent dual solve if n<4864,primal otherwise. **Tau1024 sample
equivalents is inherited,fixed**,not selected using this validation.
Normal residual/cross norm<=1e-7,denominator floor1e-12. Cast Wp+deltaW to
FP32 and encode the actual fixed32 BF16 escapes/row-Q8 remainder. For final
effective decoded W,bias=FP32(bp+n/(n+1024)*meanR-(W-Wp)*meanZ). Mean-shift
rounding error/meanY norm<=1e-5 (norm floor1e-12). Continuous fit algebra
does not override actual encoded-function scoring.

Store shared controls,16 source-parent priors,16 fitted parents,160 source
child priors,160 fitted children,parent feature variance. Read every tensor
exactly. Require **176 decoded FP32 effective coefficient matrices distinct**;
bias/physical metadata alone cannot establish this. Also retain physical
code/scale/ID/exception hashes. No noise/perturbation to force distinction.
65536 unique fit states;131072 readout-fit state exposures if children fit,
65536 otherwise. Source derivative compilation is reported separately.

## Fixed held-out development comparison and decision

After every prerequisite/solve/readback/distinctness passes,reopen the
128 validation windows. Compute fixed full-input route labels and phi;
evaluate **actual row-scale-after-reduction plus indexed BF16 sum+bias**,
grouped by selected function to bound GPU memory. Do not use a predecoded
combined-weight GEMM as the scoring operator. Controls:

- source parent priors;
- fitted E16;
- source child priors (inherit fitted parent/source derivatives);
- fitted E160;
- E160 with leaf+1 modulo10 within the same selected parent.

All window output energy must replay METH-227 exactly. Pool SSE/energy
and retain every window/arm. Frozen gates:

- complete E160 function SSE/energy<=.01;
- E160 SSE<=90% of E16,rotated E160,source-child prior and source-parent
  prior respectively;
- paired E16-E160 gain P05>0,10000 bootstrap window draws,seed240241.

No best checkpoint/history/strength/precision-count selection. Failure
stops this fixed mixed full-feature conditional recipe before native routed
bank/full-model/new quality. A pass licenses actual stored-bank C arithmetic,
routing/LUT/DRAM and n-scaled pool measurement before complete-model promotion.
This one-layer consumed function comparison is not independent LLM quality,
large donor capacity or a>=50 accepted token/s model. Real second family/
approximately10B and eventual100B when resources permit remain required.

## Resource budget and command

One local3060 run,six host threads;30min after imports,20GiB RSS,10.5GiB GPU,
4GiB free disk before launch,<1.7GB snapshot+reports. GPU prediction groups
states by function and decodes one readout at a time; no whole FP32 bank
or repeated128-state weight tensor expansion. No T4/new source download.
Partial records preserve fitted parents,all source-child audits and completed
fits. Exceptions save stage/progress;never restart an unconfirmed-live job.
Commit source/protocol before fitting and pin result CRLF byte hashes.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth240_mixed_conditional_pair.py --checkpoint results/native_expert_scaling/meth240_layer12_mixed_conditional_functions.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth240_mixed_conditional_pair_result.json
```
