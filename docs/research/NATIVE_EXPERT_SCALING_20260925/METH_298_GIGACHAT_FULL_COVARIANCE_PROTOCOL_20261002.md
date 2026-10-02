# METH-298: real GigaChat routed-input covariance, fixed rank192

## Uncertainty, reused evidence and decision

METH-180 ordinary weight rank192 retains47.243% median energy; METH-181
diagonal input weighting retains49.152% on a distinct domain and fails.
Neither measures cross-channel activation moments. After297 closes the
fixed small native profile, test this different compact-transfer variable
on the locally bound11.480B GigaChat pretrained MoE. The generic donor port
remains paused. Use exactly the previous27 identities: layers1/13/25,
experts0/32/63, gate/up/down. No sample/rank changes after observation.

Collect actual routed projection inputs from the source BF16 GGUF graph
using the SAME source-tokenizer ID calibration splits from181:106 English/
code/technical chunks for fit;125 Cyrillic chunks for evaluation. These are
previously used calibration domains, not untouched model-quality data.
No candidate quality is used to select source IDs or fit directions.

The new variable is **full uncentered cross-channel input second moments**.
Exact FP64 factors at rank192 are a necessary representation screen before
factor precision/nonlinear/model-quality/native cost checks. They are not
an export candidate qualified for deployment by themselves.

## Bound source and isolated apparatus

- Source revision `189fff27a1dee68473960c3d5bca53e0e07a3191`.
- BF16 source GGUF SHA `fabc8056f57e230ae9e6aadceb45f4abf9e5d7031fbfe8eca2671d871ef35d47`.
- Prior180 SHA `ecc9a3432a25a4a7fc0f2881c421947aeb097b7a4cf32a746ebcdefa10081b8d`;
  prior181 SHA `5b226744cde42404a4811ff32740a6b744328db0d697d3acafaf8880c0f3d7b0`.
- Fit IDs SHA `3d611152b4a145f2219a34e7ad5bc2e8929a9a600321272e9e258b92924c250a`;
  test IDs SHA `7f5333d1d52a5c94ad0026ff0f38cc022a3aeeecf1b884085b56abe8c795c524`.
- Old fit/test diagonal GGUF SHA respectively
  `ef5de87fc4fa0d1382ed2988e59e9472fb1d7fb2c9bed0e8edcced6ff9bd4f9c` /
  `5c529f9009f0f241dfc50d0a1f598b0e183c137693a951f04963dc729a344d08`.

The existing llama.cpp build is pinned at
`5b335f413e4f73b0809c4fe39af894efbcc6a0d2`. The source-ID-patched imatrix
file SHA is `49c71316c66717d8daccc90bf16c17bf4589239903eabe386d273d657777b9f3`.
Generate a SEPARATE imatrix compilation unit by four read-only insertions:
include capture header, record selected inputs immediately before original
squared-sum accumulation, label current chunk/batch before decode, finish
capture after completed computation. Reversing the four insertions exactly
recovers the original source. No source graph/operator/arithmetic changes.
No existing runtime source/library/executable is modified.

[Controller](../../../benchmarks/native_expert_scaling/meth298_gigachat_covariance.py),
[capture hook](../../../benchmarks/native_expert_scaling/meth298_capture_hook.h)
and [build manifest](meth298_capture_build.json) bind the generated source,
all seven reused static libraries, compiler/OpenMP library and18 existing
headers. Build took9.547s; executable SHA
`f87c675bdd904dcd305fa1cdf07e5ca8ca3e516b6dcaf2f98021aa2f0de53c8d`.
Build observes no model input/spectrum. Freeze source/build/protocol bytes
in Git BEFORE collecting data. Collection/scoring verifies these bytes.

## Capture closure and stops

Six CPU threads, batch128, context512, GPU layers0, no perplexity. Each
chunk clears KV; the existing collector replaces its first ID with BOS
when the source GGUF requests it. Document boundaries may cross chunks;
54,272 fit and64,000 test IDs are used, trailing IDs unused. This matches
the prior calibration approximation, not per-document heldout scoring.

Capture binary `M298ACT1`: each row has eight little-endian U32 fields
(layer,organ,expert,chunk,batch,selected slot,token-in-batch,width), followed
by width F32 input values. Organ0/1 are gate/up width1536, organ2 down
width1280. A UINT32_MAX layer/zero remainder footer records total U64 row
count; exact EOF required. The callback uses the original tensor strides
and selected expert IDs; records only actually selected real experts.

Require all27 inputs, >=384 observations each, finite values/valid indices,
no duplicate expert/token coordinates, completed footer/count/EOF; gate/up
coordinates and inputs must be byte exact. Reproduce original row-order
F32 squared accumulation and require exact equality to the newly written
imatrix. Against prior split imatrix, counts must be EXACT and each channel
moment relative difference <=1e-4. Failure stops spectral scoring; preserve
failure/log/partial binary before a narrowly frozen apparatus repair.

Expected two captures roughly45-55minutes total plus source hashes and
spectral analysis. Each child hard stop **55minutes/50GiB resident RAM**;
each capture output hard limit **2GiB**. No GPU/T4/downloads/model job
overlap/CPU performance claim. Scoring hard stop20minutes/12GiB parent RSS,
six BLAS threads. Preflight available RAM64.15GB of85.85GB decimal.

## Fixed factor comparison and gates

For each original BF16 projection W (outputs x inputs), retain every real
routed fit input X without centering. Compute Y=X W^T in FP64, eigendecompose
Y^T Y, retain its192 leading orthonormal output directions U. Form
`W_hat = U U^T W`, an actual rank192 factorization with no bias/intercept,
covariance inversion or regularization hyperparameter. This is optimal
rank192 squared fit-output reconstruction; independently verify direct fit
residual energy against retained eigenvalue fraction to absolute1e-8.

Evaluate `1 - ||X_test (W-W_hat)^T||^2 / ||X_test W^T||^2` on distinct
Cyrillic routed inputs. At the SAME rank and factor element count, fit
ordinary W SVD and diagonal-weighted `W diag(sqrt(mean(X_fit^2)))` SVD and
evaluate both on these actual test inputs. The test-domain optimum is a
diagnostic upper bound, never a fit/selection input. All three estimates
must be <= oracle+1e-8. Retain all27 results, not just favorable rows.

Full-covariance factors are eligible for separately frozen factor precision,
nonlinear composition and complete donor-relative quality/cost evaluation
ONLY IF:

1. Test energy median >=95%.
2. Every sampled test energy >=90%.
3. No row worse than ordinary factors by >0.5percentage point.
4. No row worse than diagonal factors by >0.5percentage point.

Otherwise close THIS fixed full-covariance rank192 representation. Do not
increase rank/change sample/reweight domain/relax gates using these scores.
A fail does not refute trained nonlinear or different representations.

## Active-cost and scaling limits

Three rank192 int8 factor pairs would contain1,622,016 elements per expert,
versus5,898,240 source elements. Four selected experts in25 MoE layers price
162,201,600 factor bytes/token before scales versus actual Q4 routed
356,106,240 bytes/token. This is hypothetical exact-int8 storage arithmetic;
the screen computes FP64 factors, not int8 quality or physical DRAM traffic.
Keeping all other Q4 organs leaves822,289,504 addressed bytes/token, still
above the560MB streaming allotment. Therefore expert factors ALONE cannot
license a generic full donor export. MLA/head/shared paths need their own
compact treatment and measured complete budget before deployment work.

This cell tests whether existing pretrained expert capacity admits a cheaper
selected consultation representation. It adds no experts and does not prove
useful new n, RAM-scale CPU LUT/router cost,50 acceptedtok/s or100B transfer.

## Commands (after freeze)

```powershell
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth298_gigachat_covariance.py fit
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth298_gigachat_covariance.py test
.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth298_gigachat_covariance.py score
```

Run sequentially; require each capture success before proceeding. Preserve
all raw captures locally, small byte bindings/audits in Git, and complete
capture stderr/stdout logs. No model quality/rate inference from assay time.
