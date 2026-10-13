# Actual all-layer BF16 generator inputs and three-site continuity

Freeze before observation. Actual SAME pretrained Falcon-H1-1.5B-Instruct,
all411 checkpoint tensors/source config/corpus/helper/runtime. Exact old
tiled source method descriptor must match. Forty-eight complete original
forced histories,one base prefill each/zero initial state/use_cacheFalse,
no LMhead,answer,generation,optimizer,native/RESERVED. All24 in_proj pre-hooks
save actual args[0] BF16 original bits AFTER .625/BEFORE projection;assert
shape1xhistoryx2048/finite/one observation per case/site. Do not substitute
residual/FFN input or apply multiplier again. Capture full original7 fields
x/B/C/gate/delta/y/output only0/12/23,identical observation equations to old
capture. Every layer MuP bufferBF16 bits checked. No model weights changed.

Before/after source prefills,verify each actual GPU parameter's canonical
BF16 bytes against its individual checkpoint tensor SHA,all411/all1,554,863,488
coefficients. Actual FP32 parameters must exactly equal BF16-expanded values
before canonical conversion. Actual parameter IDs/_versions unchanged. Data
are true input observations of this new source execution;continuity with old
source execution is separately gated,not assumed.

Independent data audit:all1152 H/full1008 continuity-field observations,
every BF16/F32 value finite/shapes/extents/data hashes/IDs/positions/parameter
witnesses before/after and exact checkpoint digests. EVERY case/site/field
continuity relative RMS<=1e-4 to adopted source;bit mismatches/whole/centered/
label/domain/mean/worst retained. All audits continue even continuity failure.
Capture does not prove CPU replay/original CUDA reduction/all-layer recurrent
quality or compact causal encoding. No quality/speed/n admission here.

LocalRTX3060 family1800s,reserve90s,OS12GiB,GPU allocated10GiB/reserved11GiB,
new ordered E<=6GiB. Payload4,231,620,960B:all24 H2,216,460,288B plus full7
fields3 sites2,015,160,672B. Charge manifests/method/parameter witnesses/logs.
Parent owns exact worker PID/creation time,hidden launch,hard deadline/
resources/log/output caps,terminal/first fault/partial cases. Terminate only
that owned worker if cap breached;observation timeout alone is not failure.
Preserve foreign processes;desktop GPU sharing is recorded,no perf admission.
Pin resolved Torch shipped DLL tree/CUDA/CUBLAS,nvcuda/NVML/driver query,
NumPy ABI/BLAS/psutil/safetensors/HF/source/helper/corpus/checkpoint data.
Preparation<=900s/OS6GiB;data audit1200s/OS6GiB/CPU0..5/sixBLAS.
No T4/hours allocation;user's announce/launch/PAUSE rule applies to later hours.
