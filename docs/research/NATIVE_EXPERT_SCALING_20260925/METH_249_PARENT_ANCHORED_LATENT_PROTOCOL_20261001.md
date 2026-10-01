# METH-249: frozen full-parent anchored continuous latent hierarchy

METH-248 removes final encoding and still loses27.78% consumed count gain.
Test the [changed hierarchy](PARENT_ANCHORED_LATENT_HIERARCHY_PROPOSAL_20261001.md)
before any new codec. Retain the complete actual METH-247 E16 function,
including base/factors/bias, and add a private correction through its fixed
BF16 left factor. No parent retraining,rank/key/variance/tau changes.
This is a nondeployable FP64 pilot,not an actual encoded/native bank.

## Bindings and fit algorithm

Bind METH-247 result
`a433b502960ee2c50c5ab1fa1b214d61327e4584935d27fa10682b08a4ab7ee9`,
its actual151,680,804byte snapshot,METH-248 diagnosis
`dda4fc408cf4b7b51fa2909d0e5d8cec9df999a227266d4823b3ff4b26315e7a`,
METH-240 result/snapshot,original capture/router and METH-238/246 native
identities. No download. Replay every fit label/count and actual E16
FP32 fit score exactly;require all parent decoded FP64 coefficient hashes
equal METH-247. Preserve every old field except the unused E160 factors.

Use same512 fit windows of128 BF16 states,original FP32 row-Q8/LUT features,
16 parents with10 children each and centered variance-scaled ridge tau1024.
For parent L (896x32),compute pinv=(L' L)^-1 L' in FP64;Gram condition<=1e8,
||pinv L-I||F<=1e-8. Parent P is its actually executed FP32 function,not
an unrounded readout. For child z,y,project residual y-P to latent targets
t=(y-P)pinv'. Center z and t;fit latent residual D by the existing
dual/primal system with diagonal parent feature variance and tau1024.
Normal residual relative to cross product<=1e-7,finite. Supported cells
use zero slope prior;support is exact nonzero centered feature variation.
Use bias b=gamma*mean(y-P)-L*D*mean(z),gamma=n/(n+1024).
Predict P(z)+L*D*z+b with all additive child arithmetic FP64.
Relative mean-conservation error<=1e-10;||L D||F/||parent W||F<=.25.

Exactly zero centered-feature support cannot determine a slope. Only
these cells use a projected smooth FP32 source-minus-parent Jacobian
prior in latent space;no source value reset. Bind actual BF16 source
file/tensors. Qualify feature derivative16rows/fiveJVPs<=1e-5 and actual
parent autograd9 fixed output rows `[0,1,127,255,383,511,639,767,895]`<=1e-5.
Minimum variance-weighted latent coefficient norm uses thin FP64 QR of
feature Jacobian whitened by sqrt(parent variance),condition<=1e8.
Projected gradient relative residual<=1e-5 and effective norm growth<=.25.
The bias cancels its fit-center response. This is smooth FP32 sensitivity,
not the mathematical derivative of BF16 rounding. No ID-specific noise,
forced distinctness,rank retry or derivative fallback in supported cells.

## Gates,validation and decision

Store FP64 delta_right[160,32,4864],full delta_bias[160,896],diagnostic
means[160,4864] with all retained parent/shared fields. Actual snapshot
<320,000,000bytes;read every tensor back exactly and verify old retained
fields byte equal. Require176 distinct decoded composite coefficient
matrices (bias/metadata-only differences do not count). Fit E160 SSE/energy
<=.01 and no worse than parent SSE (1e-12 relative rounding allowance).
Unlike METH-247's independently refactorized final solutions,the fit
prerequisite is nonincrease under an additive zero-centered residual;
the consumed useful-count gate below remains10%.

Only after all fit prerequisites,read the same128 already consumed
validation targets. Cache actual FP32 parent predictions. Compare additive
FP64 E160,unchanged parent,cyclically rotated child within parent,and
original source parent/child prior controls. Require exact old per-window
energies and original FP32-subtraction parent/prior scores. Separately
score all comparisons with FP64 subtraction. E160 SSE/energy<=.01 and
E160 SSE<=.9 of each E16,rotated-child,parent-prior,child-prior SSE.
Paired10,000-window bootstrap count gain P05>0,seed249250.

All gates pass: freeze a separate Q8 private-latent codec/native kernel
experiment. Otherwise close this fixed full-parent/tied-parent-left
hierarchy before codec or C bank work. No fresh independent quality,
full-model task/generation,large-n/DRAM or full accepted rate follows.

One local RTX3060 run,six host threads,20min after imports,20GiB RSS,
10.5GiB GPU,>=2GiB free disk. Preserve completed fit rows/failed stage.
No T4 or source/corpus acquisition. Freeze code/protocol before running:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth249_parent_anchored_latent_pilot.py --checkpoint results/native_expert_scaling/meth249_layer12_parent_anchored_continuous.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth249_parent_anchored_latent_result.json
```
