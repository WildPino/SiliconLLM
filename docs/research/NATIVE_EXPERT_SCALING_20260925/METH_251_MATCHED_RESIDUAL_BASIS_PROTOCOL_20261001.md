# METH-251: frozen matched residual-space full-parent hierarchy

METH-250 diagnoses2.94% inherited versus15.40% new stored output-space
coverage of actual-parent centered fit error. Compare E16/E160 with the
same new rank32 space; do not count extra active directions as expert gain.
This is a nondeployable continuous pilot before any new codec/kernel.

## Binding and fixed hierarchy

Bind METH-250 result
`8220ad28bd577730f78bb02827e8b09f66b9d2c010a453da009bbdb69a920bfb`
and basis
`c457466afb56ab19ab5f7872309a8e73ef51db5e726e6cceaff51cafbe4095a0`.
Require all250 gates/pass decision and identical source parent/capture.
Bind original METH-247/METH-240 results and snapshots, source/capture/
router and METH-238/246 operator controls. Preserve every METH-247 field
except unused old E160 factors. Original16 parent coefficient hashes,
every fit SSE and original `sum()` normalized aggregate remain exact.

Use same512 fit windows/128 states, FP32 row-Q8/LUT features, existing
16 parents/160 children/keys and original parent feature variances.
For each stored new BF16 left L, FP64 pinv=(L' L)^-1 L', Gram condition
<=1e8, inverse identity Frobenius<=1e-8. All correction arithmetic FP64.
First fit16 matched parent residuals `target-old_actual_parent` through
L, zero prior, existing centered variance-scaled ridge tau1024. Then fit
160 private residuals `target-complete_matched_parent`, retaining that
entire adapted parent function. Same L/variance/tau, zero slope prior in
supported cells. Thus both arms use old shared function plus the same
rank32 correction space; child effective right is anchor_right+child_right.

Reuse qualified METH-249 centered ridge and mean rule: normal residual
<=1e-7, bias=gamma*mean(target-anchor)-L*D*mean(phi), gamma=n/(n+1024),
mean-conservation relative error<=1e-10, finite. Each effective correction
norm <=.25 of its complete anchor coefficient norm. No parent replacement,
rank/key/strength sweep or source-point value reset.

Only exactly zero centered-feature support gets a projected smooth FP32
source-minus-**complete matched parent** gradient prior, with fit-center
response canceled by bias. Same METH-249 feature16rows/fiveJVP checks
<=1e-5, matched-parent autograd9 rows `[0,1,127,255,383,511,639,767,895]`
<=1e-5, whitened FP64 thin QR condition<=1e8, projected gradient residual
<=1e-5, norm growth<=.25. Bind original BF16 source file/tensor hashes.
No ID noise/copied-expert counting or BF16-rounding derivative claim.

## Artifacts and gates

Store new BF16 left[16,896,32], FP64 anchor right[16,32,4864]/bias[16,896]
and private right[160,32,4864]/bias[160,896] with all unchanged source-parent
fields. Snapshot<320,000,000bytes; save before pooled gates, read every
tensor back exactly, retained old fields unchanged. Require176 distinct
decoded **matched** coefficient matrices. Anchor fit SSE no worse than
original; child fit no worse than matched anchor (1e-12 relative rounding);
child fit SSE/energy<=.01. Numerical/identity controls mandatory.

Only then read same128 consumed validation targets once. Cache original
FP32 reference and complete matched FP64 parent predictions. E160 and
rotated E160 add only respective private corrections to the matched parent.
Record original reference separately and require all original FP32
parent/prior per-window energies/scores exact. FP64-subtraction scores for
all arms. E160 SSE/energy<=.01; E160 SSE<=.9 each matched E16, rotated E160,
original source parent prior and child prior. Paired10,000-window-bootstrap
gain P05>0, seed251252. All pass licenses separate codec/native work;
otherwise close this fixed matched residual-basis hierarchy. No gate tuning.

An eventual merged Q8 latent correction (anchor+child) could address an
ideal558,939,140bytes including the extra BF16 left projection, versus
560MB budget; this is shape-only and requires actual encoding/mean/numeric/
quality/native cost checks. Separate additional anchor-right and private-right
loads would increase that ledger and cannot inherit it. Keep both E16/E160
active geometry matched in any future encoded/native comparison.

Local RTX3060/six threads,20min after imports,20GiB RSS/10.5GiB GPU,
>=2GiB free disk, partial rows/failure stage preserved. No T4/download/fresh
data. No useful large-n/DRAM/route-LUT/full-quality/accepted-rate or second
family/10B/100B claim. Freeze before execution:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
.\.venv\Scripts\python.exe benchmarks/native_expert_scaling/meth251_residual_basis_child_pilot.py --checkpoint results/native_expert_scaling/meth251_layer12_matched_residual_basis_continuous.safetensors --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth251_residual_basis_child_result.json
```
