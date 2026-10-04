# M421 protocol: diagnose real directional gradient-check conditioning

Freeze NEW source/controller/protocol before any diagnostic numeric outcome.
420 retained b512542, raw17f60d043cff598cec0d3e85dfa4d23bdad67b83635422891e92729528056427:
ALL384/4895 native forward/full head byte-exact, tiny FD/STE controls PASS, first
real source256 factorA directional FD FAIL0.0002754480190625858>same1e-5.
No actual slope/projection was recorded420. No fit eligible; failure immutable.

Question: does the fixed real FD fail because of loss-subtraction conditioning,
ReLU branch crossings, or an actual smooth-autograd/reference mismatch? NEW
diagnostic only, no relaxed threshold/rank/seed/coordinate or fitted parameters.
Fresh420 raw and all384 complete prior baseline archives; fresh full source
payloads/manifests and419/420 helper sources,418 raw and exact selected native
first captures. Do NOT rerun all384 expensive forwards or acquire new data.

SAME TWO first teacher.book0.case0 positions, rank8, seed419+n, original F32
zeroA/C/nonzeroB/D converted exactly to F64 smooth twin, opposite native-source
prediction target/T1. SAME fixed Rademacher factor directions/epsilon1e-4 as420.
Record actual projections/plain central-FD slopes and actual ReLU +/- crossings,
min absolute preReLU and zeros. Original backward helper/cross-entropy unchanged.

Additional DIFFERENT diagnostic numerical forms:
- Equivalent centered cross-entropy: fixed baseline maximum-ID k,
  s=z-z[k], log1p(sum(exp(s[j]),j!=k))-dot(target,s). Removes subtraction of
  leading full logits while preserving objective mathematically. Full-real
  old/centered scalar loss difference<=1e-12; independent F64 logits<=1e-12.
  SAME central epsilon1e-4, no epsilon sweep or changed error thresholds.
- Independent local complex extension of literal norm/matrix/residual/head:
  squares x*x, fixed real ReLU positive mask/zero convention, no absolute-value
  norm. ONE imaginary step1e-20 in each SAME factor direction. Derivative
  Im(centered loss)/1e-20 vs autograd projection at SAME1e-5 with same1e-8
  denominator floor. This is a diagnostic derivative of the LOCAL smooth model,
  not a gradient through native rounding/top1 or global complex differentiability
  of ReLU. Known quadratic directional derivative arithmetic check<=1e-14.
  Split real matrix/complex-vector dot avoids full complex vocabulary matrix.
- Torch centered-loss gradient vs old autograd gradient recorded, no optimizer.

No alternative point/direction/rank/seed chosen from observed gradient. Both
sources/all four factors recorded irrespective of original central-FD gate.
If old autograd matches independent imaginary AND centered central derivatives
<=1e-5, zero ReLU crossings, supports a NEW stable-loss/check protocol. Does NOT
qualify420 retroactively or authorize fitting. Otherwise leave mismatch/crossing
unresolved with full evidence. Numeric FAIL/error retains first diagnostic.

CPU only, Torch2.6.0+cu124/NumPy2.4.6, threads/BLAS1/logical0,<=120s,
RSS/Windows peak working set<=3GiB, retained output<=16MiB. Full-head dequant
workspaces charged, no model job overlap except precisely identified old user
daemons. Same source immutable; no GPU/T4/network/new corpus/native artifact/
quality/rate/LUT/physicalDRAM/useful larger-n or additional-family claim.

After freeze:
`.venv/Scripts/python.exe benchmarks/native_expert_scaling/meth421_switch_gradient_conditioning.py --out docs/research/NATIVE_EXPERT_SCALING_20260925/meth421_switch_gradient_conditioning_result.json`

Goal active/incomplete. Whole genuine function benefit and future quality/rate
cannot be inferred from derivatives. All gradient/fit work after diagnosis must
have a new prospective objective/resource/negative-control contract.
