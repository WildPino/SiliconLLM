# Output-weighted head channel histories: frozen protocol

9 October2026. Goal ACTIVE/INCOMPLETE. One stored-only comparison: fixed8/head
versus adaptive384,48 source timescales/state256/sites0/12/23. ALL24 FIT full
histories construct maps; ALL24 DEV histories measure them. No model forwards,
optimizer, native, RESERVED or T4 calls. Raw-y PCA control remains immutable.

GY_h=equal-case mean of y_h y_h^T, without trace normalization. Actual source
F32 SiLU gate and full denominator define q=norm_weight*SiLU(gate)/denominator.
GM_h=(O_h^T O_h) Schur E[q_h q_h^T]. Source BF16 coefficients promoted F64;
F32 gate/norm arithmetic precedes F64 covariance. M is the real read matrix
before BF16 feature rounding. F64 symmetrized eigens/PSD roots; negative eigenvalues
relative to maximum absolute eigenvalue <=1e-10 may clip to0 and counts retained.
H=sqrt(GM)sqrt(GY)=U Sigma Q^T. V=sqrt(GY)Q_r Sigma^-1/2,
W=sqrt(GM)U_r Sigma^-1/2. Minimum selected sigma>1e-12; W^T V error<=1e-8.
Failed criteria stop and preserve first fault; no silent rank/head removal.

These maps minimize independent Cartesian-time pair E_(t,u)||M_t(I-VW^T)y_u||^2
per head. The objective excludes same-time correlations, cross-head cancellation,
causal decay, BF16 rounding and chatbot loss. Adaptive ranks start1/head, allocate
336 next largest sigma^2, lowest head index tie, max64/head. Independent global
marginal sorting/prefix witness must agree. Fixed8 and adaptive each carry384.
Independent NumPy F64 roots/SVD verify weighted truncated kernel<=1e-8 relative
and residual tail energy delta/total kernel energy<=1e-10. Store actual Grams,
eigens, SVD, both V/W, all ranks/offsets/norms/tails and audit values.

Actual F32 x'=x_h W_h feeds unchanged source chunk128 SSD from zero state.
Group heads with equal rank: no3072 padded scan. Shared source B/C256 and actual
delta/A/D retained. Grouped diagnostic repeats common B/C kernel contractions;
record CUDA-synchronized scan seconds/group count, not native speed. Compare
generated packed384 to source y_h W_h and recurrent-only after Dskip removal,
global relative RMS<=1e-4. All144 case-sites must pass for BOTH variants.
One full1507-position FIT history/site/variant independently sequential F64,
heads[0,7,13,19,25,31,37,47], coordinate min(witness index,head rank-1), <=1e-4.
Witness uses actual mixed F32 x promoted F64; six actual witnesses required.

Reconstruct yhat=V y'. Actual FULL3072 source gate and pinned norm/output with
source BF16 feature rounding. Original full source output reconstruction<=1e-4.
Two modes: actual source denominator and reconstructed gated-yhat denominator.
Both require full source functions and are not deployable compact decoders.
Equal-case full/centered/label output RMS,domain means/worst and ungated errors.
Per-history/per-channel centering; F64 metrics with denominator clamp1e-12.

EVERYsite DEV mean<=10%,case worst<=20%,domain mean worst<=20%,centered mean<=10%.
Only a passing diagnostic earns conditional-generator study; no chatbot/quality/
native/speed admission. Choose lowest maxsite mean, then sumsite means, then
lexical layout/mode. FAIL is scoped to the tested fixed read/write maps and decoder;
does not exclude conditional functions, learned cores or other representations.

Retain both actual packed F32(T,384) histories,207,793,152B across144 case-sites,
head/coordinate offsets in frozen basis records. One held family1800s/reserve90s,
OSworker+launcher6GiB,GPU4allocated/5reservedGiB,output256MiB/log4MiB. Torch6/
inter-op1/workerCPU0..5/NumPy1/launcherCPU11, deterministic/TF32off/CUBLAS :4096:8.
Principal runtime bytes (not full DLL tree), all operands/protocol/code bound
pre/post; held PID/creation/exit/OS through exit/GPU/time, first faults retained.
No owned overlap; exact publisher exception. Observation timeout is not terminal.
No unrecorded retry/overwrite. Reuse immutable channel launcher and mixed scan.

```text
original_falcon_balanced_heads.py --bind --out <binding>
original_falcon_balanced_heads.py --launch --binding <binding> --binding-sha <SHA> --freeze <commit> --directory <new namespace> --out <new result>
```

Python312 -I -S -B -X utf8. New worker/protocol/binding frozen before observations.
Mixed nonlinear generator W^T SiLU(conv(...)) and full gate/readout replacements
remain the real conversion problem; extra stored ternary functions must solve
them with bounded selected work and measured CPU routing/DRAM.
