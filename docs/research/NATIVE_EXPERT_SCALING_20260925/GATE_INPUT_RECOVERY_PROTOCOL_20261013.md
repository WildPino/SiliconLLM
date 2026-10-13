# Fixed OFFLINE input witness:rounded least squares and OTHER operand replay

13 October2026. Freeze before recovery observation. Reuse qualified gate
operators/source weights/all48 captured histories/sites0/12/23. P=solve(G^T G,G^T)
with the exact audited Gram;NO ridge/floor/rank/optimizer/DEV selection.
H64=(gate/effective_BF16_MuP_gate)P^T;fixed candidateQ_BF16(H64),unrounded
H64 stored diagnostic only. Input is AFTER ssm_in_multiplier/BEFORE in_proj,
not residual hidden or FFN input;do not apply input multiplier a second time.

Semantics before execution:source loading uses dtypeBF16,local_torch_dtype
construction creates compute_mup_vector bufferBF16,nonpersistent cloned per
layer;capture save asserts gateBF16. Probe exact MuP section bits using same
config helper under same dtype context;check restore/default/noCUDA initialization.
Independent CPU BF16 conv4 orientation/SiLU/softplus toy witnesses,not an original
CUDA bit-equivalence assertion. Sourcecode padding3,tap0=t-3/tap3=current,
groups3584/bias,MuP widths3072/3072/256/256/48,delta bias-addBF16/softplus1,
threshold20/clamp source limits fixed. Primary resolved CPU Torch/NumPy/BLAS/
psutil/HF constructor source binaries and weight/config/data hashes bound.

Actual replay:promotedF64 exact linear contraction on BF16 candidate/source
coefficients then explicit BF16 raw projection and MuP result;causal conv4
F64 products/bias then BF16;SiLU promoted/rounded;delta addBF16,softplus/round/
clamp. It approximates the original CUDA BF16/F32 reductions;DO NOT claim those
bits or source Hbits are recovered. True input error remains unknown.

ALL144 histories and five fields gate/x/B/C/delta,all history positions/label
positions/domain/centered/worst stats. Fixed approximation-data budget EACH
case EACH field relative RMS<=.01. Independent-other-field admission requires
x/B/C/delta all pass;gate residual alone is circular. This1% budget is a
calibration-consistency gate justified as a small functional approximation
relative to subsequent.1 output budgets,not an end-to-end guarantee or relaxed
1e-4 native arithmetic gate. No source model/hidden/causal quality admission.

Store every H64/H_BF16/predictedBF16 field and three maps. Full independent
rectangularQR inverse/recovery and shifted-window conv/dtype/activation/delta
replay;all arrays/metrics/flags/aggregates/hash checks EVEN catastrophic failure.
Numeric gates inverseleft max1e-10/independentinverse1e-10/Hrelative1e-9,
stored replay vectors1e-6/metrics1e-8 and exact integer BF16 candidate fromH64.
No successful subset,parameter changes or unknown-input labels as truth.

Budgetprepare900s/producer900s/audit1200s,OS6GiB,newE output3GiB,
CPU0..5/six BLAS threads,no GPU/T4/RESERVED/model forward/new teacher replies.
If independent fields fail,retain result and study correction or capture TRUE
input under separate protocol. If pass,data is an uncertainty-qualified
initializer/proxy only;compact causal generator/readout/ternary functions,
source categorical/own-history/model quality and same useful50 still missing.
