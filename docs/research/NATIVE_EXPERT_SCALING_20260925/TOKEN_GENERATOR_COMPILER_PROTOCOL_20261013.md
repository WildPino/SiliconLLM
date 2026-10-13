# Frozen complete source-only token generator CPU compiler

Full65537 vocabulary rows,including65536,canonical source embedding65537x2048,
layer0 input norm2048,in_proj6704x2048,conv/bias/delta/config/effective BF16 MuP.
No table row copied from observed response or trained on FIT/DEV. Build ALL
rows before ANY response comparison. Initial eval/no-padding token-ID source
graph/token witness22547rows/3816IDs only;later23 generators contextual.

Norm uses literal source CPU Torch operations:BF16 embedding*Python config
multiplier,F32 square/mean/rsqrt(eps1e-5),F32 multiply,cast BF16 BEFORE BF16
gamma multiplication,then BF16*SSMin .625. Explicit new CPU backend,not
assumed CUDA reduction identity. Freeze CPU6 Torch threads/interops1/no GPU.
Scalar-control every BF16 value with abs<=4:literal Torch Python scalar
multiply versus declared NumPy F32/config and BF16-effective scalar variants.
Archive all values/outputs and differences;no variant chosen using responses.

Decode H0 BF16 toF64,in_proj F64 contraction in256-row vocabulary blocks,
F32 cast/BF16 nearest-even boundary,decode then effective BF16 MuP multiplication
and BF16 boundary. Same promoted generator function equation as accepted
actual-H replay;inputs now computed from token. Store row-major raw H0.BF16
65537x2048=268,439,552B and F0.BF1665537x6704=878,720,096B. F0 rows13,408B,
IDs uint32. Source-only conv parameter NPZ/scalar control/small receipts.

Then all48 histories using only token IDs and F0 rows:causal depthwise conv4
(tap0=t-3,tap3=current),bias,BF16,stable SiLU/BF16;delta BF16 bias-add,
softplus threshold20/BF16/config clamp. Archive all five BF16 fields. Actual
captured H/fields accessed ONLY for validation after full vocabulary compile.
Zero prefix conv state. This compiler result does not yet execute native SSM
state or prove incremental/full-prefix composition;those need separate binding.

Quality policies frozen before observation:H0 relative RMS<=1e-4 EACH history
(new normalization budget,0.01%),all five source-field relative RMS<=.01 EACH
case (unchanged1% source calibration),each field incremental versus accepted
actual-H function<=.01. Centered/label metrics,all source bit differences,
FIT/DEV means/worst/failed counts retained. Full table and all48 cases/audit
complete even if quality fails. No native cost/composition admission unless
ALL numeric/quality flags pass;do not hide failure behind average scores.

Independent auditor imports no producer:actual source tensor ranges/complete
vocabulary in64-row blocks,separate literal Torch CPU norm program,independent
F64 einsum projection,manual BF16 low-bit/parity RNE,effective MuP,row indices,
causal window contraction/different stable sigmoid/softplus identities,EVERY
stored field/source/metadata/gate. Full H0/F0 BF16 bits EXACT independently;
field-reference relative<=1e-6 (original field-replay numerical threshold),
field bit differences counted,metadata abs(delta)/max(abs(reference),1)<=1e-8.
Shared Torch CPU norm kernels are a declared auditor limitation;no claim of
independent kernel implementation. Reference F64 matmul reduction bits are
not original CUDA bits. Explicit alias/dtype/shape/coverage checks/hash before
and after. Parent complete source parameter/capture continuity preserved.

Prepare/run900s/audit1200s/OS6GiB/new ordered E token_generator_compiler_20261013
<=2GiB (~1.45GB table+H0+48 five-field packets). All complete actual source
weights/config/source implementation/actual NumPy,BLAS,Python,psutil,Torch
principal binary/library files/pinned source reports/48 original H0+fields/
accepted replay packets/engine/foreign files hash before/after. This is not a
full transient DLL-tree trace. Torch CPU only;GPU never initialized,source
MODEL forwards0/optimizer0/SSM calls0/RESERVED unqueried. Source function
compiler pass1 counts separately. No T4,external resource or hours job.
Whole CHATBOT/compact causal capacity/useful-n/actual DRAM/same useful50 open.
