# Common C operands: one complete core/norm backend comparison

9 October2026. Freeze implementation/protocol/inputs before first NEW core/norm
values. Reuse actual packed model,3132 C records and all6 case boundaries; original
native13/32 RMS passes/19 failures/all32 IDs remain unchanged. Common banks found
one material difference in3132 rows; it does not explain all full-prefix failures.

Evaluate each of12 individual SSM/SWA modules on stored C `core_input` sequences,
separately for each case, initial state zero per module/case. Preserve all within-
case recurrence/conv/attention evolution; no per-token reset. Actual learned F32
fields loaded from packed model, strict state keys/shapes, RoPE frequencies from
actual export, source-compatible target config, F32/TF32 off, SSD chunk16/eager
fallback. No teacher, full target, embedding/head, bank evaluation, fit or old
whole-prefix replay. These are controlled NEW common C operands, not old GPU states.

Separately evaluate source-compatible input/FF RMS norms on saved C operands.
Input norm uses saved residual input; FF norm uses F32(saved input+saved C core
output), never the NEW GPU core output. Compare all3132 core outputs and6264
norm vectors to stored C results. Save ALL NEW F32 outputs and row metrics, with
durable per-case receipts. Any repair must reuse completed observations rather
than repeat a completed module/case. Preserve first failure and partials.

Fixed material threshold: EACH relative vector RMS>1e-4. F64 energy/error sums,
exact-zero reference allowed only with equal-zero error; otherwise explicit fault.
Any material core row ->COMMON_INPUT_CORE_DIFFERENCE. No material core but any
material norm ->COMMON_INPUT_NORM_DIFFERENCE. Otherwise ->COMMON_INPUT_CORE_AND_NORM_CLOSE.
All12 sites/all6 sequences/all261 inputs per site, no favorable subset/early exit.
Stop this numerical diagnostic after one complete comparison.

Interpretation: material common-core errors localize arithmetic/scan ordering;
close cores/norms support investigating discontinuous bank activation quantization
under small upstream perturbations. Neither outcome causally explains every whole
logit error, qualifies long context/own-history or demonstrates source quality.
Do not loosen the old native gate. A corrected forward needs a NEW frozen artifact.

Local isolated Python3.12.10/Torch2.6.0+cu124/Transformers5.13.1/NumPy2.4.6,
no optional hub/SSM kernels, offline, CPU6 threads, no children. Hard600s FAMILY/
4GiB OS/2GiB allocated GPU/3GiB reserved/64MiB output/4MiB log. Expected19.24MB
raw vectors plus receipts/metrics. Worker held through actual exit/OS peak; all
bound inputs before/after, foreign SHA unchanged, no timing overlap/T4.
