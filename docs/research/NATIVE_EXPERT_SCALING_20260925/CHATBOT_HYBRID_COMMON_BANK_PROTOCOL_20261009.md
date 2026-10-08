# NEW controlled bank operands: common C-state backend probe

9 October2026. Freeze before first NEW point-input bank observation. Original
whole C prefix gate remains FAIL19/32, exact certificates verified; all32 winners
equal. This is a directed numerical diagnostic, not a repeated source/local1%
fidelity search or new quality criterion. No whole C/Python prefix replay.

Read the actual packed model (no master re-quantization) and ALL3132 C-trace FF
input vectors, paired with C FF outputs/72 scores/top8 IDs/masses. Evaluate the
stateless CUDA F32 bank function on EXACT same inputs: decoded pair-code weights,
saved scales/router; AQ63 reciprocal/nearest-even/clip, exact integer-dot range,
SiLU gate/signed up/new down AQ, stable top8 selected softmax, ascending-expert
index_add. No SSM/SWA/source model/embedding/head/teacher or learning call.
Different common operands are the new variable; these are not the original
GPU own states. CUDA float router and nonlinear/quantizer arithmetic now isolated
from upstream C-vs-GPU evolving-state differences.

All12 sites/261 inputs each, no favorable subset. Record per-frame relative FF
output RMS (exact zeros only allowed with equal zero), complete router IDs and
score differences, per-layer same-ID mass errors; retain complete NEW F32 FF
vectors. Classify RMS>1e-4 as a material stateless discrepancy. Any router ID
change ->COMMON_INPUT_ROUTER_DIFFERENCE. Otherwise material same-ID bank errors
->COMMON_INPUT_BANK_DIFFERENCE. Otherwise ->COMMON_INPUT_BANKS_CLOSE and diagnose
upstream core/normalization/evolving state. No change to original whole gate.

Interpretation limits: a same-input bank difference is a contributing backend
mechanism, not a proof that it explains all original whole errors. A close probe
does not exclude threshold crossings after tiny upstream perturbations. Source
model capacity and quality remain separate. Quantization is discontinuous:
q=round(63*x/absmax(x)); a perturbation can cross a half-integer boundary even
when its floating norm is small. This motivates a hypothesis to test, not a
posthoc relaxation or an already qualified robust-margin training recipe.

Local isolated Torch2.6.0+cu124/NumPy2.4.6, TF32 off, CPU6 threads, no children.
Hard600s FAMILY/4GiB OS/GPU allocated512MiB/reserved1GiB/output16MiB/log4MiB.
Held worker through actual exit/peak, bound inputs before/after. About6.4MB new
FF vectors plus frame metrics. Preserve first faults and completed layer outputs;
any repair must reuse completed point observations. No T4/long fitting here.
