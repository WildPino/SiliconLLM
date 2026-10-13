# OFFLINE approximate input recovery and independent generator replay

13 October2026. **ORIGINAL PROPOSAL / NOW COMPLETED.**
Superseded operationally by [full result](GATE_INPUT_RECOVERY_RESULT_20261013.md)
and [temporal propagation next](RECOVERED_INPUT_TEMPORAL_NEXT_20261013.md).
The original proposal below is retained as the pre-observation rationale.
Reuse [fully qualified gate geometry](GATE_INPUT_OPERATOR_RESULT_20261013.md),
all48 existing histories/sites0/12/23 and pinned source weights/config/runtime.
Native full3072/state96 has useful local quality with small recurrent cost;
source generator/readout and compact causal hidden functions remain the bottleneck.

## Question and fixed scope

Can the existing pre-SiLU gate measurements supply sufficiently faithful
APPROXIMATE in_proj inputs for calibration of source functions,without a new
full donor run? It is not a runtime reconstruction from a compact state:gate
itself is supplied by the donor. Source input AFTER ssm_in_multiplier and before
in_proj is the unknown variable. Do not substitute FFN input or residual hidden.
No new model weights/basis/DEV-selected reconstruction/no source response/RESERVED.

1. Verify actual nonpersistent MuP buffer dtype/values and source rounding order
   from pinned construction/load code and captured dtype witnesses. Source
   in_observer forms output*mup_vector;save asserts gateBF16. Do not silently
   replace rounded buffer values with full-precision config scalars. Also pin
   input/projection/conv/activation/delta dtype stages and source4tap orientation.
2. Fixed full-column-rank least-squares recovery from unscaled G and descaled
   captured gates,using qualified Gram/QR/SVD without a numerical ridge/floor
   chosen on DEV. Include declared BF16 quantization/reduction uncertainties.
   BF16-rounded recovered input is the fixed candidate;unrounded reconstruction
   is diagnostic only,both retained if implemented. No optimizer/refit or
   selecting whichever arm passes DEV.
3. Replay source full in_proj,MuP,causal conv4/SiLU and delta bias/softplus/clamp
   against OTHER independently captured x/B/C/delta fields on ALL48 histories
   andALL3 sites. Gate residual is necessary but circular alone. Separate CPU
   promoted exact equations/explicit rounding from actual donor CUDA reduction
   bits;they are not automatically bit-equivalent.
4. Freeze consistency/error gates BEFORE any recovery observation. Suggested
   calibration-only functional budget:each case/field relative RMS<=.01;
   all label/domain/worst statistics and finite/hash/rounding controls retained.
   This is a new approximation-data budget,not a relaxed1e-4 native arithmetic
   control or a source hidden-bit guarantee. Exact final protocol must justify
   and freeze the budget. True input error remains unknown without actual h.

If cross-operand fidelity is insufficient,retain the complete counterexample
and capture TRUE in_proj input under a new pinned donor/runtime/cost protocol.
Do not keep optimizing gate reconstruction alone or relabel inferred inputs
as ground truth. All audits continue even obvious fidelity failure.

## Why geometry alone is insufficient

Full source matrices have conditions27.28/71.26/54.79 and unscaled inverse
gains2.384/4.579/3.391. A bound||delta_h||<=||G^+||*||delta_gate||needs correct
MuP units and actual BF16/reduction error bounds. It establishes no empirical
small relative h-error and no reconstructed hidden bits. BF16 lattice/input
norm information may sharpen uncertainty only under explicitly proved constraints.
Do not call numerical injectivity of G a statement about all dynamic target
state information,output-code dimension or future causal closure.

## Resources and next transfer step

Proposal:producer900s/audit1200s,OS6GiB/newE arrays<=3GiB;no GPU/model forward/
T4 required for a promoted/rounded replay. Actual code/runtime/dependencies,
dataset/weights/recovery policy/gates/storage/launch frozen before execution.
Explicitly enumerate resolved NumPy/BLAS/psutil binaries (parent directory
prefix is insufficient for individually linked packages). Preserve prior metadata
gap and numerical failures. Hours-long work follows user's announce/launch/PAUSE.

If candidate passes CONSISTENCY ONLY,compile source-function calibration data
with approximation bounds and use it as an initializer/proxy,not exact state
supervision. Learn/derive cheap generator/readout/conditional ternary functions
in a core that creates its OWN causal history;validate actual source categorical
loss/held-out tasks/own emitted history and native equivalence before admission.
Useful n,CPU IDs+masses,actual expert DRAM,conversionO(n),multiple families/scales
and>=50 useful batch1 tokens/s on the SAME complete artifact remain required.
