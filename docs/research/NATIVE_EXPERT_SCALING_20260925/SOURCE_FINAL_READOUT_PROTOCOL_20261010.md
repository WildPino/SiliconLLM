# Source final readout qualification from retained histories

10 October2026. PROSPECTIVE, UNEXECUTED. No codec fit or history learner.
Prerequisite for the [paired output investigation](PAIRED_OUTPUT_CODEC_NEXT_20261010.md).

## New uncertainty and reusable evidence

The actual51 heads cannot read h24P. The previous Adam1 mathematical readout
probe already found feasible low-loss features on72 FIT labels; do not replay
that oracle or infer a width ceiling. First establish that the saved full h24
can reproduce the source labels, which were captured during cached generation.
The full-prefill boundary capture and cached-generation labels have not yet
been qualified together through the final head. This is an alignment and
operator prerequisite, not another compression or final-head scale sweep.

Reuse ALL48 raw h24 BF16 histories/22547 positions and ALL8808 retained labels
(4422 FIT/4386 DEV), complete65537 vocabulary; use only the already saved label
positions. Read source final normalization and LM head from the pinned BF16
safetensors. Source hidden2048, norm epsilon1e-5, output multiplier0.01953125.
Bind actual source implementation/config/weights, parent boundary qualification,
current attribution result/audit, consumed raw arrays, code and principal runtime.
Source parameter values may be read; no model/sequence/history/generation call.

## Two fixed readouts, no fitting

1. Reconstruct the source norm exactly as its inspected code: cast raw h24 to
   F32, compute squared mean/rsqrt, multiply h in F32, cast normalized h to BF16,
   multiply BF16 gamma in BF16. Save these full2048 post-norm features at labels.
   Apply the BF16 LM head one label at a time on the RTX3060, then its scalar
   multiplier in BF16. This matches the cached head's row shape, not necessarily
   the cached *history state*. Save every score losslessly as BF16.
2. Apply the same saved BF16 post-norm feature and BF16 source head in F64 on
   CPU, with the exact scalar multiplier. Save every score losslessly in F64.
   This isolates head/output rounding on the same normalized features; it is
   not a hypothetical F64 normalization or an optimized compact decoder.

Report each split/case/domain separately: full-V KL, argmax disagreement,
teacher entropy/uniform KL and BF16-versus-F64 readout differences. Compare
the stable max-subtract sum-exp metrics with independent logaddexp/dot F64
reductions; maximum per-label absolute discrepancy<=1e-10. No label subset,
map selection, decoder fit, original-engine call or RESERVED/T4.

Frozen source-label alignment gate, separately for EACH readout and split:
case-weighted mean KL<=0.01 and disagreement<=0.01; EVERY case KL<=0.05 and
disagreement<=0.05. Also report all domain means under the same0.05 screens.
This is a stricter prerequisite than eventual converted-model KL1/dis20%; it
does not assert bitwise equality between cached and full-prefill histories.
Scientific gate FAIL must still retain/finish all cases and select an alignment
diagnosis rather than abort or relax the gate. No codec fit follows a failed gate.

Independent stored audit recomputes metrics/decisions from both saved score
formats. It also checks saved BF16 normalized-feature relative RMS against the
analytic F64 source norm<=0.01 per case, and first/middle/last label scalar
head witnesses at vocabulary0/1/65536/teacher argmax by math.fsum, absolute
F64 score error<=1e-8. These qualify specified operators/numerics, not whole
source inference, a compact target, causal histories or new chatbot responses.

## Finite measured cost and custody

8808 new BF16 source-head calls and8808 new F64 head contractions are counted
explicitly. Source history/model/generation calls0; optimizer/native calls0.
Projected full-V storage: BF16 scores1,154,499,792B, F64 scores4,617,999,168B,
BF16 features36,077,568B, combined5,808,576,528B plus summaries. Expected3–8min
plus IO; held1200s/OS6GiB/GPU allocated2GiB/reserved3GiB/output8GiB/log8MiB.
CPU six threads, BLAS six; separate held stored audit900s/OS6GiB/log8MiB/out2MiB,
BLAS one, no GPU or new head contractions. Actual costs override forecasts.
Binding preparation/commits and the independent audit are outside the held
readout family; never count these operations as accepted-token throughput.

Hold worker through Win32 exit, bind inputs before/after, seal all outputs;
bind Python DLL, Torch CPU/CUDA/principal cuBLAS, NumPy/BLAS/psutil and Windows
peak-memory helper. Full old-runtime gaps remain explicit. Preserve foreign
publisher; exact daemon exemption only, no owned benchmark overlap. Stop on
resource deadline/caps, nonfinite data, extent/hash mismatch, unexpected child
or numerical audit failure. Preserve first fault and durable cases; any missing
continuation must use a NEW bound namespace and zero completed-case replay.

If BOTH readouts pass source-label alignment, proceed to a FIT-only paired
compact output representation with explicit original RMS compatibility and
unchanged DEV gates. Otherwise diagnose cached/full-prefill or readout alignment
before any compression fit. Neither outcome changes the full goal or admits
quality/speed/useful n/DRAM/family generality on the original engine.
