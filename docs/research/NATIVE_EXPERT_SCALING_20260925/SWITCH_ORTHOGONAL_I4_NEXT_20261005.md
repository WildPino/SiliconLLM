# Proposed NEW451: orthogonal input bases before fixed block64-I4

**5 October 2026; proposal only, no451 source/protocol/numerical outcome.**
Freeze ONE new source/math/controller/protocol before import/transform/forward.
Current full objective unchanged: useful conditional capacity growing with RAM,
CPU LUT/routing/real DRAM, original-relative whole quality AND>=50 acceptedIDs/s
on SAME artifact, reusable method across families/scales/actual~100B.

## Why this variable now

446 reference0/Frobenius factors fail storage/function gates.447row and448block64
I4 retain all coefficients with half/about56% nominal source bytes and small
local KL, but fail argmax gates.450 ALL8apparatusPASS proves4/4row and9/9block
events cross the selected head decision plane from state displacement even in
smooth F64 readout. Reducing average matrix/KL error alone did not preserve
all directional margins. No sole head-quantizer repair or scalar-precision grid.

NEW variable is input basis, keeping448's block64/scalar precision. This can
spread large coefficient magnitudes before rounding without requiring per-expert
exposure/fit. Whether it reduces the relevant native state error is UNKNOWN.
Do not select basis/seed/encoding from the consumed validation examples.

## Primary evidence and what is adopted

[QuaRot v2](https://arxiv.org/html/2404.00456v2),29Oct2024, demonstrates orthogonal
processing of LLM weights/activations and an online Hadamard transform before
the FFN down projection. Its full Transformer/GPU configuration and quality are
not this source's CPU/A16/one-bank configuration. We adopt the motivation and
local linear-map invariance, not reported accuracy/speed.

[QuIP# v2](https://arxiv.org/html/2402.04396v2),4Jun2024, uses randomized Hadamard
incoherence processing together with lattice vector quantization and fine-tuning.
The proposed scalar-I4/unfitted pilot does not reproduce that complete method.

[SpinQuant v4](https://arxiv.org/abs/2405.16406v4),20Feb2025, reports that different
rotations can yield different quantized quality and learns rotations. One fixed
unfitted transform here may fail; no seed search or transfer of those metrics.
Primary sources retrieved5Oct2026. No external result proves our conditional
expert usefulness, native primal, actual DRAM or SAMEartifact50.

## Exact-real algebra, preserving private nonlinearity

For column-vector convention, original f_e(x)=WO_e ReLU(WI_e x). For orthogonal
Q_D and Q_M define

    x'=Q_D x;  WI'_e=WI_e Q_D^T
    a_e=ReLU(WI'_e x')
    a'_e=Q_M a_e; WO'_e=WO_e Q_M^T
    WO'_e a'_e=WO_e ReLU(WI_e x).

This is EXACT in real arithmetic before coefficient/activation quantization.
The ReLU stays in the original neuron coordinates BEFORE Q_M. Rotating gates
or moving Q_M through ReLU would invalidate this identity. Original source
input/router/probability/core/output/readout stay in their original coordinates.
For k>1, Q_D input can be shared; each private activation needs its own Q_M.

Proposal at this geometry: Q_D=diag(H256,H256,H256)S_D,
Q_M=diag(H1024,H1024,H1024)S_M, normalized Walsh matrices and fixed signs.
These are block orthogonal matrices, not a full Hadamard768/3072 or a guaranteed
globally incoherent basis. Select ONE fixed deterministic sign specification
prospectively, retain all signs, no seed/grid/clipping/calibration/learning.

## Conversion can have an exact integer witness

Original represented source rows are q8*s8. Compute sign changes and unnormalized
Walsh transform of I8 q8 in I32, then multiply F64(s8) by integer coefficients
and divide by16(WI) or32(WO). Thus no intermediate requantization to I8 is needed.
First qualify actual source code range within[-127,127]; otherwise stop.
Max forward absolute integer<=127*256 or127*1024. Unnormalized forward/inverse
roundtrip gives N*q8 exactly; conservative N^2*127 at N1024=133,169,152<2^31.
With represented F32 s8, integer-products mantissas fit F64 at this geometry.
These are algebraic witnesses to implement/qualify on all256 real matrices,
not completed native/runtime measurements. General shapes need their own bounds.

After that transform, encode SAME signed-7..7/two-nibble64-column-block I4 with
F32 block scales and448's exact integer-block/ascending-F64 arithmetic. Keep all
coefficients/IDs. Nominal bank remains339,738,624B plus declared sign/header
metadata, not lower from rank truncation. No native equality is inferred from
the exact-real identity: transformed inputs have new rounding/A16 codes.

## Prospective staged qualification before any C work

NEW451 protocol must freeze all exact details before observation:

1. Fixed signs/order, input activation transform precision/normalization/cast,
   independent explicit Walsh matrices and tiny inverse/orthogonality/dyadic tests.
2. ALL256 original coefficient integer transform/inverse witnesses; full original
  336 native heads/FFNs exact first; unchanged448 pack/primal component bindings.
3. Rotated UNCOMPRESSED source-coefficient control at SAME original input/p:
   qualify the new activation rounding/A16 arithmetic against independent reference
   and predeclared strict source-relative KL/argmax limits BEFORE I4 conclusions.
   This control has higher stored precision and is not the compact final artifact.
4. Correct-ID rotated block64-I4, ID+1 and removed at SAME original source p,
   all336 positions. Preserve meanKL<=.01/every book<=.05/argmax<=3of336/
   identity meanKL increase>=.01/storage<=.60; all apparatus required.
5. Store full compact bank/scales/signs, source/runtime hashes and all complete
   control heads/states. Bound conversion/extra control time/output/RAM explicitly.
   CPU-only/zero fitting; FIRST failure retained before NEW numbered repair.

No new algorithm imported/executed or source weights transformed under this
proposal. Source/protocol/resource limits must be made concrete and frozen next.

## Charge the whole cost if it passes

Logical activation-transform adds/subtracts at this geometry:
D log2(256)+M log2(1024)=768*8+3072*10=36,864 per top1 function, plus sign,
normalization/casts and A16 quantization. Original two projections have2DM=
4,718,592 coefficient products. Different instructions/memory access mean these
counts are not a latency prediction. Table builders must use actual transformed
activations and all costs remain charged; no assumed cache residency.

Only if local staged qualification/gates pass, prepare NEW actual C direct-
packed versus pair-activation-LUT cost/primal gate, then whole export/fresh
held-out/generation/tasks/SAMEartifact accepted rate and hardware DRAM. All-bank
transformation changes upstream states/routes: captured original p/routes are
local control inputs, not valid deployment shortcuts.

Useful n increments/second family/actual~100B require real weights/reference/
capacity evidence; neither literature70B nor nominal bit accounting supplies it.
If the fixed basis also fails, retain and reassess conditional information metric/
exposure before any fit or seed/precision sweep.447/448 remain CLOSED.
