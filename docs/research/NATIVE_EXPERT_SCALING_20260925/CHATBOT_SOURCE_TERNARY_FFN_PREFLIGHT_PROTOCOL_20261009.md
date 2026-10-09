# Source-width ternary FFN: first fixed cost/arithmetic control

9 October2026. Frozen before model observations. New FFN arithmetic variable,
not a replay of broad24 or a generic native donor runtime. No training or T4.

## Uncertainty and decision

Can the pretrained source's complete FFNs be converted into our deployable
ternary/AQ63 primitives at feasible offline memory/time, and what whole-output
loss does that first conversion cause before width/depth/selection changes?
Reuse pinned Falcon1.5B/source package, qualified donor SSD storage helper,
all source identity/tokenizer/roles, adopted broad8808 labels. Select shortest
and longest FIT teacher-forcing length by metadata/ID:the existing58/1507 cases.
Measure both forced cached trajectories against their saved donor packets.
There is no regenerated source response, optimizer step or cropped prefix.
If resource/arithmetic passes, freeze a full48 control/own-history criteria
using this actual price and the exact packed FFN payload. Quality failure here
directs recovery at this stage;it does not admit compressed-target quality.

## Exact mapping and approximations

Actual source FFN is
`down(up(x)*silu(gate(x)*gate_multiplier))*down_multiplier`,
width2048/hidden4608/biases absent;24 blocks. Source multipliers are
0.4419417382415922 and0.13020833333333331,applied at those ORIGINAL positions.
Keep original BF16 organs/attention/SSM/readout,24-block parallel composition,
all4608 FFN neurons. Only FFN arithmetic changes:

1. Promote original BF16 FFN weights to F32; fit row scales with the EXISTING
   eight-round least-squares `row_scale` routine (floor1e-8).
2. Trit=F32 round(weight/scale),clip[-1,1]. Quantize activation with ORIGINAL
   F32 absmax/63 andnearest-even round/clamp[-63,63],no training dither.
3. Integer-valued F32 dot,then multiply row-scale andactivation-scale;gate
   multiplier,SiLU andgate/up product remain F32 inside the converted FFN.
   Down projection repeats AQ63. Apply down multiplier;cast final output BF16.

Weight/activation quantization andchanged internal FFN cast boundaries are an
APPROXIMATION package,not an exact BF16 rewrite or separately isolated effects.
No source MUP scalar is silently folded across a source cast. Retained external
organs receive the original dtype. Keep source files immutable.

Actual FFN coefficients3*24*2048*4608=679,477,248. Store int8 codes/scales for
offline Torch;replace FFN modules anddiscard their original BF16 master tensors
from the runtime. The remaining339 named floating source parameters total
875,386,240 elements. This is full dense coverage,not sparse capacity admission.

Export all72 projections to one actual packed FFN payload using the SAME
engine pair encoding:first digit*3+second digit,pair/row tile-major byte0..8;
row scales little-endianF32. Decode ALL codes before writing,retain offset/
shape/SHA metadata. Expected payload339,738,624 pair bytes+1,081,344 scale bytes
=340,819,968B. It is a converted FFN sector,not a complete native model.

## Arithmetic and observations

From block0,save the last prompt row's actual quantized input/int8 matrix/
F32 dot for gate/up/down on EACH ofthe two trajectories. Verify all output
coordinates independently against F64 CPU dot using these integer operands.
Maximum integer accumulation63*4608=290304<2^24;exact equality required for
all6 dots. This qualifies actual integer contractions,not all nonlinear/cast
or whole C equivalence. Packed pair roundtrip for ALL679,477,248 trits must pass.
All72 scales/codes/output rows finite/range-valid;original source non-FFN
parameters preserve object identity throughout replacement.

Persist changed full-vocabulary BF16 logits on the ORIGINAL donor-forced
prefix/cache trajectory,case/label KL in F64,source-greedy disagreement,
stop/EOS/partial provenance,timing andactual memory. No changed-output quality
threshold selects whether to continue the two-case price;always report both.
This FIT-only preflight is not fresh preservation/usefulness evidence. The
existing14/16 source screen and8808 packets remain unchanged/reused.

## Resources and stops

Family<=600s,reserve45s,OS<=8GiB,CUDAallocated<=10GiB/reserved<=11GiB,
output<=512MiB/log4MiB,six worker cores/no overlapping model/compiler work.
Costs include once-only row calibration/export andboth cached trajectories;
separate their timers. Save every completed step/case before later guards.
Stop/preserve first fault;caps fixed. No optional-kernel install/native source
port/source regeneration/dataset acquisition/RESERVED query.
