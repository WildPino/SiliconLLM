# Proposal: preserve parent weights and encode conditional corrections separately

**Status: combined source-fixture C operator implemented/pass;learned
fixed rank32 count comparison implemented/rejected.** METH-245 source/numeric controls
pass but four-team cost10.390ms fails. METH-246 conserves all384 vectors
bitwise and passes9.637ms with one team. METH-247 separately freezes
actual raw-solution factorization:all176 functions distinct,fit gain11.72%,
but consumed E160 loses2.26% to E16. METH-248 raw readouts lose27.78% too;
change continuous hierarchy before codec-only count work. Original prerequisite
proposal below is retained. METH-244 replays176 fixed fits:continuous FP64
E160 has49.53% lower fit SSE than continuous E16,but requantization erases
the gain. METH-242/243 intercept changes fail. No raw held-out/full quality
gain is known. The new variable is coefficient representation;retain the
fixed raw solutions before considering any new training recipe.

## Representation and shape arithmetic

Retain the actual fitted parent readout. Child output adds a separately
encoded correction over the same4864 nonlinear source features:

`y=mixed_parent(phi)+left_BF16*(right_BF16*phi)+child_bias_delta`.

Start with fixed rank32,right32x4864,left896x32,BF16 storage and FP32
accumulation. Decode factors in the C kernel;do not price predecoded FP32
as BF16. Parent gate/up,513-entry LUT and all mixed readout bytes remain
unchanged. Factors approximate raw-child-minus-stored-parent coefficients.
Truncation and BF16 factor rounding need actual function measurements;
low rank alone cannot establish preservation. METH-180's original Giga
weight-energy failure does not answer this conditional-residual question.

Factor payload `(4864+896)*32*2*24`=**8,847,360bytes** across24 layers;
separate bias adds86,016bytes. METH-238's ideal whole selected ledger
544,804,868 becomes **553,738,244bytes**,below560MB. These are addressed
shape bytes,not a complete export/DRAM trace/rate. Two additional matrix
reductions and dispatch have real CPU cost. Parent residency remains;
160/1600 such child corrections add1,429,340,160/14,293,401,600bytes.
No corresponding bank or learned capacity is yet evidenced.

## Next exact prerequisite: METH-245 combined native operator

Implement/freeze the actual mixed full-source plus rank32 BF16 residual
C operator,serialization and separate GPU oracle before learned-bank
factorization. Use all24 bound source layers/METH-125 states;nonzero
source-derived factors of original FP32 down-minus-decoded mixed down
exercise the added operator. Declare fixed construction and reduction/
layout before execution. These source fixtures price the operator,not
trained conditional quality or extra n.

Require input/table/base-readout bytes unchanged,every factor readback
exact,C/GPU median<=1e-4/max<=5e-4 relative L2,source-function SSE/energy<=.01,
and six-thread24-layer median<=10ms over three fixed passes. Freeze local
resources/stop before running. Native failure closes this fixed rank32
operator before learned factorization;no retiming/rank retry.

Pass licenses a separately frozen fit-bound factorization of actual raw
solutions with mean/intercept conservation,actual encoded E16/E160/rotated/
prior gates and distinctness checks. Activity-weighted factor selection,
source information in repeated-input cells,factor rounding and conversion
cost still need specification in that later protocol. Do not force
distinctness with noise or count copies. Raw fit gain is insufficient:
consumed function validation and then fresh full-model prediction,
generation/tasks and >=50 acceptedtok/s must qualify the stored artifact.

## Larger-donor boundary

Frozen GigaChat assets,source/BF16/Q4 quality and partial C fidelity are
reusable. Routed projection shapes are1280x1536 or1536x1280,64 experts/
top4/shared path;nonlinear maps differ by expert. This dense Qwen common
feature correction needs a separately priced sparse feature-sharing/
precision variant. Preserve closed fidelity/rank cells and paused generic
port. Actual10B/100B capacity,CPU route/LUT/DRAM and quality remain unproven.
