# NEW456 proposal: exact WI coefficients grouped by output coordinates

5 October 2026.455 valid diagnosis selects this ONE physical inquiry. No456
source/controller/compile/observation yet.454 failed recipes stay closed. Goal
ACTIVE/INCOMPLETE. No symmetric-LUT/paired-WO/native-WI sweep in this inquiry.

## Evidence and uncertainty

455 direct WI coupled decode/integer-block-reduction/F64 scales is188.9428us,
80.1103% of direct FFN. Original WI120.1746us; sparse WO saves roughly69us, which
direct WI's roughly69us overhead offsets. All3360 states BYTE exact and all
perturbation controls pass. This selects a different physical layout, not lower
precision or a claim about DRAM. LUT build/application are160.3541/522.4023us;
no evidence that build-only reduction would fix the current LUT recipe.

The NEW uncertainty: can exact permutation of the SAME WI coefficients/scales
remove repeated scalar horizontal reductions and improve whole selected-FFN cost
without changing the represented function or453 stored budget?

## Fixed mathematical invariant to realize in prospective456 source

For each original output row r and block b=0..11:

    z_r = F32(alpha * sum_b_in_order F64(s_rb) * I32(dot(c_rb,q_b)))

All64 coefficients c are the SAME signed I4[-7,7] values and q are SAME A16 codes.
Every block sum is exact; bound64*7*32767=14,679,616. EACH row's F64 block
contributions must be added in ascending b order; alpha multiplied last. No FMA,
changed grouping of floating contributions or new coefficient rounding.

Proposed ONE tile width16 output rows. Permute packed coefficients to:

    [output_tile=192, block=12, input_pair=32, output_in_tile=16]

Each coefficient-pair plane contains16 packed bytes and every tile/block is512B.
Store F32 scales as `[tile,block,output]`, SAME147,456B. Full packed WI remains
1,179,648B, no expanded I8 deployment bank. Original signs and native WO columns/
scales unchanged; stride/header/size/alignment prospectively fixed. Inverse ALL128
experts/all coefficients/scales byte-exact existing454. Padding unnecessary because
M3072 is divisible by16. No shared/deduplicated/rounded function substitution.

At each pair, decode16 packed bytes to two sets of8 outputs. Interleave each
output's two signed I16 coefficients; broadcast the SAME two A16 input codes;
AVX2 pair multiply-add produces one I32 value per output. Sum32 pair products in
I32 vectors. Convert exact block sums to F64 in groups of4 output coordinates,
multiply by their original scale and add to each row's F64 accumulator. Iterate
12 blocks in ascending order then multiply alpha/cast. This avoids per-row
horizontal reduction by accumulating across output coordinates.

The permutation introduces coefficient-plane loads and scale access behavior
that must be measured. It does not prove faster cache use, DRAM traffic or cycles.
Keep SAME basis/A16/ReLU/sparse-WO functions and full selected-FFN timer charges.

## Before first compile/import/numeric observation

Implement456 C/controller and protocol, then freeze/commit. Reuse455/455-R1 and
454 raw/archives/compiler/runtime/original source with fresh full hashes. Future
command tracking must copy immutable argv, addressing retained455 metadata fault.

Qualify all valid225 signed coefficient pairs crossed with9 A16 extrema/zero
inputs for16 different output lanes; unused--8 faults detected before lookup.
Use independent scalar expected values and qualified original454 byte controls.
Qualify transformed input/raw WI/down/codes/scales and correct/ID+1 states BEFORE
timing; source baseline remains UNCHANGED original374/454 `mv`, never a slower
split surrogate. Old direct454 may be a diagnostic comparator, not a candidate
rescue. No profiler timers in the new cost kernel.

Freeze actual full336-trace rotating-arm warmup/repetition order and resource
stops. Recommended fixed cost gates SAME454: mean<=.80 original, every book<=1.00
original, p95<=1.00 original. They must be stated before observations and NEVER
weakened afterwards. Also preserve453 nominal stored budget and new mapped bytes.

Decision: if qualification or cost fails, close this fixed tile16 recipe; no tile
size/autotuning/scale sweep. Use failure to reassess transfer economics. If all
pass, prepare all-bank source geometry/sparsity/export and then whole new routes,
fresh donor-relative quality and SAME accepted rate. New candidate cost is not
whole-model speed/quality or useful-n proof.

## Scope and stopping point

Current next action: make this proposal concrete as NEW456 frozen C/controller/
protocol; no456 scientific execution exists yet. No new GPU/model acquisition,
optimization updates or engine changes authorized by this local cost inquiry.
Do not inherit391 three-worker/context rate, or455 selected-FFN fraction as full f.
Keep ultimate useful-n/router mass/actual DRAM/cross-family/~100B scope intact.
