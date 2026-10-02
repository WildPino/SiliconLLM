# Full-feature source coefficient dictionaries and shared input tables

**Status: unmeasured hypothesis,not a frozen experiment or conversion recipe.**
297 closes the fixed native private128 profile;298 rejects global linear
rank192 factors on real GigaChat activations;299 rejects direct nonlinear
32-channel omission at half width. Preserve all source feature channels
and instead test sharing repeated coefficient products across expert banks.

## Intended transformation and distinction from earlier methods

For one source projection W[out,in],with even input width,retain every
original row and input channel. Encode each consecutive TWO normalized
coefficients with a U8 palette index;store a positive row scale explicitly.
A small,non-Cartesian palette of two-dimensional real vectors is shared
across macro experts for an organ/layer. This differs from global matrix
rank reduction,source channel pruning,scalar Q4/IQ2 and old Q15 coding of
already low-rank child-B factors. No result from those formats transfers.

For input x,build `table[pair,index] = palette[index,0]*x[2*pair] +
palette[index,1]*x[2*pair+1]`. A selected row sums its encoded table values
and applies its archived scale. Table size depends on source input width
and palette size,not number of macro experts. Codes/scales occupy RAM
proportional to expert count;actual selected reads must be charged. Down
inputs differ per selected expert,so their tables cannot be shared as if
all four experts saw the same post-SwiGLU vector.

Candidate palettes must be learned or calibrated from pinned source weights
and permitted calibration inputs BEFORE independent quality data. A
Cartesian scalar codebook is a fixed matched control. Optimizing a shared
palette against ACTUAL projection outputs can incorporate cross-channel
input effects;do not claim a diagonal weight-distance fit does so.
Specify normalization,initialization,codes,optimization,precision and
zero/outlier handling in a new frozen protocol before observations.

This preserves matrix rank potential and the source nonlinear feature
inventory. It does NOT preserve exact BF16 coefficient values. Lower
coefficient error,local projection fidelity or BF16 emulation cannot
establish composed SwiGLU/router/full-model/native quality.

## GigaChat accounting example (calculation only)

For225 two-weight palette entries,source input widths1536(gate/up) and
1280(down),there are768/640 input pairs. Every1280 source nonlinear
channel remains in this hypothesis.

| Quantity | Source-derived hypothetical value |
| --- | ---: |
| U8 codes/expert across gate/up/down | 2,949,120bytes |
| FP32 row scales/expert (4096 rows) | 16,384bytes |
| Top4 codes+scales over25 routed layers | 296,550,400bytes/token |
| Three shared FP32 palettes/layer | 5,400bytes |
| All25 layer palettes stored | 135,000bytes |
| Gate/up query table entries/layer | 345,600 |
| Four DIFFERENT down query table entries/layer | 576,000 |
| All six query tables written/layer,FP32 | 3,686,400bytes |
| All query table writes/token | 92,160,000bytes |
| Encoded row lookup/add count/token | 294,912,000 |

The lookup count equals the number of encoded pair indices,not original
matrix multiply FLOPs. It may bind CPU/cache bandwidth and dispatch even
if source-weight DRAM payload shrinks. Palettes are small enough to make
cache reuse a hypothesis,but do not assume measured residency. Query table
writes,random row gathers,scales,activation/LUT conversion and down-table
regeneration need actual C measurements. No rate is inferred here.

Compared with actual Q4 routed356,106,240bytes/token,this optimistic
codes/scales-only ledger saves59,555,840bytes/token;keeping other organs
unchanged still leaves956,638,304 addressed bytes/token BEFORE palette,
table/state/KV/routing costs. Therefore a routed-only implementation cannot
qualify the50tok/s architecture. A coherent MLA/head/shared-core treatment
and complete budget are prerequisites for full donor-port/export work.

## Next exact research action and promotion boundary

First build a source/header-only preflight for this representation on the
fixed GigaChat organ inventory:price codebooks,codes/scales,all per-query
table work and compulsory reads,including MLA/shared/dense/head. Determine
which input tables can really be reused and identify a costed complete
path before launching palette optimization. If no plausible complete path
fits20ms,do not train this as a final candidate;retain the useful analysis.

If licensed by that cost screen,freeze a small source-bound palette/output
fidelity experiment using298 captures and exact raw tensor hashes. Compare
learned versus Cartesian scalar control on a distinct calibration domain;
then validate composed nonlinear/source routing and independent whole
quality before native integration and same-artifact accepted-rate tests.
Thresholds,budget and stop conditions remain to be fixed;no training result
or artifact is claimed by this proposal.

Existing GigaChat64 pretrained expert parameters are real capacity. Encoding
them does not add new experts. Wider banks require genuinely different
pretrained/trained functions,source exposure/usefulness checks and measured
CPU routing/DRAM costs. Synthetic/copy expansion can never establish the
user's useful RAM-scale n or10B-to100B transfer requirement.
