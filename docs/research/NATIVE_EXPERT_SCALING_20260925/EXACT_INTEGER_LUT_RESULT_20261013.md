# Exact engine-style byte LUT: arithmetic admitted, acceleration rejected

13 October2026. **COMPLETE / INDEPENDENT_STORED_PASS / COST FAILURE.**
[Protocol](EXACT_INTEGER_LUT_PROTOCOL_20261013.md),
[producer](exact_integer_lut_result_20261013.json),
[complete independent audit](exact_integer_lut_adjudication_20261013.json),
[cost protocol](EXACT_INTEGER_LUT_COST_PROTOCOL_20261013.md),
[360 observations](exact_integer_lut_cost_result_20261013.json),
[full ledger/campaign custody](exact_integer_lut_cost_review_20261013.json).

## Exact adaptation of the original primitive

Physically use the unchanged donor_adaptation/nibble_pack.h
matvec_lut_full_n and literal acc_add_i8x32 from phase60/engine.c,with empty
OMP_PFOR for one CPU thread. Six balanced trit weight planes reconstruct
the admitted p6 q EXACTLY. Four unsigned-low/signed-top radix16 digits
reconstruct each admitted I16 input EXACTLY. Each digit pair builds a16-entry
signed-byte table,valid two-trit codes0..8,invalid9..15 explicitly0. Low table
magnitude<=30,top<=16;no byte overflow. Current engine AQ63 and its original
single-plane tables fit<=126;direct I16 table building would overflow.

Per-trit/digit int32 partial bound46,080,weighted low16,773,120/top8,945,664.
Sign-extend before powers1/16/256/4096,int64 full dot bound36,640,321,536.
Identical F64 input scale THEN row scale/F32/BF16 rounding as admitted integer
reference. No extra quantizer/precision loss/refit. Codes[p,768,2048] use
original adjacent-REDUCTION-pair nibble packing;all fixed dimensions even/
output2048 multiple32,neutral code4. Source-only encoded packets before
responses;do not confuse this with conditional routing or new knowledge.

## Complete independent numerical evidence

All144 histories/sites0/12/23 completed. Independent auditor imports no
producer;decodes every low/high packed code to actual q,checks original
scales,full bounded-integer F64 contractions/exact I64 conversion,scale order
and independent BF16 RNE. ALL outputs BIT-EQUAL both independent integer
reference and prior complete accepted archive:

|Independent check|Count|Differences|
|---|---:|---:|
|Packed code bytes|28,311,552|0 reconstructed coefficients|
|q coefficients|18,874,368|0|
|Reused actual I16 input words|207,793,152|parent admitted|
|New I64 dots|138,528,768|0|
|New scaled F64 score words|138,528,768|0 bits|
|New BF16 words|138,528,768|0 bits|
|Exhaustive digit words|262,144|0|
|Exhaustive pair-table elements|4,194,304|0|

All65536 signed16 words,including INT16_MIN solely for bit-decomposition
control;actual histories still exclude INT16_MIN. Table controls pair each
unsigned word with its bitwise complement,all16 entries/all4 digits stored
and independently checked. Four sign-extreme controls/all2048 outputs,
one trit mutation affecting only output0 by-32767,zero-input and counted/
uncounted native parity pass;auditor checks those producer receipts rather
than rerunning controls. Full history fields independently recomputed.

Every inherited local source/fidelity gate passes through complete output
identity to the [qualified integer result](INTEGER_READOUT_RESULT_20261013.md).
This is an exact implementation of that accepted approximation,not a return
to original unquantized-source output bits. Same local source calibration;
no new OWN-history/all-layer chatbot/task/generation evidence.

## Same audited binary isolated cost

Both EXACT all-history-audited LUT and integer DLLs used. Same18 prior FIT
positions/5 paired alternating reps/core0/batch1,warm-repeat/source-flushed,
360 observations. Actual BF16 decode,F64 conversion and identical literal
Python quantizer including diagnostics/allocations INSIDE BOTH call times.
LUT also includes all table construction/heap/24 kernel passes/combine/round;
integer includes its native dot/scales/round. Every warmup/measured input/
scale/I64/raw/BF16 output bit equals full archive. No replacement binary.

Median microseconds,30 values per site/mode/method:

|Site|Warm LUT /integer|Source-flushed LUT /integer|Flushed LUT/integer|
|---:|---:|---:|---:|
|0|6859.20 /806.50|7490.25 /825.05|9.079x|
|12|7651.50 /875.30|7664.10 /842.05|9.102x|
|23|7734.30 /854.25|7667.60 /856.45|8.953x|

Full immutable ledger/duplicates/counts/samples/source bytes/statistics/
hashes reconciled independently;no retiming. LUT slower8.50--9.10x at EVERY
site/mode. Reject THIS unchanged24-path implementation as acceleration.
This does not prove every LUT is slower or that low-precision trained
experts cannot be useful. It disproves borrowing original single-plane
synthetic throughput for this accurate multi-plane source conversion.

Packed physical source9,453,568B incl scales versus integer12,599,296B,
but24 literal paths read weight planes four times. Actual native header
counter records37,748,736B of logical code-vector operands/input. Counter
disabled in ordinary/cost calls;this is not PMU/DRAM traffic. Scratch139,264B
(tables98,304+tmp8192+four totals32,768),reallocated per history call.
Smaller resident codes do not mean smaller active work/traffic.

CPUIDCLFLUSH64B/MFENCE evicts each source line incl unaligned tail;flush
duration separately recorded. Input/scratch not flushed,warm repeat not cache
residency guarantee. Shared desktop/frequency/scheduler unisolated;LUT worst
19.4869ms and integer worst1.6415ms retained. No tail bound/full-model token
rate inferred. Earlier dense/F64-LUT runs remain separate immutable evidence.

## Provenance/resources/custody

Numerical source/protocol29e5418,producer launchf588622,audit launch7f03f32,
audit admissionf7bf50d,cost/review sourcee52821f,cost launch5e9a6d9.
Binding d165e0c316ed0b5a61eca6ebf23e26aea11568c4b2acb1b75cf9c0ff320b4ed9;
producer51c039a9f30e71269c52197376625eebba1d6805648bbd464d381b7043d50d07;
audit a410c175cf66944b52a5acc277f7230e0942d691cf8d8d8ace233752cae3ec90;
cost4c0be377d5d1f03331308a99b7fea3a8781034417e689ceaef3e95d7a157769e;
review845724374ad6d6da7e8588124b70a5cfe0ba616fb4cce26e11870bc63e4e2c02.
Exact commands/runtime/compiler/ABI/protected engine/header/full original
archive extents pinned in bindings/launches;not a full transient DLL-tree proof.

Preparation14.547s/OS172,359,680B,producer599.406s/OS212,774,912B,
full audit39.75s/OS370,327,552B,cost prepare1.141s/OS40,865,792B,
cost run10.75s/OS85,139,456B. All five scientific commands terminal0,
PID+creation matches absent/protected engine/header/foreign hashes unchanged;
publishing daemons observed/untouched. No new mechanical fault in this campaign.
New ordered E exact_integer_lut_20261013:147 run/control NPZ2,498,349,442B,
with three packed weight packets and DLL/log2,526,781,296B,<3GiB. Cost E
family empty marker,existing binaries/data reused;observations in research docs.
All recorded family extents rehashed in closure. No GPU/T4/source model
forward/recurrence/optimizer/RESERVED/new donor allocation.

## Decision and return to the full pipeline

Keep PMADDWD integer shared readout as the available lower-cost accurate
primitive;retain byte-LUT exactness as reference,freeze its performance failure.
Do not keep varying this component. Next
[token-source generator and causal composition](TOKEN_GENERATOR_COMPILER_NEXT_20261013.md)
uses a structural finite-input opportunity to connect source functions to
the trained native state,then the accepted readout. Initial-layer scope
explicit,later23 contextual generators still open. A faster/readout-compatible
component is insufficient for source CHATBOT ->compact shared SSM/SWA +useful
selective ternary capacity ->engine.c. Need complete own histories/task/
generation quality,actual IDs AND masses/DRAM,useful n atRAM-driven scales,
multiple families/conversion cost/O(n) master and gradient resolution and
SAME useful>=50 token/s. Goal ACTIVE/INCOMPLETE;no owned scientific job live.
