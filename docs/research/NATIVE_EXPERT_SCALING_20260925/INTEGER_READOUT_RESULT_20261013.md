# Same-p6 integer readout: exact arithmetic, local fidelity, measured lower cost

13 October2026. **COMPLETE / INDEPENDENT_STORED_PASS / ISOLATED COST ADVANTAGE.**
[Numerical protocol](INTEGER_READOUT_PROTOCOL_20261013.md),
[producer](integer_readout_result_20261013.json),
[complete corrected V3 audit](integer_readout_adjudication_v3_20261013.json),
[cost protocol](INTEGER_READOUT_COST_PROTOCOL_20261013.md),
[360 cost observations](integer_readout_cost_result_20261013.json),
[independent ledger and campaign closure](integer_readout_cost_review_20261013.json).

## What changed and what the evidence establishes

Preserve the qualified source-only p6 weights q in [-364,364] and F64 row
scales of O2048x3072. Quantize the actual normalized BF16 input Z using
a=max(abs(Z))/32767, r=RNE(Z/a), explicit zero rows, int16. This is a NEW
activation approximation, separately measured against the earlier accurate
unquantized-input p6 output. The deployed weight array is now int16; this C
kernel executes PMADDWD products, not ternary table lookups. Its source
weights remain exactly reconstructible from six balanced ternary planes.

Native pairs accumulate in int32 blocks of128 coordinates, then sign-extend
to int64. Absolute block bound1,526,680,064<2^31; full3072 dot bound
36,640,321,536<2^53. Neither operand is INT16_MIN. Convert exact integer dot
toF64, multiply input scale FIRST then weight row scale, F32/BF16 RNE.
Full independent audit reconstructs weights from trits, computes input RNE
with sign/floor/parity, and uses exact bounded-integer F64 contractions as
an independent integer reference. Every possible partial sum fits exactly
inF64; equality is checked word by word, not hidden behind a score tolerance.

All144 histories atsites0/12/23 were completed and audited. Counts:

|Independent check|Words|Differences|
|---|---:|---:|
|Source integer coefficients|18,874,368|0|
|Actual int16 inputs|207,793,152|0|
|Input F64 scales|67,641|0|
|Native int64 dots|138,528,768|0|
|Scaled F64 scores|138,528,768|relative error0|
|BF16 rounding of integer reference|138,528,768|0|

Independent metadata residual0; maximum activation half-step ratio
1.000000000000005 under frozen1+1e-10. Four sign-extreme, single-coordinate
mutation, zero-input controls are producer receipts; they were checked as
receipts by the auditor, not independently rerun. All-history equality is
independently recomputed. BF16 equality above is against the new integer
reference, not the original unquantized model; their differences are retained.

Every case passes original extra raw/BF16 1% gates, new incremental activation
raw0.1% gate, and all original local donor mean/worst/centered/domain gates
for both FIT and DEV. Maximum incremental activation output error0.0268492%;
maximum input RMS quantization error0.0312487%. Worst extra raw0.367436% and
BF16 output0.431668%. DEV per-case means and worst values:

|Site|Raw mean /worst %|BF16 mean /worst %|Local source BF16 mean %|
|---:|---:|---:|---:|
|0|0.290093 /0.296847|0.361976 /0.365860|7.639468|
|12|0.350761 /0.367148|0.415751 /0.430693|8.724563|
|23|0.297247 /0.311293|0.367279 /0.380777|2.511640|

The larger source errors include the existing source-informed reduced-state
component. This readout assay uses supplied actual native normalized histories;
it is not OWN-history chatbot quality or a guarantee of composition across layers.

## Same audited binary cost, including input conversion

The EXACT full-audited integer DLL was timed, not a replacement wrapper.
18 fixed prior FIT positions, five paired alternating repetitions, CPU0,
batch1, warm-repeat and explicit source-flushed modes. BF16 decoding is INSIDE
the timed region for both methods. Integer timing also includes F32->F64,
the literal Python quantizer INCLUDING its diagnostic norms/bounds/temporary
allocations, ctypes call, scales and BF16 output. Dense uses the original
qualified AVX F32 dot/BF16 path. Every warmup and measured input/scale/int64/
raw/BF16 word matches its full-audited archive; dense raw/BF16 bits also match.

Median call microseconds,30 observations per site/mode/method:

|Site|Warm integer /dense|Source-flushed integer /dense|Flushed integer/dense|
|---:|---:|---:|---:|
|0|698.90 /1602.30|823.15 /1616.65|0.5092|
|12|673.55 /1487.50|768.15 /1554.05|0.4943|
|23|730.10 /1513.65|858.40 /1587.00|0.5409|

All360 immutable observations/duplicates/sample counts/source bytes/statistics
reconciled independently, without retiming. Relative median advantage
1.85--2.29x in these conditions. Weight source12,599,296B including scales
versus dense25,165,824B. Warm means immediate repeat, not cache residency.
CLFLUSH64B/MFENCE evicts every source line; flush duration separately recorded.
Initial architectural source eviction is not PMU measurement of total DRAM
traffic. Shared desktop/frequency/scheduler are not isolated: raw extrema
include integer8.6724ms and dense9.8921ms. Do not interpret medians as a
worst-case latency bound. Earlier F64 LUT cost excluded BF16 decode and was
a different run; comparison to that run is not a simultaneous matched benchmark.

## Mechanical audit failures preserved and fully corrected

V1 completed the entire144-case audit, with zero integer/BF16 differences,
but marked numeric failure: its diagnostic activation RMS helper divided by
the reconstructed input norm instead of the original input norm. Maximum
metadata difference3.46565e-10 exceeded frozen1e-10. Preserve
[V1 adjudication](integer_readout_adjudication_20261013.json).

V2 corrected the denominator and repeated all cases, then failed the FINAL
hash check because the dependency list variable `extra` had been overwritten
by a boolean. No V2 adjudication was saved; preserve its terminal1 and
[V2 correction protocol](INTEGER_READOUT_AUDIT_V2_PROTOCOL_20261013.md).
V3 uses a distinct dependency-list name, repeats the ENTIRE audit again,
checks all hashes before/after and passes. No parameter/input/threshold/gate
changed. [V3 protocol](INTEGER_READOUT_AUDIT_V3_PROTOCOL_20261013.md).
An unexecuted V3 launch argument indexing typo was corrected in Git before
launch (042f8dd ->b224fa7); no experiment used the malformed argv.

## Provenance, resources, and closure

Implementation freeze85dda17, launchac6ceda; V3 audit freeze dd43ceb,
corrected launchb224fa7, admission/reviewer freeze e4ea3fd; cost launch3102031.
Binding SHA ecd85844c50a4eaee92b9bfc9f5309c48122b4fca912f12c78782bffcd53af95.
Producer SHA f4e3d831f83edd42eb3bd54073a89c9fe3465ea3fb8a7a693c6964707708ee95.
V3 SHA3893da830e06f320355900acd22c6675d6ce0c09dc214b82167e2c8aa8a3402a.
Cost SHA6fd57bc671a43b77f1251ed23679316af1eab8180bda7ae8986beb0d5ef0768b.
Ledger/closure SHA08edc05326f9cb5bbc47e2ebec9cd9f87c5a5e3b545939e7b0942c75aac0819a.
Exact commands/environments/parameter/runtime/compiler/data extents in frozen
bindings and launch JSON; terminal PID+creation/exit/resource receipts retained.

Prepare11.547s/OS40,480,768B; producer143.391s/OS516,304,896B;
V1 audit66.547s/OS678,875,136B; V2 audit57.813s/OS678,895,616B;
V3 audit73.765s/OS679,120,896B. Cost prepare1.312s/OS42,221,568B;
cost run8.172s/OS125,825,024B. All seven matching processes absent;
V2 terminal1 retained, other six terminal0. Protected engine/foreign file
hashes unchanged; foreign publishing daemons observed, untouched.

New ordered E family integer_readout_20261013:144 NPZ2,909,829,576B;
DLL/log with arrays2,909,894,600B under3GiB. Cost E family is an empty ordered
marker; receipts/observations in research docs, existing binaries reused.
All recorded extents rehashed in closure. No GPU/T4/source model forward/
recurrence/optimizer/RESERVED data/new donor or deployment allocation.

## Decision toward the whole pipeline

Retain this as an accurate, lower-cost shared readout option and a reference
for exact integer implementations. It preserves full output dimensions and
adds negligible measured local error; it does not create selective capacity.
Do not call it proof of a fast ternary LUT: it physically deploys int16 weights.
[Exact byte-table representation](EXACT_INTEGER_LUT_RESULT_20261013.md) now
COMPLETE/full independent auditPASS,all archived arithmetic identical,but
same audited paired cost8.50--9.10x slower than integer. Preserve original
measurements;reject that24-path implementation as acceleration,stop readout
variants. Next [source token generator/causal composition](TOKEN_GENERATOR_COMPILER_NEXT_20261013.md)
proposed/unbound,initial-layer scope explicitly separated from full model.

Whole source CHATBOT ->compact reusable causal core +useful selective ternary
functions ->engine.c still needs joined generator/FFN/attention/head/tokenizer,
all-layer own histories, held-out generation/task quality, useful RAM-driven n,
multiple source families/scales, conversion master/gradient O(n) resolution,
and SAME useful>=50 token/s. Goal ACTIVE/INCOMPLETE; no owned job live.
