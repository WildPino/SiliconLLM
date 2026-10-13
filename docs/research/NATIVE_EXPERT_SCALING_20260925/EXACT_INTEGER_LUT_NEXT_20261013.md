# Next representation question: exact integer arithmetic in engine-style byte LUTs

13 October2026. **HISTORICAL PROPOSAL, NOW COMPLETE.**
[Full exact byte-LUT result](EXACT_INTEGER_LUT_RESULT_20261013.md): complete
independent auditPASS/same integer words/ALL sampled cost comparisonsFAIL.
Original proposal retained below; next joined source compiler
[here](TOKEN_GENERATOR_COMPILER_NEXT_20261013.md) proposed/unbound.
[Full integer readout result](INTEGER_READOUT_RESULT_20261013.md) supplies an
accurate, measured faster reference. Existing p6 F64 gather LUT is slower;
PMADDWD uses int16 physical weights. The remaining representation uncertainty
is whether byte shuffles and packed two-trit codes can execute the SAME
qualified integer arithmetic at competitive cost. This addresses the original
engine primitive without assuming that its synthetic rates transfer to p6.

## Exact algebra: preserve today's accuracy boundary

Keep all existing q, input r int16, scales, and output rounding. Represent
weights q=sum(p=0..5)3^p*t_p, t_p in{-1,0,1}, exactly as audited. For each
r in[-32767,32767], let u be its unsigned16 two's-complement word and define
d0=u&15, d1=(u>>4)&15, d2=(u>>8)&15, d3=(u>>12)-16*(u>=32768).
Then r=d0+16*d1+256*d2+4096*d3 EXACTLY. Low digits0..15; top digit-8..7.
No new quantizer, clipping, response fitting, coordinate transform or loss.

For each pair of coordinates and each digit slice, build a16-entry signed
BYTE table using engine convention c=3*(t0+1)+(t1+1), valid c0..8:
L[c]=(c//3-1)*d_i+(c%3-1)*d_(i+1); entries9..15 explicitly zero.
Every low-slice value magnitude<=30; top-slice<=16, so byte lookup exact.
Actual engine.c uses the same two-trit/16-entry pshufb convention and AQ63,
with maximum single-plane table magnitude126. Passing arbitrary int16 values
directly to its byte-table builder would overflow; exact digit slicing avoids
that mechanical error. This is an extension, not unchanged engine execution.

Native partial per trit plane and digit: sum3072 coordinate products,
abs<=3072*15=46,080, within int32. Combine all six weight planes for each
low digit into int32: abs<=3072*364*15=16,773,120; top<=8,945,664.
Sign-extend FOUR totals to int64 BEFORE multiplying1/16/256/4096 and
adding. Final abs<=36,640,321,536, exactly the already qualified integer
dot bound. Then identical F64 scale order/F32/BF16 output. If int16 partial
vector accumulators are used for speed, freeze a pair-block bound first
(e.g.128 pairs*30=3840), widen every block; never silently wrap full sums.

## Code and traffic are costs, not evidence of speed

Six two-trit planes:6*1536*2048 codes; nibble packing two codes per byte
requires9,437,184B plus16,384B scales. This differs from five-trit p6
7,573,504B and integer source12,599,296B. Original nibble header packs
adjacent REDUCTION pairs in each byte, not adjacent output rows. Preserve
that layout or explicitly freeze/qualify any new one; preserve neutral code4.

Four16-entry table slices for1536 pairs require98,304B if all materialized;
32-bit four-output totals32,768B plus other scratch must be counted. A
literal six-plane/four-slice implementation executes24 single-plane paths.
Calling the existing kernel separately may read the six packed weight planes
four times (~37.75MB). Reusing each decoded code across four slices can
reduce code reads but adds registers/accumulations. Packed storage alone
does not establish traffic or latency. Exact zero-slice skipping is permitted
only if its input-dependent condition, scan cost and mathematical neutrality
are explicit and audited. No approximate selection of planes after seeing DEV.

## Proposed bounded decision sequence

1. Freeze one layout/kernel and formal byte/int32/int64 bounds; reuse actual
   engine/nibble primitives where practical. No new source/model/T4 job.
2. Exhaustive16-bit input decomposition, all pair-table entries including
   invalid codes, sign-extreme/zero/mutation controls, scalar-vector parity.
3. Replay all144 frozen actual input r/a vectors with source p6 weights.
   Require EVERY dot int64, scaled raw F64 and BF16 word BIT-EQUAL to today's
   full-audited archive; independently decode all codes/recompute the same
   exact integer reference. Keep full audits if a gate fails. No extra quality
   budget required for an exact arithmetic representation, but check the
   inherited source/quality metadata and before/after hashes.
4. Only after that gate, same fixed18 FIT samples/360 observations, actual
   BF16 decode+qualified input quantizer+tables/lookup/combine included, CPU0,
   paired warm/source-flushed comparison with SAME integer reference DLL.
   Record literal bytes, measured outputs, raw latency tails and full ledger.
   If slower, reject this implementation as acceleration; retain exact-
   representation result. Do not keep tuning a failed unchanged backend.

Proposed correctness run900s/full audit1200s/OS6GiB/new ordered E family<=3GiB
for packed coefficients plus full dots/raw/BF16 archives; reuse original
input witnesses, no duplicate input files. Cost900s/16MiB. These are proposals,
not bound execution permissions/receipts: create actual storage marker, code,
protocol/binding/launch revisions before execution. No hours-long job planned.

## Stop the component from becoming the goal

This single comparison resolves whether an exact accurate byte-LUT version
belongs among available conversion primitives. Then return to joined causal
source functions: readout alone cannot establish chatbot quality, conditional
expert usefulness or50 token/s. Remaining generator/FFN/head/attention costs
are unremoved; multiplying isolated site medians by24 is not a full-model
measurement. Source-only low-rank/global and fixed affine-bank coverage
failures remain constraints, not a universal impossibility proof. A useful
conditional compiler must carry residual information in experts, measure
actual IDs AND masses/DRAM, and pass own-history task/generation quality.
Full goal unchanged, ACTIVE/INCOMPLETE.
