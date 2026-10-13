# Native token -> persistent convolution/state -> own integer readout

13 October2026. **COMPLETE / INDEPENDENT_STORED_PASS / ALL LOCAL GATES PASS.**
[Protocol](TOKEN_STATE_PROTOCOL_20261013.md),[producer](token_state_result_20261013.json),
[full audit](token_state_adjudication_20261013.json),
[single-token cost protocol](TOKEN_STATE_COST_PROTOCOL_20261013.md),
[60 observations](token_state_cost_result_20261013.json),
[full custody/ledger review](token_state_session_closure_20261013.json).

## A joined source-derived native branch

The new C API consumes actual uint32 token IDs and the
[complete source-only F0 table](TOKEN_GENERATOR_COMPILER_RESULT_20261013.md),
not captured source hidden states/gates/B/C/delta/answers. It owns a three-row
3584 BF16 convolution ring and3072x96 F32 trained SSM state. New history starts
at zero;every next token updates the SAME context. Native snapshot/last-ID
exports expose continuation/ring/count. ID65536 supported;out-of-vocabulary
65537 rejected before state changes. Direct lookup has implicit mass1;this
is a finite token function,not learned semantic expert selection.

Selected13,408B preconv/scaled-generator row ->conv4/bias/BF16/SiLU and delta
bias/softplus/BF16 ->unchanged learned Ws/Vs projected B/C ->original AVX FMA
state update/per-head expf/full skip ->own F64 gate-RMS/full3072 gamma/BF16
norm ->new C implementation of admitted max-scaled I16 quantizer ->included
original PMADDWD128/I64 readout/F64 input scale THEN row scale/F32/BF16.
FE_TONEAREST required;all integer bounds unchanged. Code includes the qualified
full_channel_native.c/integer_readout.c;new compiled binary earns its own audit.
No source-target denominator,teacher-state reset or output supplied to deployable
branch. SSMout multiplier remains at later original decoder integration.

Original trained core parameters/site0 basis,p6 q/scales,conv/delta params
unchanged. Actual engine hsum/dot/FMA lineage retained,wide geometry intentional.
This is a standalone native research API,not a complete engine.c chatbot entry.

## Complete causal and numeric qualification

All48 histories/22547 actual positions/12 domains,FIT24+DEV24. Producer compares
one complete block with EVERY one-token continuation:all field/core/norm/I16/
scale/I64/raw/BF16 arrays and final full state/ring/count BIT-EXACT. Auditor
repeats the complete continuation independently in a separate process/program,
reconstructs final ring from actual selected source rows,and verifies every
actual native ID/implicit mass. Boundary-ID controls checked as producer receipts.

Old qualified prefix binary receives actual native fields as CONTROL ONLY,
ownonly1 with sourceinv unusedzeros:every B'/C'/latent timestep/finalstate/own
norm bit EXACT to new native branch. Control computes original dense readout,
which is unused by deployable branch and omitted from measured branch call.
Independent F64 projections/sequential recurrence/gating/normalization and
exact bounded-integer contractions additionally verify equations and all outputs.

|Independent check|Words|Outcome|
|---|---:|---|
|All native BF16 fields|151,155,088|0 differences to compiler packets|
|Projected B'/C'|4,329,024|prefix bits exact;F64 relative<=1.263e-7|
|Own latent timesteps|69,264,384|prefix bits exact;F64 relative<=3.336e-7|
|Final recurrent states|14,155,776|prefix bits exact;F64 relative<=4.232e-7|
|Own BF16 normalized words|69,264,384|prefix bits exact;smooth difference0|
|C I16 input words|69,264,384|exact independent quantizer/scale equality|
|I64 dots|46,176,256|0 independent differences|
|Scaled raw F64 scores|46,176,256|0 bit differences|
|BF16 outputs|46,176,256|0 bit differences|
|Actual token ID trace/implicit mass|22,547 each|all consistent|

Metadata residual0. No numerical/quality gate widened. ALL48 extra own-output
<=1% against accepted integer baseline and finalstate<=1% against accepted
actual-H temporal baseline PASS. All original local donor mean/worst/centered/
domain criteria PASS,FIT andDEV. New composed variant differs from its supplied-
actual-H baseline at most0.0617393% output and0.0150182% final state. Those
small differences include declared CPU normalization backend variation;do not
call the new path bit-identical to the original CUDA source.

|Split|Extra output mean /worst %|Source output mean /worst %|
|---|---:|---:|
|FIT|0.0044393 /0.0488325|6.506172 /8.874912|
|DEV|0.0119995 /0.0617393|7.639461 /9.341194|

The larger source error includes existing rank96 state approximation and p6
readout. Local source gate passing is not evidence that24-layer composition
preserves language/task quality. This experiment finally owns its initial SSM
trajectory from token IDs,but has no parallel attention/residual/FFN/later layers
or LM head/tokenizer decoder. It does not yet produce a chatbot.

## State-ready cost of the SAME audited artifact

Six fixed site0 FIT positions from prior min/max-history,first/middle/last;
five reps/two modes/core0/batch1,total60 calls. Before EVERY warmup and timed
call,regenerate the actual own prefix from zero and then time a single next
token. Prefix/context initialization and pointer arrays outside clock. Warm
means no forced source eviction;state is warmed by prefix preparation. Inside:
actual ID/row lookup,conv,B/C/scan/full skip/own norm/C quantizer/readout/all
debug output writes/FFI/ID view and pointer access overhead. Every warmup and
measured output word matches complete archive position;actual ID/count/ring
also match. Full ledger/keys/counts/source-byte totals/statistics reviewed,
without retiming or changing the DLL.

|Mode|Median us|Minimum us|Maximum us|
|---|---:|---:|---:|
|Warm ready|1012.90|658.90|1638.10|
|Source flushed|1044.05|876.90|2224.40|

Cold source verifies CPUIDCLFLUSH64B/MFENCE and evicts current13,408B F0 row,
q/scales,five core tensors and conv parameters:12,965,728B literal source
operands/call. Flush time separately. Initial architectural source eviction
is not PMU DRAM bytes,cold storage/page-in latency or useful expert-n scaling.
Shared desktop/frequency/scheduler not isolated;tails retained,no worst-case
guarantee. This is ONE state-ready SSM branch,not model accepted tokens/s.
Do not extrapolate it to the other23 contextual generators or complete model.

## Provenance/resources/closure

Source/protocol6b370d1,producer launchb48cecc,audit launchdaa1606,
audit/cost/closure admission458426c,cost launch2d87d75. Exact commands and
source/runtime/compiler/data/parameter/actual DLL extents in immutable records.
Binding28ec489ceeb9c1e066df07a198756fd2e5f29e2e3b971bf2f2239f02834e5576;
producer e0063e31363552567fbb26eb1226d5799baea4525cce9ad8a16be0cf1a35c15a;
audit59b1ebfc63525caecaa97377d15ff211af1f7c77286bc0a66e1afbe0f2d47130;
cost4c66b34484dc33b9fac5fa21677a6eef27b21407391dc503ef767f426e0ecbca;
closure856c551f46918dab97430838037cbf9701b6e6f6a87aae41c736875b925abd4f.

Prepare19.110s/OS47,628,288B,producer140.812s/OS541,216,768B,
full audit120.047s/OS927,617,024B,cost prepare1.297s/OS42,659,840B,
cost run39.172s/OS179,437,568B. All5 terminal0/PID+creation matches absent;
48 full NPZ1,763,094,640B plus DLL/log1,763,186,800B,new ordered E
token_state_20261013 under2GiB. Source tables/weights/bases referenced,not
duplicated. Cost E empty marker;ledger/receipts in research docs. All extents
rehashed,protected engine/foreign hashes unchanged,publishing daemons untouched.
No GPU/T4/source model forward/optimizer/RESERVED query/new donor. Principal
binary/source pins,not complete transient DLL-tree trace.

## Next decision toward the original objective

Stop first-branch refinements. [Full-width all-layer quality bridge](FULL_WIDTH_CAUSAL_CHAT_NEXT_20261013.md)
is selected next:existing trained bases cover all24 sites;preserve source
hidden2048/FFN/attention/head/tokenizer while testing actual own native state/
readout composition,without the FAILED D256 global projection. First qualify
generic row/state ABI and all24 source parameter/basis identities;then whole
held-out/chat/own-answer behavior. GPU source operators in such a quality
bridge must be identified and do not count as final CPU deployment/speed.

Whole goal requires complete source CHATBOT ->compact reusable SSM/SWA +useful
selective ternary capacity ->engine.c;RAM-driven useful n/10B/100B,actual CPU
IDs AND masses/DRAM,multiple source families/scales,conversion O(n) master/
gradient resolution and SAME useful>=50 tokens/s. All remain explicit open
requirements. No quality-preserving complete chatbot yet,no goal completion.
