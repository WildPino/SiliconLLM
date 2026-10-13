# Return to composition: source token generator -> causal state -> accepted readout

13 October2026. **COMPILER COMPLETE / NATIVE COMPOSITION UNIMPLEMENTED / UNBOUND.**
[Full compiler result](TOKEN_GENERATOR_COMPILER_RESULT_20261013.md): complete
source-only vocabulary/all norm+fields gates/full independent auditPASS.
Original proposal below retained;resume at native persistent state ABI,
full temporal/output/prefix-continuation audits then joined cost. No repeated
compiler build or isolated readout variants. The exact byte-LUT
comparison is [complete and cost-rejected](EXACT_INTEGER_LUT_RESULT_20261013.md).
Keep the accepted integer readout. Stop readout variants and remove a supplied
input boundary from the actual source-to-native path.

## Finite source function, not a fitted response cache

[Token input witness](source_token_input_witness_20261013.json) already
qualifies22547 source H0 rows/3816 distinct token IDs/18731 repeats with
ZERO repeated-word differences. Pinned source graph and configuration show
initial embedding(token)*embedding_multiplier ->layer0 input RMSNorm ->
ssm_in_multiplier ->in_proj ->effective BF16 MuP vector. There is no causal
history before this generator in the observed eval/no-padding token-ID path.
Define F0(token) as the6704 BF16 pre-convolution/scaled-generator values.
Finite vocabulary65537 makes a complete source-only table possible:
65537*6704*2=878,720,096B,13,408B per selected row,uint32 token IDs.

Build ALL vocabulary rows from canonical source embedding,layer0 norm,
in_proj weights/config/effective MuP with explicitly declared source dtype
boundaries. Do not assemble a table by copying observed FIT/DEV responses,
fit a vocabulary subset,or use cached captures as unobserved-word fallback.
Freeze the computation before comparing responses. Preserve old full source
and supplied-actual-H assays;this is a new compiler variant.

Source literal RMSNorm uses F32 square/mean/rsqrt,casts back BF16 BEFORE
gamma multiplication;embedding multiplication and SSMin have their own BF16
boundaries. BF16 input projection then BF16 MuP precedes causal conv/SiLU/
delta. Those roundings are not an interchangeable F64-normalize formula.
Determine actual primitive dtype/reduction behavior from pinned code/runtime
and complete controls. Mathematical lookup identity holds ONLY for the
function actually compiled;do not assume original CUDA reduction identity.

## What gets joined

Native input becomes actual uint32 token IDs plus the complete F0 table,
not captured H/x/B/C/delta/gate arrays. Read selected13,408B row,decode
BF16,apply history-dependent depthwise conv4/bias/SiLU and delta bias/
softplus/clamp at qualified boundaries. Feed fields into the SAME trained
full3072/state96 native core with full skip/own gate-RMS and SAME accepted
p6/integer readout;preserve weights/state maps and quality gates. Zero initial
states then token-by-token own updates,not teacher-state resets. Source
SSMout multiplier applies only at original decoder integration boundary.

Record actual token IDs,mass1,table addresses/bytes,convolution state,SSM
state and output. Compare a full-prefix reference and one-token incremental
execution over the SAME histories;causal convolution/SSM state must compose.
No original-source hidden states or target-dependent denominator in the
deployable own-normalized path. Keep fixed-source-denominator diagnostics
separate and preserve their known failures.

## Proposed admissions and bounded resources

Before execution freeze actual source-only compiler,complete-vocabulary
layout/coverage,source norm/projection semantics,numeric gates and resources.
Controls cover all observed22547 H0 rows,full table coverage including token
65536,zero/nonzero norms,special IDs,and original source generator/field
calibration. Distinguish BF16 bit identity from a new backend approximation;
report EVERY difference/field/temporal flag and complete independent audits
even on quality failure. Do not silently widen inherited1% generator-field/
own-output/final-state gates or original local donor criteria.

Independent audit should read every compiled row and actual source coefficient
range/recompute declared equations,all histories/fields/latent steps/final
states/own output,incremental-versus-prefill state identity,all hashes/receipts.
Source-only table provenance must survive alias/dtype/rounding checks. Only
then isolate source-flushed per-token row-fetch+conv+core+readout cost on
SAME accurate joined artifact,including conversions/routing. Initial source
eviction is not PMU traffic. Do not infer end-to-end50 from a layer assay.

Expected table878.72MB,optional H0 norm audit witnesses268.44MB;proposed
compiler/audit OS6GiB,run900s/audit1200s,new orderedE<=3GiB BEFORE additional
state archives. Compute storage of complete latent/final/output fields and
choose actual cap before binding. No hours planned/noT4. Prefer bounded CPU
compiler;any local3060 source-primitive use must be announced with its own
GPU budget/runtime pins and stop policy first. No resource allocated now.

## How this changes the pipeline decision

Success would compile one real trained function into direct selective RAM
access and JOIN it to causal own-state computation. It removes one generator
matrix call and supplied-field dependency,with explicit memory expansion.
It preserves source information,adds no intelligence,new trained capacity or
evidence that duplicated experts transfer100B knowledge. Vocabulary65537
is fixed by this donor;it is not RAM-driven semantic expert n.

This is the initial native SSM branch only,not a complete Falcon decoder
layer (parallel attention,residual,FFN still needed),not all24 layers or a
chatbot. Later23 generators depend on evolving context and cannot use this
token-only table. That is the next generalization boundary:cheap context
functions/useful conditional residuals or explicit further training,with
actual IDs AND masses/DRAM. Other source families/scales,own-history held-out
chat/task quality,conversion O(n) master/gradients and SAME useful50 remain
requirements. Stop after the joined compiler result/cost and reassess the
all-layer plan before another narrow kernel campaign. Goal unchanged.
