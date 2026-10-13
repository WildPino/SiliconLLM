# Complete source-only token generator compiler

13 October2026. **COMPLETE / INDEPENDENT_STORED_PASS / ALL CALIBRATION GATES PASS.**
[Protocol](TOKEN_GENERATOR_COMPILER_PROTOCOL_20261013.md),
[binding](token_generator_compiler_binding_20261013.json),
[producer](token_generator_compiler_result_20261013.json),
[complete audit](token_generator_compiler_adjudication_20261013.json),
[custody closure](token_generator_compiler_session_closure_20261013.json).

## New reproducible conversion artifact

Canonical Falcon-H1-1.5B-Instruct source embedding,layer0 input norm,in_proj,
conv/delta parameters/config/effective BF16 MuP compile ALL65537 vocabulary
rows,including65536. Source-only build finishes every row BEFORE consulting
captured responses. No fitted rows,FIT/DEV response copies,vocabulary subset
or observed-word fallback. Rows remain a source function,not newly trained
capacity or proof of RAM-driven semantic expert n.

H0(token) follows literal CPU Torch source operations:BF16 embedding times
Python config scalar,F32 square/mean/rsqrt,F32 multiply,cast BF16 BEFORE
gamma multiplication,then BF16 SSMin .625. This declares a new CPU backend;
original CUDA reduction identity is NOT assumed. F0(token) uses promoted
H0/in_proj F64 contraction,F32/BF16 boundary,effective BF16 MuP and second
BF16 boundary,the same promoted generator equation as the accepted actual-H
function replay. Row-major F0.BF1665537x6704=878,720,096B,H0.BF16
65537x2048=268,439,552B. Actual uint32 IDs select13,408B F0 rows;token65536
cannot fit uint16. Gate/x/B/C/delta pre-convolution layout fully retained.

After vocabulary compilation,all48 full source histories select F0 rows
using token IDs,apply causal depthwise conv4/bias/BF16/SiLU and delta bias/
softplus/clamp. Five full BF16 field packets per history stored. Thus this
artifact removes captured H/generator responses as INPUTS to field generation;
the captured fields remain validation targets. Convolution is own-history
with zero prefix,but native SSM state and incremental inference are not yet
joined in this experiment.

## Full independent data/equation verification

Separate auditor imports no producer. Complete actual source coefficients,
all vocabulary rows,manual BF16 low-bit/parity RNE,effective MuP,different
projection chunk/einsum schedule,causal window contraction and independent
stable sigmoid/softplus equations. Every metric/flag/hash checked before/after.

|Verified array/control|Words|Independent differences|
|---|---:|---:|
|Complete H0 vocabulary BF16|134,219,776|0 bits|
|Complete F0 vocabulary BF16|439,360,048|0 bits|
|Source H0 calibration words|46,176,256|source differences retained below|
|All five causal fields/all48 histories|151,155,088|0 bits|
|Scalar-promotion controls|33,026 inputs|all stored variants checked|

Field-reference errors0,metadata residual0. H0/F0 exactness is against the
DECLARED CPU compiler equation,not original CUDA outputs. The CPU norm
audit uses a separate program with the SAME literal Torch CPU kernels;
it is not an independently implemented Torch reduction/rsqrt kernel. Full
original source calibration addresses a different numerical boundary.

All48 H0 gates<=1e-4 per case,all240 source-field gates<=1%,and all240
incremental-field versus accepted actual-H function gates<=1% PASS. Worst
incremental field error0.0374494%. Original-source H0 differs in357 BF16
words across11 histories out of46,176,256 words;these differences are not
called bit identity. Maximum H0 relative9.2774937924e-5 (0.00927749%),close
to fixed0.01% budget. No threshold widened. Original source fields differ
at declared BF16 backend boundaries;all differences retained case by case.

Per-case relative RMS percentages,source calibration:

|Field|FIT mean /worst %|DEV mean /worst %|
|---|---:|---:|
|H0|0.0004297 /0.0044320|0.0012775 /0.0092775|
|Gate|0.0131509 /0.0173111|0.0134823 /0.0181624|
|x|0.0127884 /0.0251600|0.0122596 /0.0212376|
|B|0.0225288 /0.0404356|0.0225197 /0.0433294|
|C|0.0088808 /0.0351871|0.0096350 /0.0202312|
|Delta|0.0016081 /0.0104020|0.0012184 /0.0049809|

These are source-function/causal-field metrics,not native recurrent-state
error,own-normalized readout error or chatbot quality. A recurrent system
can amplify a small field error;composition requires its own full checks.

## Scalar precision observation relevant to the converter

For ALL33,026 finite BF16 values with abs<=4,CPU Torch BF16 tensor times
the Python embedding config scalar agrees EXACTLY with the tested F32-scalar
calculation;rounding that scalar toBF16 FIRST changes772 results. The scalar
is5.656854249492381,not automatically the BF16 effective constant5.65625.
Do not conflate source Python scalar multipliers with the source MuP VECTOR,
which actually has BF16 dtype/effective coefficients. This is a bounded
actual-runtime witness,not proof for every dtype/device/value or future version.
No variant was selected using responses;literal source operation was frozen.

## Resources, provenance and closure

Source/protocol freeze140267f,producer launch3a6d366,audit/closure freeze
1d27ece. Exact launch commands/environments and actual runtime/principal
NumPy/BLAS/Python/psutil/Torch binaries/library extents in frozen records.
Full canonical model.safetensors/config/source implementation and source
capture continuity pinned;not a full transient DLL-tree trace.
Binding440fce2def09965c2127ce09948460e22c4b16b4fe02d0ac95709654eebe92dc;
producerfac01645c30fffbb82bcdcf4469065923765c865e8f8ddd8d0eff573d9774e67;
audit8f0c204183d038a5347665322b614e7df2f0830133d1e381126ecd55e73d1ee4;
closurefd5b081969213ba535835463c85a8e7ed45967b7752fe2b7f5068a4c408aafaa.

Preparation22.812s/OS407,216,128B,producer139.797s/OS2,802,675,712B,
full audit88.407s/OS2,763,546,624B. All3 terminal0/PID+creation matches absent;
all52 files/new ordered E token_generator_compiler_20261013 family
1,449,937,450B<2GiB. Whole F0/H0/params/controls/48 packets rehashed in
closure;canonical source/protected engine/foreign hashes unchanged,publishing
daemons observed/untouched. CPU0..5/Torch threads6/interops1. No GPU ever
initialized,T4,source MODEL forward,SSM call,optimizer or RESERVED query.
Source-function compiler pass1 explicitly distinguished from model forwards.
No source storage or conversion cost hidden behind a sparse-table claim.

## Next actual composition and complete objective

[Initial native composition/cost](TOKEN_STATE_RESULT_20261013.md) is now
COMPLETE/full independent auditPASS/all48 temporal/output/source local gates
PASS. Actual token IDs ->own conv/state/norm ->accepted integer readout,
full block versus every continuation exact,extra output worst0.06174%/final
state0.01502%,same audited state-ready median~1.01--1.04ms. Retain original
compiler and source differences;no rerun needed. Next
[full-width all-layer quality bridge](FULL_WIDTH_CAUSAL_CHAT_NEXT_20261013.md)
proposed/unbound,source hidden2048 preserved,whole behavior before further
component tuning. State-ready first-branch costs do not prove a complete
chatbot or accepted50/s.

This remains initial SSM branch only. Parallel attention,residual,FFN,other23
contextual generators,head/tokenizer and full OWN-history chatbot are missing.
Finite vocabulary compilation is not a general contextual expert compiler or
new knowledge. Useful RAM-driven n/10B/100B/multiple source families/scales,
conversion O(n) master/gradient storage and SAME useful>=50 token/s remain
goal requirements. Source method gains a real reproducible compiler artifact;
whole goal ACTIVE/INCOMPLETE,no owned job live.
