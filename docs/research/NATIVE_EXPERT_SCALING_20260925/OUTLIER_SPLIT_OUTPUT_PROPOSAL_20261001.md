# Proposal: BF16 output outliers plus row-Q8 remainder over full source features

**Status: codec/native operator and source priors implemented/pass;
fixed conditional count recipe rejected.**
METH-238's actual operator passes at8.727ms,METH-239 all16 stored source
priors pass. METH-240 produces176 distinct accurate functions but E160
loses22.39% against E16. METH-241 confirms an exact BF16 source-target
contract and motivates the separately frozen METH-242 source-value-only
correction. The original proposal and shape arithmetic below are historical;
operator/prior acceptance does not establish useful expert-count scaling.
METH-234's full4864-feature row-Q8/FP32/LUT operator passes source-function
fidelity/cost. METH-235 continuous source derivatives transfer, but actual
int8 output encodings have1.05%–1.18% slope error. METH-236 excludes scale-only
repair at those codes. METH-237's four encoded-feedback cycles still fail
all16 stored derivatives. This does not establish a universal int8 limit.

## Changed precision/encoding mechanism

Keep full source gate/up row-Q8 maps and SiLU lookup. Change only output
readouts: select **32** largest-absolute coefficients independently in
each output row,stable descending magnitude with lower original feature
ID first at ties. Preserve those coefficient values in BF16,store their
original feature IDs as uint16 (4864 fits),and set the corresponding dense
int8 codes to zero. Encode the remaining coefficients with symmetric
row-max/127 int8 and FP32 row scale. Zero remainder uses scale1.

Actual output row is `scale * dot(q_as_float,phi)` plus a separate FP32
sum of32 decoded-BF16 coefficients times indexed phi,then FP32 bias.
All source units remain accessible; no input/hidden activation quantization
or affine term. This trades a small indexed higher-precision subset for
a smaller quantization range in the remainder. Neither source gradient
gain nor CPU gather cost is known. Weight magnitude is a fixed codec
rule,not a claim that those rows alone carry most useful function capacity.

For original-source component fixtures BF16 exceptions are exact original
weights; projected FP32 priors would round these exceptions to BF16 and
must independently qualify stored derivatives/function. Same codec cannot
be assumed to conserve learned readouts merely because it prices a source.

## Shape arithmetic and expert-count accounting

896 output rows x32 x(2-byte index+2-byte BF16)=114,688 added bytes/layer;
across24 layers **2,752,512bytes**. Dense int8 body remains full-size with
zero escape codes; no claimed removal/deduplication of those bytes.
METH-234 fixture plus escape payload would be317,646,876bytes (including
the existing header/table); hypothetical whole ideal selected ledger
542,052,356+2,752,512=**544,804,868bytes**,below560MB. These are addressed
shape bytes,not complete model traffic or full accepted rate.

Each independently stored output function costs107,520,000bytes across24
layers instead of104,767,488. Shared gate/up remains210,124,800bytes plus
2052-byte table. Hypothetical E16/E160/E1600 function residency becomes
1,930,446,852 /17,413,326,852 /172,242,126,852bytes,excluding other core/router
organs. No such bank exists. One output readout active/token; higher n
still requires useful distinct functions,actual route/LUT/DRAM measurements
and quality preservation; exception indices cannot substitute for capacity.

## Next exact prerequisite: METH-238

Freeze/implement this fixed32-exception codec and complete AVX2 row-Q8+
indexed-BF16 readout operator on all24 original source layers/METH-125
states. Bind unchanged gate/up/table to METH-234 bytes; verify every escape
ID/value,selection rule,zeroed int8 position,scale and serialization read.
The declared GPU oracle must preserve separate row-scale-after-reduction
and exception accumulation rather than silently using a predecoded GEMM.

Retain C/oracle median<=1e-4,max<=5e-4 relative output L2, pooled original
BF16-weight/FP32 source-function SSE/energy<=.01 and six-thread24-layer
component median<=10ms across three fixed passes,no retiming. Freeze
layout/exception reduction and local budget before observing results.
An operator failure stops this fixed mixed-precision representation before
repeating source priors or conditional output fitting.

A pass licenses a separately frozen source-output prior qualification
using the actual mixed codec and unchanged1% stored-derivative gate. No
additional METH-237 cycles,old-scale repair or tolerance adjustment.
Full learned E16/E160 function gain,stored routed bank,independent LLM
quality/generation/tasks,>=50 accepted token/s and actual approximately10B/
second family remain required. GigaChat's sparse/top4/shared/full-width
operator and donor-adaptation fidelity evidence need their own priced
precision/feature-sharing variant; no automatic applicability is claimed.
