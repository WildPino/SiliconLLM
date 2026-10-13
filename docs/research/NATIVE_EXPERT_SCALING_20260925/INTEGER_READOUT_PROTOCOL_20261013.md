# Fixed p6-weight I16 input/dot readout correctness and fidelity

All144 histories/sites0/12/23,same successful p6 q/scales as prior native LUT.
q_weight int16[-364,364],source-only weights unchanged. New input quantizer:
F64 a=max(abs(actual own-norm BF16 Z))/32767,explicit all-zero vector a0/r0,
RNE of actual F64 ratio Z/a clipped[-32767,32767],int16. No FIT/DEV tuning or
new teacher/source/recurrence. This changes activation precision;separate error
budget,not an inherited claim of BF16-preserving LUT execution.

Native one-core PMADDWD16x16 pairs,int32 partial per128 coordinates,reduce
eight lanes via pair/hadd operations within signed int32 bound,accumulate24
block results in int64. Product max11,927,188,pair23,854,376,per lane-block
190,835,008,full block1,526,680,064<2^31,total36,640,321,536<2^53 and2^63.
Weights/input never-32768;PMADDWD special double INT16_MIN overflow excluded.
Before actual histories,producer checks all2048 outputs at4 sign-extreme
combinations,one coordinate mutation(-32767 delta only output0),zero inputs.
Controls are producer receipts;full independent all-history audit follows.

Exact signed integer dot ->F64 conversion (integer<2^53),multiply input a
FIRST then p6 row scale,then F32 cast/BF16 RNE,ffp-contract=off. Every native
integer dot must EXACTLY equal independent reference. F64 matmul of bounded
integer operands is exact:every product/possible partial sum absolute bound
is below2^53,so normal IEEE operations/FMA represent the integers exactly.
Verify actual runtime sums integral/in bound and every int64 word equal.
No score-relative tolerance may hide an integer mismatch.

Archive all input int16 words/scales,dots int64,scores F64,outputs BF16 for
every case. Independent auditor does not import producer:source p6 q from
digit decoder,manual sign/floor/parity input RNE,all32-row exact-F64 integer
reference/converted int64,independent scale order and low-bit BF16 RNE,
EVERY full array/metric/flag/aggregate/hash/source gate. Input scales/F64 score
relative<=1e-12;metadata abs(delta)/max(abs(ref),1)<=1e-10;activation-coordinate
half-step max<=1+1e-10,exact int64 and BF16 bits required. Zero rows explicit,
metric norm guard1e-30 only,not a hidden quantizer scale/floor.

Quality gates frozen:extra raw against original native Y<=.01 and BF16 against
original native output<=.01 EACH case;incremental raw activation error versus
qualified unquantized-input p6<=.001 EACH case. Original donor mean.1/worst.2/
centered.1/domain.2,FIT/DEV separate,unchanged. Whole/centered/labels/domains/
worst errors and p6 BF16 mismatches retained. Even failed quality completes
all data/audits;native cost admission only after correctness/fidelity success.

Source/protocol/storage freeze before compiler/prepare/binding/run. Compiler
source/reference actual runtime ABI/NumPy/BLAS/Python/psutil/prior reports/
adjudications/144 native+source+p6-LUT packets/three q/scales/protected engine/
foreign files pinned before/after. Compiler child allowed only bounded45s
prepare,then no descendant jobs. Source provenance through qualified parent,
no source weight/model load. Principal DLL pins,not full transient tree proof.

Prepare/run900s/audit1200s/OS6GiB/new orderedE3GiB,BLAS6/CPU0..5,actual C calls
CPU0 only. Full144 output families~2.91GB decimal including input witnesses,
below3GiB. Two compile artifacts in family too. No GPU/T4/RESERVED/optimizer.
No native timers here;after full audit separate SAME accurate artifact cost
on fixed warm/source-flushed samples/counter scope. Int16 q deployed12.60MB
including scales versus original25.17MB and ternary-pattern7.57MB;count real
layout bytes,not cheap arithmetic or model-rate assumptions.

Goal still source CHATBOT->compact shared causal core+useful selective ternary
functions->engine.c,OWN histories/multiple donors/scales/RAM-driven n/conversion
memory/SAME useful50. Shared operator success cannot complete the pipeline.
