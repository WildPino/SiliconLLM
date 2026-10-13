# Frozen exact engine-style integer byte LUT

Same admitted p6 weight q/scales,actual input I16/a,exact I64 dot and F64/BF16
outputs from integer_readout_result/full V3 audit. No new approximation or
quality-budget change. Parent donor/additional gates must pass before prepare.
Source-only packing from q before any input responses;all three sites/all144
histories,original ordered records. Retain complete arrays/audit on failure.

Source includes actual donor_adaptation/nibble_pack.h unchanged,with literal
acc_add_i8x32 from phase60/engine.c,OMP_PFOR empty (one thread). Full nibble
matvec_lut_full_n called24 times/input (six weight planes/four input slices).
Layout codes[p,768,2048],two adjacent REDUCTION two-trit codes/byte,low first/
high second. Code3*(t0+1)+(t1+1) in0..8,neutral4,no padding required because
2048 multiple32/3072 even/fixed ABI. Physical codes9,437,184B/scales16,384B.
Include actual engine/header/compile/runtime/ABI/full parent archives/protected
foreign files in immutable before/after pins. No engine/header modification.

Unsigned16 word decomposition d0=u&15,d1=(u>>4)&15,d2=(u>>8)&15,
d3=(u>>12)-16*(u>=32768);sum[1,16,256,4096]*d exactly signed input.
Actual input range[-32767,32767] remains enforced in independent audit;
exhaustive decomposition control includes all65536 signed words including
-32768 solely to check bit representation. Low slices0..15/top-8..7.
Pair table16 signed bytes,valid0..8 as original trit formula,invalid9..15
explicit0. Low bound30/top16,each within signed8. Every table for all65536
first coordinate words/bitwise-complement second coordinate archived and
independently checked(all4,194,304 entries);all262,144 digit words archived.

Each trit/digit sum<=46,080 int32;weighted low sum<=16,773,120/top8,945,664.
Sign-extend four weighted sums BEFORE powers1/16/256/4096 and int64 addition.
Full bound36,640,321,536<2^53;same F64 dot conversion/input a THEN row scale/
F32 cast/BF16 RNE as qualified integer kernel,ffp-contract=off. Every new dot,
rawF64 bit pattern and BF16 word must be BIT-EQUAL to prior qualified archive.
Independent auditor imports no producer:decode low/high codes and all q,
manual integer bit decomposition/control tables,exact bounded-integer F64
contraction then I64,scale order/manual BF16,EVERY stored word/check metadata.
Audited raw bits requirement exact (no relative norm allowance). Parent
quality gates are inherited ONLY through complete archived output identity.

Native producer four all2048 sign-extreme dots,one p0 trit mutation(only
output0 delta-32767),zero input. Producer counted-once instrument switches
original header code-load counter:expected24*768*2048=37,748,736B logical
vector-load operands/input. It is NOT cache/PMU DRAM. Counted/uncounted output
parity required;counter disabled for ordinary calls and later cost. Controls
are producer receipts checked by auditor,not a new independent native rerun.

Actual scratch allocations:four full tables98,304B,temporary2048 I32=8192B,
four2048 weighted I32 totals32,768B,total139,264B/history call. Per-token
table rebuild/24 literal kernel passes/combine/rounding,preallocated full
history outputs. No plane skipping/layout tuning/vector int16 modification.
Sample cost later on SAME fully audited binary includes tables/allocations
and original input BF16 decode/qualified Python quantizer,paired against SAME
admitted PMADDWD binary. No integer or dense retiming to replace old evidence.

Prepare/run900s/audit1200s/OS6GiB/compile-child45s/all-owned arrays+compiler
family3GiB,new ordered E exact_integer_lut_20261013. Expected new144 dot/raw/
BF packets2.493GB +source-only codes28.4MB+controls4.72MB. Parent inputs/scales
reused,not duplicated. Core0 native producer,BLAS6 only independent audit;
before/after full extents,terminal PID+creation/exit/resources. No GPU/T4/
new source/recurrence/optimizer/RESERVED/model or bank allocation. Whole
pipeline/useful-n/same useful50 still open. One cost comparison resolves this
primitive;then return to joined causal source functions,not endless variants.
