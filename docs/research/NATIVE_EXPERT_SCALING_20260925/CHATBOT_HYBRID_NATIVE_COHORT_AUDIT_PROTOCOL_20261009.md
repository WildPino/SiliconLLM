# Streamed cohort audit with bounded F64 and exact-dyadic fallback

9 October2026. PRE-OBSERVATION code/protocol. Prepared/UNEXECUTED. No old
native/audit/model/source replay. Frozen before new certificate-check values;
test runner and cohort auditor run only after original live recovery exit.

## Purpose and coverage

The original exact-integer32-row audit independently certifies19 numerical
failures. The NEW160-case/5746-row interface needs complete certification
without blindly applying376,575,602 Python coordinate loops. Retain the SAME
exact predicate `squared_error*100000000<=reference_energy`,not a larger
tolerance. A conservative floating-point interval decides rows far from the
boundary; ambiguous rows use exact integer arithmetic. Any discrepancy with
the native producer's classification stops promotion and retains the data.

Bind exact NEW native/export/learner/corpus/results and all retained full-vocab
reference/source/native/trace/state packets. Require successful native exit
and matching model. Bind helper/auditor/protocol/Python/current launcher bytes,
NumPy2.4.6/psutil7.2.2. No source/student/native/optimizer call or GPU use.
The helper's separate algebraic checker must pass before a cohort audit launch.

## Algebra and arithmetic assumptions

All finite F32 values embed exactly in F64. A nonzero difference of two F32
values has magnitude>=2^-149 and<=2*F32max; its square is in the normal F64
range. Positive sums over n<=65537 stay normal/finite. Every reference square
is exact in F64 (at most48 significant bits); its reduction can round.

Use `u=2^-52`,covering relative error of normal IEEE F64 operations even under
directed rounding. Let m*u<1/2. Standard repeated product/error expansion gives
`gamma_m=m*u/(1-m*u)<=2*m*u`. Subtraction,squaring and at most(n-1) positive
additions give `|Ehat/E-1|<=gE=2*(n+2)*u`. Reference product exactness gives
`|Rhat/R-1|<=gR=2*(n-1)*u`. For n65537 these are about2.91e-11. These are
conservative operation bounds,not measured errors or permission to loosen1e-4.
Assume correct IEEE F64 arithmetic and ordinary NumPy sum of positive F64
terms; arbitrary compiler/library faults are outside this mathematical model.
No full native DLL certification is claimed. Zero energy is handled directly.

Exact real energies lie in `Ehat/(1+gE)..Ehat/(1-gE)` and corresponding R
interval. Each computed endpoint is moved outward by `nextafter`,including
the final exact-rational comparison multiplier100000000. Certify PASS when
upper(E)*100000000<=lower(R),FAIL when lower(E)*100000000>upper(R).
Otherwise decode each F32 into signed integer units2^-149 and sum exact
integer squared differences/reference energy. Equality passes. Zero/zero
passes;positive error/zero reference fails. No sampled row substitutes for ALL.

## Required implementation check

One bounded CPU-only check<=30s,output<=1MiB after freeze. Separate gold oracle
uses Python float.as_integer_ratio after exact F32->F64 conversion,then common
2^-149 units;it does not call the helper's bit decoder. Explicit exact boundary
10000->10001,nextafter neighbors,zero/zero,zero-reference,subnormal identity/
difference,largest finite sign reversal. Seed20261009,100 random finite raw-bit
vectors of length1..128,identity/one-ULP-neighbor/sign-reversal variants.
ALL308 vectors must equal the exact-ratio oracle;exact boundary must exercise
integer fallback. This validates implemented cases,not a replacement for the
algebraic bound or a target quality measurement. No completed model output used.

## Actual NEW cohort audit

Check all212 packed descriptors/contiguous aligned offsets/field hashes/shape
extents/dtypes,all84,934,656 code bytes0..8 and340,253,952 finite F32 bytes/scales.
No inference master/unpacked reference arrays. Whole model425,210,736B.
ALL160 metadata keys andactual input/position mapping equal source corpus.
ALL5746 full-vocabulary C-versus-learner rows receive an interval/zero/exact
certificate and exact argmax comparison. Require agreement with every native
row and its final PASS/FAIL decision;report all certificate methods/intervals
and actual failures. A successful audit certifies a FAIL when that is the result.

ALL191,988 trace records(15,999 inputs*12) streamed in1024-record chunks,
all fields finite,stable top8 IDs exact,F64 normalized mass<=1e-6. Count visits
per layer with correct absolute index across non-12-aligned chunk boundaries;
each layer sum==15,999*8. ALL160 state packets streamed finite in1MiB chunks,
1,458,831,360B. Direct donor-relative prefix metrics remain the separately
retained native producer's F64 observation;this auditor does not claim their
independent reconstruction or fresh chatbot quality.

ONE local CPU-only family600s/4GiB OS/16MiB output/4MiB log,no overlap or
children. Held actual worker handle/peak through exit and all input hashes
before/after required. Streaming bounds memory but real peak/time gates.
No rerun after a metadata fault once complete observations exist. Routine
Graphify disabled. Original old19/32FAIL and all goal quality+rate,n/RAM,
structured LUT winner AND mass,physical DRAM/family/10B/100B requirements remain.
