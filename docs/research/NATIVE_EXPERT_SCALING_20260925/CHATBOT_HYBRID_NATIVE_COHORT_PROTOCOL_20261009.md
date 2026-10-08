# Learned balanced cohort -> packed original-LUT C target

9 October2026. PRE-OBSERVATION extension. Code prepared,not yet compiled/run.
Original native6-case/32-row gate remains13 PASS/19 FAIL. No old C replay.
The original recovery worker is still live; do not modify its bound launcher,
target or learner code until actual exit. This extension changes the native
fixture interface and streamed checking only,not core/LUT arithmetic.

## Reused inputs, eligibility and decision

Use an actual512-update HYBRID_RECOVERY_RESULT_V1 with
BALANCED_RECOVERY_ELIGIBLE and an exit0 terminal receipt,plus the corresponding
SAVED_RECOVERY_AUDIT_PASS/full_primary/checkpoint_boundary512/all reproduced
gates true. Audit primary/checkpoint/corpus hashes must match this exact cohort.
No partial/failed checkpoint qualifies through this interface. A partial
attempt first needs separately frozen evaluation/continuation; missing outputs
are not manufactured or treated as eligible. All source calls remain0.

Questions:can the NEW actual learned model execute ALL160 saved teacher prefixes
through original C machinery,including actual inputs exceeding128,with its
whole target output qualified? What are direct donor-relative prefix metrics
on THIS C artifact? These remain calibration,not fresh own-history chatbot or
accepted50. A full parity failure retains every completed C byte/trace rather
than rerunning prefixes or changing tolerance. No long T4 work from this alone.

## Export and dynamic bounded fixture interface

Use existing packed exporter:211 actual F32 master/control tensors,36 ternary
gate/up/down arrays replaced by reversible tile-major pair codes,one64-value
RoPE field.212 fields,425,188,608B payload,425,210,736B model. No master or
unpacked expert copies in native model. CUDA F32 rounding exactly matches
learner; no model/source forward or optimizer update. Geometry unchanged:
D512/L12/SSM10/SWA2/E72/k8/h128/V65537,AQ63,original F32 control organs/scales.
Existing old exporter/node receipts keep their frozen identities.

Driver accepts explicit --learner-result,--learner-audit,--corpus. No implicit
selection of a checkpoint or favorable DEV epoch. Main C query API now accepts
1..1024 cases,1..512 input IDs/case and1..input_count increasing supervised
positions. Query/output header carries actual case/row count,not6/32 constants.
Original magic/layout and uint32 IDs unchanged. Every ID<V,position in bounds,
header rows equal actual rows; Python independently checks total input extent.
C report schema HYBRID_NATIVE_C_V2 records actual counts. Stack buffers512;
all operator/loader/fixtures/FE_TONEAREST/FTZ/DAZ/kernel bodies unchanged.

For balanced cohort ALL160 canonical student_input_ids/positions must equal
source-prompt + saved continuation[:-1] and saved labels including EOS.
Entire state zero once/case then continuously evolves; SWA window remains128.
Actual longest corpus input183; this can exercise window eviction but does not
prove preservation of the source's long-context capabilities. No per-token
reset or changed activation. Flat72 stable top8/selected normalized mass.

## Complete streamed checks and direct C-to-source metrics

Compare ALL5746 full65,537-vocabulary NEW C rows against already saved final
learner rows. Fixed relative row logit RMS<=1e-4 EACH and ALL greedy learner
IDs equal. Zero-reference/zero-error row passes;positive-error/zero-reference
fails. This is the same criterion as old numerical gate,not a loosened rule.
Read case-sized pieces instead of whole multi-GB arrays. All inputs/outputs
finite,exact file extents/header,all code/original fixtures pass.

ALL trace records(total_input_IDs*12,12640B each) checked in1024-record chunks:
all fields finite,independent stable NumPy top8 IDs exactly equal,C normalized
mass versus F64 selected softmax max absolute error<=1e-6. ALL160 final state
packets9,117,696B each scanned finite in1MiB chunks. Streaming only changes
inspection memory;all original data are retained. Native CPU/model/state/LUT
cost still charged,not an accepted throughput benchmark.

Also read original adopted BF16 source rows,exact bit-lift,F64 stable logsumexp,
and compute forward KL/ID disagreement directly versus NEW C full outputs.
Report equal-case AND label-weighted per FIT/DEV/domain,plus all per-label KL.
This does not equate learner parity with source preservation.23 partial source
answers remain prefix supervision;related templates and consumed DEV limit
generalization. No new source generation or student own-prefix observation.

## Resource budget and exact launch sequence

Only after actual recovery exit/required saved audit and eligibility:
compile/review changed driver/C and freeze all code/protocol before binding.
Then separate one export family and one native family,no overlap. Export
unchanged600s/8GiB OS/512MiB allocated GPU/1GiB reserved/1GiB output.
Native600s/4GiB OS/6GiB output/4MiB log,one CPU C thread and compiler child;
original held-handle/descendant guards and preserved foreign hashes.

Price exact total_input_IDs from cohort BEFORE launch:
`16+5746*65537*4 + total_input_IDs*12*12640 +160*9117696 +16MiB`
must fit6GiB. Full logits1,506,302,424B including header,states1,458,831,360B;
Actual sum15,999 gives2,426,728,320B trace and5,408,639,320B priced total
including16MiB allowance,below6,442,450,944B cap. These are metadata/dimension
deductions,not actual C writes or measured memory. Worker checks in case/chunk-sized pieces
with expected workspace below~1GiB;actual family peaks,not this estimate,gate.
Model/states original~434MB native heap before other allocations.

Freeze compiler/worker/code/kernel/checkpoint/source/reference/cohort/audit
bytes and exact version paths. Native compile uses original body extraction,
Clang O3/AVX2/FMA/scalar contraction off,no fastmath or extra reference banks.
Save command/PIDs/direct-child through-exit peaks/all headers/fields/traces/
states/logits/report. First faults and numbered missing-scope repairs retained;
metadata/comparison fault after C does not authorize completed prefix replay.
Prepared apparatus has no execution/result claim. Further fresh own-history/
chat tokenization+stopping,accepted50 on SAME useful artifact,large useful n/
structured CPU LUT IDs AND mass,physical DRAM/family/10B/100B remain open.
