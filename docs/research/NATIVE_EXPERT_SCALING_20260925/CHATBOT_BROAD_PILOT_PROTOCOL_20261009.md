# One broader-data transfer pass (pre-observation protocol)

9 October 2026. Conditional on the separately qualified F32 SSD storage
schedule. The pilot isolates new supervised coverage on the existing target;
it does not change engine.c geometry, selected work, precision or routing.

## Question, reused evidence and decision

Can one fixed 24-update pass over the adopted 24 FIT examples improve the
unseen 24 DEV examples while retaining old32 DEV behavior? Reuse the audited
286-recovery/Adam294 checkpoint, all8808 donor labels, original old32 DEV
logits/summary, and the completed storage qualification. No donor regeneration,
old286 fit replay, old baseline replay, RESERVED query, native run or T4.
New48 inputs have not had original-inference whole-model metrics at this state.

Keep D512/L12,10SSM/2SWA,window128,E72/k8/h128,V65537,F32 organs/scales,
ternary SwiGLU/AQ63 and flat router. Install EXACT qualified storage helper
in the worker. This is an offline memory schedule, not a new deployed operator.

## Fixed experiment

Measure all48 original-inference before rows; one domain-interleaved pass over
24 FIT, two examples per12 strata; measure all48 after and old32 DEV retention.
Sort domains lexicographically and their two FIT IDs lexicographically; visit
each domain's first example then each domain's second. Freeze explicit order
in the binding. No shuffling, DEV checkpoint selection or early favorable stop.
Final boundary24/Adam318 is selected before observations. New candidate schema
is distinct from the original incomplete512 eligibility contract.

Restore audited CPU/CUDA RNG after module/Adam construction and before new
observations; evaluation has original AQ/no dither and consumes no training
randomness. Start updates from the same saved RNG (restore again after before).
Train-only normalized AQ dither .025; original F32 full-vocabulary case-mean
KL/temp1,lr5e-5,clip1,wd0,betas(.9,.999),eps1e-8,foreachFalse. Nonreentrant
whole-block checkpoint/preserve_rng_state=True. Clamp expert row scales to
>=1e-8 after each update, as in the original recovery recipe.

Record every case/label KL, predicted IDs and disagreements, EOS/partial
status, domain/split, full F32 before/after/retention logits, all211 finite
gradients and12 positive core/bank/norm norms at each update, routing counts
by stage/domain. Collect training routing only during initial forward (disable
collection during checkpoint recomputation); expected8 selections/token/site.
Coverage counts describe consultation, not expert usefulness.

Atomically persist model/all Adam moments/CPU+CUDA RNG plus observation/support/
update metadata after each COMPLETE update to one bounded checkpoint path.
Keep the prior durable file until the next fsync+replace completes. A fault
preserves the last durable boundary and does not certify an in-memory later
update. No automatic retry/replay. Original source checkpoint stays immutable.

## Fixed gates and limits

Reuse original numerical development gates: new FIT case-KL ratio<=.50;
new DEV ratio<=.75; new DEV case-KL<=1 and label disagreement<=.20;
EVERY new DEV stratum case-KL<=2 and disagreement<=.35. Retained old32 DEV
case-KL and label disagreement must each be <=1.10 times their previously
measured checkpoint286 values (3.080980644 and459/1103; use exact bound JSON).
These are finite transfer/retention criteria, not validated chatbot usefulness
criteria; own-history response/task and same-artifact native/rate gates follow.
Report all failed gates; relative gains cannot override absolute failures.

All48/48/32 cases and24 updates, finite state/moments/gradients, exact Adam318,
correct support counts, and resource/input gates must be complete. No missing
historical support reconstruction or reinterpretation of failed original caps.
Family<=2400s with90s reserve,OS<=16GiB,CUDAallocated<=11GiB/reserved<=12GiB,
output<=12GiB/log4MiB; six worker cores/no overlapping model/native work.
Based on qualified actual long-case backward plus fixed evaluation/checkpoint
allowance. Stop/preserve first fault. No longer fit or architecture expansion
until the actual new coverage/retention observations change that decision.
