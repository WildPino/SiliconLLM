# Next execution experiment after the actual bank-size comparison

**Historical prospective plan, now completed.**374/375/376 passed; see the
current resumption below. Preserve the original rationale and proposed gates.

## Why this variable can change the decision

Full256 has363 bounded original-primary whole quality PASS;364 same binary
accepted full rate45.1114 ordinary IDs/s including markers, lower42.5362<50.
Single CPU core is stable but encoder+crossKV accounts about60% of that workload.
Six-thread358/360 profiles failed frozen per-fixture repeat criteria. Those
experiments used passive waiting and process affinity, with no independently
verified persistent assignment of each worker to a separate physical core.
An exact new execution profile can address the observed stability/throughput
constraint while retaining learned capacity. This is not an unchanged retry.

373 fixed-length CPU cost supports64-to256 pool growth with full upper95
overhead3.62% and decode1.27%, but cannot replace364 natural accepted rate.
372 shows mixed quality at smaller n; choose the qualified full256 model for
the speed investigation. No smaller-bank quality inheritance or selection
based solely on the favorable128 masked NLL.

## Proposed new mechanism, to be implemented and frozen

Clone the qualified356 execution path under a new374 engine opt-in. Retain
every mathematical projection/norm/attention/router/probability/capacity/cache/
head/argmax/stop operation. No new weights or representation. Default engine
body and earlier opt-ins must remain byte-identical.

For the current six-core Windows host, discover actual physical topology and
permitted logical CPUs, cross-check359: one worker on each of0,2,4,6,8,10.
Use explicit Windows per-thread affinity with readbacks, not a process-wide
pool as a substitute for worker binding. Restrict to a supported single
processor group and exact allowed masks; fail clearly on unsupported topology.
Record OpenMP slot, actual Windows thread ID, group and actual one-bit mask.
If a new worker appears, it must be bound and verified before computing.
Verify the executed teams and readbacks, including startup and repetitions;
do not assume worker persistence from the OpenMP slot alone. Any required
binding work during inference must stay inside the primary full timer.

Use dynamic teams disabled and fixed six workers, OMP_WAIT_POLICY=ACTIVE,
KMP_AFFINITY=none and OMP_PROC_BIND absent. Explicitly sanitize inherited
runtime placement/blocktime variables, and specify the exact retained values
before freezing. ACTIVE is a runtime waiting policy/hint, not a measured
proof that migration or parking caused the earlier failures. Treat worker
placement and waiting together as the new execution recipe; do not claim
their individual causal effects without a separate factorial experiment.

Primary API/documentation checked4October2026:
[SetThreadAffinityMask](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadaffinitymask),
[GetThreadGroupAffinity](https://learn.microsoft.com/en-us/windows/win32/api/processtopologyapi/nf-processtopologyapi-getthreadgroupaffinity),
[LLVM runtime environment](https://openmp.llvm.org/design/Runtimes.html),
[OpenMP waiting policy](https://www.openmp.org/spec-html/5.0/openmpse55.html).
Thread masks must be within process masks; group readbacks are required.
API availability and runtime behavior must be tested with the pinned native
compiler/libomp, not inferred from generic documentation alone.

## Sequential qualification and decision

1. **374 complete execution contract.** Before observations freeze code,
   protocol, source/math identity checks, topology/environment, costs and gates.
   Reuse356 independent I64/A16 primitive/Tiny/nine-fault evidence only where
   the math remains identical. Verify actual worker placement and full arrays
   on engineering/long controls at1/6 workers and profiles, then ALL96 consumed
   363 complete forced AND own-natural outputs SHA EXACT363 at the new primary
   six-worker recipe. Full source payload/spec/binary/compiler/runtime hashes
   fresh; no whole original reload needed for unchanged mathematics. No altered
   metric thresholds. Any mismatch blocks scoped quality inheritance.
2. **375 separately frozen CPU cost.** Only after all374 gates pass. Same
   binary/weights/worker masks/environment; old source9/64 forced32 controls,
   warm1/rep3, <=20ms decode/full and <=1.10 repeats under the earlier rubric,
   complete bytes exact. Preserve the first result; no optional unchanged retry.
3. **376 separately frozen accepted full rate.** Only if375 passes. Same
   actual binary/weights/profile and ALL96 complete own-natural SHA exact363.
   Reuse364 acceptance masks and metric rubric, ALL rejected-case full times
   charged. Ordinary generated IDs including sentinels and prose-only rates
   explicit; primary one-sided book-bootstrap lower95 >=50, same repeat1.10.
   Warm/startup/load/tokenization/serialization boundaries stated, encoder/
   crossKV/greedy/cache/stop included. Exact same-artifact quality/rate only.

Proposed374 budget MAIN30min/16GiB RSS, local only, expected<=10min plus ALL96
bridges. No T4/download/training/new model family or competing timing job.
375/376 each needs its own prospective budget. Stop and retain any first
failure before a narrowly specified new variable. Do not weaken the existing
50/quality/repeat gates or label forced decoder positions accepted tokens.

Even a successful376 closes only this bounded full256 Switch short-infilling
quality/rate step. Useful >256, real physical DRAM/LUT/hierarchical routing,
new contexts/tasks, other families/scales and~100B remain open parts of the
method. Frozen donor-adaptation/GigaChat assets remain reusable evidence;
resuming the costly generic GigaChat runtime unchanged is not the next step.

## Current resumption, 4 October: supersedes earlier proposed steps

389 both original sources/ALL192 teacher+natural whole outputs exact363/387 at
three actual workers[0,2,4], numerical contract PASS.390 original128 ALL6 cost
PASS; full256 three-worker repeat1.105770>1.10 FAIL retained29dfdf8. Original
256 six-worker374/375/376 route remains qualified.383 original128 six-worker
failure and386/388 apparatus failures preserved; drafts384/385 ineligible.
391 original128 SAMEartifact accepted FULL rate ALL5 PASS4764544:96.56385
ordinary IDs/s/lower94.58827,55.97659 prose IDs/s/lower54.59729,96/96 healthy,
662 prose IDs, repeat1.012403. Two actual same-family pretrained source scales
have complete bounded quality/rate; worker variants independently qualified.
No causal n/other-family/general-context/physical DRAM/~100B conclusion.
392 read-only routing audit PASS62bde68: selected probability medians~.10/.25,
so exact selected identity alone cannot inherit full softmax scaling. Next393
capture full router inputs/scores under exact whole-output identity, then
measure oracle retained tail mass before choosing hierarchy/LUT approximation.
More useful choices remain the priority; no unchanged rate polishing or old
GigaChat generic-port restart. See INDEX.md/METHOD.md for exact current state.
