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

Qualified original Switch128391/256376 whole-quality/rate intact.401 union
fits RAM but403/404/406 specified interfaces FAIL;405 full unregularized fit
ineligible. No384 selector/model. 408 frozenfb897e9/retainedd49c951: ALL7 numeric/complete-archive gates PASS,
synthetic full503.905MB Ling operator medians37.338-40.777ms (3 workers) and
23.719-39.412ms (6);14ms cost FAIL.409 frozenb64aea0/retained9bced64: ALL10
controls PASS/180SHA+60 complete archives exact408; six-worker coded matrices
12.452-14.396ms/head6.695-7.733ms/router1.516-1.580ms. No universal one-component
exclusion; original408 format closed before acquisition/fitting. No actual Ling
values/model quality/rate/useful n/DRAM. Next separately frozen joint smaller
second-palette/static pair-LUT plus Q4-head cost hypothesis; planning only.
See LING_MINI_JOINT_FORMAT_NEXT_20261004.md. All408/409 handles terminal.
Use INDEX.md/METHOD.md; preserve unrelated work.
