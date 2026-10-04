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

374 complete explicit physical-worker contract ALL7 PASS (0de8cf0), including
ALL96 teacher/own-natural full SHA exact363.375 same-artifact CPU cost ALL7 PASS
(aa3cf47).376 accepted FULL rate ALL5 PASS (56922f6):63.5254 ordinary accepted
IDs/s, lower95 59.5118;895 IDs from81/96 cases, all rejected-case times charged.
Includes405 structural sentinels; prose34.7793/s, lower32.6497. Warm pretokenized
short infilling, SAME338 weights/374 binary and six verified physical workers.
Original14.664B donor-relative quality inherited only through complete374 bytes
exact363. See SWITCH_BASE256_REPRODUCTION_20261004.md for the real procedure.

372 first mixed subset quality remains retained: no monotonic n improvement.
373 actual64/128/256 cost remains qualified; physical DRAM is unmeasured.
377 acquisition apparatus failed before any network bytes (82ca1dc). Corrected
378 frozen029e5b7 passed model-free committed-input preflight and is acquiring
the independent original base128, 29.863GB/90min/40GiB/2GiB RSS; no overlapping
model/native timing. This is a second same-family7.4B/14.7B source-scale test,
not a pruned subset or second family. Then379 all-original tensor/tie/parameter
binding, source-specific export/numerics and NEW original-primary quality/rate.
Useful >256, hierarchical CPU LUT/routing, physical DRAM, broader contexts,
other families/~100B and scalable useful capacity remain open. Goal active.
