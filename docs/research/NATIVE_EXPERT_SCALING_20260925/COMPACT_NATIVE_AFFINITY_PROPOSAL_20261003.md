# Physical-core bound native profile after307 variation: proposal

**Implemented308/309, no cost pass.** [Result and decision](METH_309_WINDOWS_AFFINITY_RESULT_20261003.md):308 startup failure,309 verified binding but inconclusive timing. Historical prospective proposal follows; its next action is superseded. 307's exact306 binary ACTIVE
profile has14.780/11.764/11.320ms medians but fails repeatability/EACH14ms;
PASSIVE A/C also drifts15.63%. No quality training is licensed. Affinity is
a new execution-profile variable, not an explanation already established.

## Authoritative local topology metadata

After307 terminalexit0, Windows GetLogicalProcessorInformation returned six
RelationProcessorCore masks, logical-ID pairs0/1,2/3,4/5,6/7,8/9,10/11.
The caller process affinity permits0–11. Lowest permitted logical processor
per distinct physical core gives0,2,4,6,8,10. Probe with the native Windows
API at the NEXT run, rather than assuming this numbering for other hosts.
No performance/source-quality observation was collected by this metadata read.

A Windows64 SYSTEM_LOGICAL_PROCESSOR_INFORMATION record is32 bytes here:
ULONG_PTR mask, DWORD relationship, aligned16-byte union. Read required byte
count first (ERROR_INSUFFICIENT_BUFFER122), then parse relationship0/core
records; require six disjoint masks covering12 logical processors and all
chosen IDs in current process affinity. Fail unsupported topology, do not
silently substitute sibling threads or a partial core set.

## Next exact action

Freeze NEW controller/protocol before observations, preserve306/307 sources/
results. Use exact306 executable/spec/components/640-function bank/head and
ACTIVE200ms profile, six threads. Add explicit one-CPU-per-physical-core
binding through runtime-supported affinity, with verbose actual bindings.
Candidate local setting: KMP_AFFINITY=verbose,granularity=thread,
proclist=[0,2,4,6,8,10],explicit. Clear competing inherited OMP_PLACES/
OMP_PROC_BIND and document the resulting child environment. LLVM supports
explicit processor lists on Windows; verify actual thread->OS masks rather
than treating the environment request as proof. Source:
[LLVM affinity configuration](https://openmp.llvm.org/design/Runtimes.html).

Keep all prior scalar/format/head/router/fault controls and EVERY output/
route hash exact306. Keep original timing recipe/two warmups, repeatability
<=1.10/EACHmedian<=14ms, physical RAM and600s/24GiB resource stops. Decide
before observations whether one or multiple fresh processes are required;
retain all observations, no favorable repetition subset. No extra warmups/
gate changes to regrade307. If variation persists, inspect actual resource/
scheduling evidence rather than attributing success to later/faster trials.

Only a credible stable full cost profile can license separately frozen real
teacher-function fit and I4 precision quality. Reuse298 FFN inputs for a
defined function pilot where applicable; no blind49min recapture. Attention/
whole-mixture targets require separately planned instrumentation/data. Model
quality, learned distinct child utility/exposure, real CPU large-n routing/
DRAM, causal cache/head and SAMEartifact accepted50 remain mandatory, with
multiple real donor families/scales. Synthetic cost itself cannot complete
pretrained capacity transfer.
