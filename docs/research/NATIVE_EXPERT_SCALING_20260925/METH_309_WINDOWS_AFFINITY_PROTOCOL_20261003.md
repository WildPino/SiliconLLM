# METH-309: direct Windows physical-core execution profile

Frozen before observations, 2026-10-03. METH-308 startup access violation remains a separate apparatus failure. This changes the binding mechanism and binary, retaining immutable METH-306 computation and all controls. No claim of isolated affinity speedup.

## Fixed execution and observations

- Engine opt-in SILICON_WINDOWS_AFFINITY_PROFILE, clang21.1.8, original306 compiler flags and libomp; default engine unchanged.
- All original 640-function/layer packed-I4 synthetic coefficients, source Q6 head, router, norms and exact spec. No learned weights or new source capture.
- Six physical cores from Windows GetLogicalProcessorInformation. Choose the lowest allowed logical ID of each core, without timing-dependent selection. Group0/64-bit only; stop on incompatible topology.
- ACTIVE/200ms, KMP_AFFINITY=none, six threads, no dynamic teams. Remove KMP_LIBRARY/OMP_PLACES/OMP_PROC_BIND. Record requested environment before launch and effective runtime settings afterward.
- SetThreadAffinityMask once per worker before initialization. GetThreadGroupAffinity checks singleton masks at bootstrap and before/after each of three repetitions. GetCurrentProcessorNumber must match at all six subsequent checks. Seven JSON records per process; no added operator warmups.
- Three fresh sequential processes, each original ten fixtures/three repetitions/two warmups/eight measured. All90 hashes, entire selftest and fixed ready fields must match306. All processes/observations retained; no retries or subset rescue.
- Each process max/min repetition median <=1.10; cross-process max/min process median <=1.10; all nine repetition medians <=14ms. Same 560MB descriptor budget (310571680B addressed); not physical DRAM measurement.
- >=16GiB available before each process; stop >24GiB peakRSS or >600s/process. Save failure JSON/logs for apparatus failure; no scientific inference from missing measurements.

## Decision and limits

Failure of variation gates is inconclusive; stable failure of cost closes this execution profile before collection/training. Only all passing gates permit a separately frozen real-source function-fit/precision pilot. It does not qualify donor quality, increased useful capacity, complete causal model, end-to-end rate, actual100B, or cross-family transfer. Recompilation can change code generation; comparison with old binary is observational.
