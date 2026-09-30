# METH-199: six-core OpenMP binding for large-n LUT timing

METH-198's six-thread no-prefetch E=12,800 and E=128,000 runs fail
their <=1.10 within-size repeatability limits, so their scaling
ratio is inconclusive. Test whether OpenMP thread placement explains
that instability. Reuse the exact METH-197/METH-198 binary/source;
do not change arithmetic, routes, selected bytes, or pool layout.

For each process set `OMP_PLACES=cores`, `OMP_PROC_BIND=spread` and
`OMP_DISPLAY_ENV=VERBOSE`. Require the OpenMP runtime's captured
environment output to confirm those binding settings. Use six
threads and the same 128 warm tokens and four 512-token repetitions,
scoring the median of repetitions 2-4. Run three paired cycles in
small/large, large/small, small/large order. Verify the same binary
and source hashes, rerun the 73,024-check self-test, and require
identical checksum sequences within each E across all three runs.

Within each size require max/min median <=1.10. If either size
fails, mark the CPU ratio inconclusive. If repeatable, require every
paired E128,000/E12,800 ratio <=1.25 and every large-pool median
<=5,000 µs/token to pass the synthetic component gate. Report all
repetitions, environment output, process times, memory and ratios.
This does not measure full model speed or demonstrate that the
larger expert set has more useful functions.

Require >=64 GiB available RAM before each E128,000 process.
Cap each run at 15 minutes and outputs at 1 GB. Stop on hash,
self-test, binding confirmation, checksum, memory or process
failure. No T4 is authorized or relevant.
