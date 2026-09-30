# METH-200: single-thread control for large-n LUT timing variability

METH-198 and METH-199 could not produce repeatable six-thread
E128,000/E12,800 selected-LUT ratios; the small pool varied by
1.423× and 1.589× across three runs, respectively. Test the exact
same METH-197 C binary's **one-thread, no-prefetch** path to isolate
whether OpenMP is implicated. Do not change routes, selected factor
format, LUT arithmetic, or pool contents. Reverify the binary/source
hashes and 73,024-check self-test.

Run three pairs in small/large, large/small, small/large order. Each
process uses 128 warm tokens, four 512-token timed repetitions,
and the median of repetitions 2-4. Require identical checksum
sequences among runs of the same E. Require max/min of the three
run medians <=1.10 within each pool size. If either fails, report
single-thread scaling inconclusive. If repeatable, require all
paired E128,000/E12,800 ratios <=1.25 and all large-pool medians
<=5,000 µs/token for a **single-thread component pass**; otherwise
report component failure. Neither outcome can establish the
six-thread or full-model accepted-token gate.

Require >=64 GiB available RAM before each large process. Cap each
process at 15 minutes and logs/results at 1 GB. Stop on binding,
self-test, checksum, resource or process failure. No T4 is used.
