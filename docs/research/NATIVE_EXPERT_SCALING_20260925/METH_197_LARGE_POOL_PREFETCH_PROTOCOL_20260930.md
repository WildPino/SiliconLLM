# METH-197: selected-expert prefetch at 10× resident pool size

METH-177 used the real `engine.c` rank-8 LUT kernel on synthetic
Qwen-shaped expert factors with 12,800 versus 128,000 experts, 24
layers and top-4. Its six-thread median grew from 429.94 to
562.92 µs/token (1.309×), missing the frozen <=1.25 scaling gate.
The absolute cost was small, but a large n should not silently worsen
CPU routing/selected-factor access. Test one memory-latency
intervention without changing selected IDs, LUT arithmetic, factors,
or bytes/token.

Add an opt-in benchmark-only `--rank8-prefetch` flag. After building
the A-input LUT and before processing the four selected experts of
each layer, issue a read prefetch for the first cache line of each
selected A factor and B factor. Keep the existing kernel call and
route order. Benchmark both flag off/on in the **same newly compiled
binary** at E=12,800 and E=128,000 and threads=1,6. Use the same
METH-177 128 warm tokens, four 512-token timed repetitions, and
median of repetitions 2-4. Run the built-in 73,024-check LUT
self-test first. The baseline and prefetch arms must have identical
checksum sequences per E/thread and across thread counts. Keep
METH-177's 1,720,320 selected code bytes/token fixed.

Promotion of this prefetch idea requires six-thread E128,000/E12,800
median ratio <=1.25, E128,000 median <=5,000 µs/token, and
E12,800 prefetch median <=1.10× its same-binary baseline. Report
both sizes' absolute medians and all repetitions, regardless of
gate. These are synthetic selected-factor timings, not full-model
accepted-token rate or learned route quality.

Before each E128,000 allocation require >=64 GiB available RAM.
Cap each process at 15 minutes and all logs/results at 1 GB.
Stop on self-test, build, checksum or memory failure. No T4 is
authorized or useful for this CPU experiment.
