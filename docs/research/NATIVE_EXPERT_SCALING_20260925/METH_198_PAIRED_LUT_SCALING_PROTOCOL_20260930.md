# METH-198: paired repeatability audit for the 10× LUT pool ratio

The no-prefetch six-thread E128,000/E12,800 median ratio was 1.309
in METH-177 and 0.917 in METH-197. Audit the difference before
claiming either a scaling pass or failure. Use the exact METH-197
C binary/source SHA and the default no-prefetch `--rank8-lut-pool`
path. Do not change the LUT kernel, route generator, selected-byte
count, or host settings during this audit. The same binary already
passed the 73,024-check self-test and checksum parity across threads
and prefetch modes; bind that result and repeat its self-test.

Run three six-thread pairs, each with the original 128 warm tokens,
four 512-token timed repetitions and median of repetitions 2-4:
small/large, large/small, small/large. Record all repetitions,
checksum sequences, available memory and process elapsed time.
All three small and all three large checksum sequences must match.
Within each pool size, require max/min of the three run medians
<=1.10 for timing repeatability. If that fails, label the ratio
**inconclusive** regardless of its value. If repeatability holds,
apply the frozen METH-177 large/small <=1.25 ratio gate to every
paired ratio; only all three passing constitutes a robust component
scaling pass. Also require every large-pool median <=5,000 µs/token.
Report each ratio and the median ratio. This is still synthetic
selected-factor timing, not full C accepted-token rate or evidence
that routes to more experts retain quality.

Before every E128,000 run require >=64 GiB available RAM. Cap each
process at 15 minutes and all outputs at 1 GB. Stop on binary/source
binding, self-test, checksum, allocation or process failure. No T4
is authorized or relevant.
