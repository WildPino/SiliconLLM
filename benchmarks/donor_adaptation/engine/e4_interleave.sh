#!/usr/bin/env bash
# E4 s9.2 -- A/B/A/B/A/B at T10 @800.  One rep per invocation so that each adjacent pair straddles
# the same few minutes: this machine's BETWEEN-sweep drift is 5-7% (s9.1) and differencing inside a
# pair is the only way to read a ratio to better than that.
set -e
R=../results/e4/interleave.json
rm -f "$R"
for K in 1 2 3; do
  python e3_bench.py --weights D:/_ktmp/e3/T10_th.bin --bench 800 --reps 1       --attn serial_e3 --label "pair${K}_A_serial_e3" --out "$R"
  python e3_bench.py --weights D:/_ktmp/e3/T10_th.bin --bench 800 --reps 1       --attn avx4 --label "pair${K}_B_avx4" --out "$R"
done
echo "INTERLEAVE DONE"
