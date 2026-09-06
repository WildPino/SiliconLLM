#!/usr/bin/env bash
# E3 arms -- strictly sequential: two timers must never overlap.
set -e
R=../results/e3/arms.json
rm -f "$R"
for B in 300 800; do
  for S in S05 S15 S3 M7 Q8 T10; do
    python e3_bench.py --weights D:/_ktmp/e3/${S}_th.bin --bench $B --reps 3 \
        --label "${S}_b${B}" --out "$R"
  done
done
echo "ARMS DONE"
