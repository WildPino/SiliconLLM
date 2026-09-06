#!/usr/bin/env bash
# E4 run 2 -- BRIEF_E4_ATTENTION_ACCUMULATORS.md s8.3.  Strictly sequential.
# EVERY arm is re-measured under THIS binary, including the ones run 1 already timed: run 1's
# defect was a comparison across two different pieces of code, and the fix is not to repeat it.
set -e
R=../results/e4/arms2.json
rm -f "$R"
ARMS="serial_e3 serial serial2 serial3 ilp4 avx1 avx4"

# S05 first: HD=64, cheap, and the shape SPEED_LEDGER was written on.
for B in 300 800; do
  for A in $ARMS; do
    python e3_bench.py --weights D:/_ktmp/e3/S05_th.bin --bench $B --reps 3         --attn $A --label "S05_${A}_b${B}" --out "$R"
  done
done

# T10: the judging point.
for B in 300 800; do
  for A in $ARMS; do
    python e3_bench.py --weights D:/_ktmp/e3/T10_th.bin --bench $B --reps 3         --attn $A --label "T10_${A}_b${B}" --out "$R"
  done
done
echo "RUN2 DONE"
