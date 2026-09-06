#!/usr/bin/env bash
# E5 -- BRIEF_E5_DECOMPOSE_R.md, pushed at c1bdd70 before any arm existed.  Strictly sequential:
# two timers must never overlap (feedback_perf_parallelization -- a contended timing is not a
# timing), and G3 requires ALL arms in ONE sweep because between-sweep drift here is 5-10%.
#
# The sweep carries E4's `serial` and `serial2` as well as E5's own arms.  That is not padding:
# R is what E5 decomposes, and R can only be had as `organ(serial) - (organ(serial2)-organ(serial))`.
# Taking R from E4's published 9.816 instead would put a number from another sweep into the
# denominator of every share E5 reports -- exactly the mistake G3 exists to prevent.  It also
# re-derives E4's central split inside this sweep, for free.
set -e
R=../results/e5/arms.json
rm -f "$R"
W5=D:/_ktmp/e3/S05_th.bin
WT=D:/_ktmp/e3/T10_th.bin

run() {  # run <weights> <shape> <bench> <attn> <attnr>
  python e3_bench.py --weights "$1" --bench "$3" --reps 3 --threads 6 \
      --attn "$4" --attnr "$5" --label "$2_$4_$5_b$3" --out "$R"
}

# S05 first (HD=64, cheap), then T10 (the judging point), both contexts.
for B in 300 800; do
  run "$W5" S05 "$B" serial  none
  run "$W5" S05 "$B" serial2 none
  for A in none sm2 sm3 av2 av3 fork2; do run "$W5" S05 "$B" avx4 "$A"; done
done

for B in 300 800; do
  run "$WT" T10 "$B" serial  none
  run "$WT" T10 "$B" serial2 none
  for A in none sm2 sm3 av2 av3 fork2; do run "$WT" T10 "$B" avx4 "$A"; done
done
echo "E5 RUN DONE"
