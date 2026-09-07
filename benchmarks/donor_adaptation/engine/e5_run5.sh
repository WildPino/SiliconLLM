#!/usr/bin/env bash
# E5 run 5 -- BRIEF_E5_DECOMPOSE_R.md s11, pushed before this file ran.  Strictly sequential.
#
# Runs 1-4 compared arms that lived in DIFFERENT PROCESSES, minutes apart.  At T10 @800 the weight
# path is 301 ms, so G0 at 1% admits a 3.0 ms excursion while the component being estimated, Y, is
# 1.9 ms: no between-process tolerance can be both protective and passable, and all four cells of
# run 4 came back VOID at the gate fixed before it ran.
#
# Here the ten arms are interleaved PER TOKEN inside ONE process, under a palindrome schedule of
# period 2n so every arm's mean context length is exactly equal.  Five independent processes per
# cell: the arm differences inside a process are the measurement, and the spread of those
# differences across the five is the reproducibility interval.
set -e
R=../results/e5/sweep5.json
rm -f "$R"
W5=D:/_ktmp/e3/S05_th.bin
WT=D:/_ktmp/e3/T10_th.bin

one() {  # one <weights> <shape> <bench> <run>
  python e5_sweep.py --weights "$1" --shape "$2" --bench "$3" --run "$4" \
      --threads 6 --prio high --out "$R"
}

for RUN in 1 2 3 4 5; do
  for B in 300 800; do one "$W5" S05 "$B" "$RUN"; done
  for B in 300 800; do one "$WT" T10 "$B" "$RUN"; done
  echo "RUN $RUN DONE"
done
echo "E5 RUN5 DONE"
