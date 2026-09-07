#!/usr/bin/env bash
# E5 run 6 -- BRIEF_E5_DECOMPOSE_R.md s13.4, pushed before this file ran.  Strictly sequential.
#
# Run 5 measured S and Y and refuted its own X: +0.289 ms at T10 @800 and -0.287 at @300, and a
# loop cannot take negative time.  R came from 2*organ(serial) - organ(serial2), which cancels the
# Q.K term exactly and lets the whole per-arm cold-start cost land in R; X = organ(none) - R then
# carried d_avx4 - d_serial.  Two changes:
#
#   1. qk1/qk2/qk3 -- the Q.K loop at one, two and three passes ON THE AVX4 PATH.  X is now a
#      difference inside one code family, with its own 3x test, and serial/serial2 leave the sweep.
#   2. --sweepd -- three of its four entries run the IDENTICAL `none` code and differ only in their
#      neighbours (isolated between strangers, or inside a run of its own kind), at equal mean
#      context length.  d = organ(none_iso) - organ(none_hot) is the price of switching arms,
#      measured rather than assumed.
set -e
R=../results/e5/sweep6.json
rm -f "$R"
W5=D:/_ktmp/e3/S05_th.bin
WT=D:/_ktmp/e3/T10_th.bin

one() {  # one <weights> <shape> <bench> <run> <mode>
  python e5_sweep.py --weights "$1" --shape "$2" --bench "$3" --run "$4" --mode "$5" \
      --threads 6 --prio high --out "$R"
}

for RUN in 1 2 3 4 5; do
  for M in sweep6 sweepd; do
    for B in 300 800; do one "$W5" S05 "$B" "$RUN" "$M"; done
    for B in 300 800; do one "$WT" T10 "$B" "$RUN" "$M"; done
  done
  echo "RUN $RUN DONE"
done
echo "E5 RUN6 DONE"
