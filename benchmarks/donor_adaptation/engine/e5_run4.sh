#!/usr/bin/env bash
# E5 run 4 -- BRIEF_E5_DECOMPOSE_R.md s10, pushed before this file ran.  Strictly sequential.
#
# Two things changed after run 3, and only one of them is about noise.
#
#   1. THE ARMS.  Run 3 solved each component as (2x - none) and tested it at 3x.  Both
#      (av2 - none) and (av3 - av2) are "one extra A.V pass" and they disagreed by 1.28x at
#      T10 @800 and 1.71x at S05 @300, because `none` runs a DIFFERENT code block from the
#      wrapped arms.  sm1/av1 are the 1x point INSIDE the wrapped path: the solve and the 3x
#      test now share a code shape, and (1x - none) prices the code-path difference itself.
#
#   2. THE MACHINE.  Run 3 passed G0 at 5% everywhere and still failed G1, because 5% of the
#      weight path is bigger than the components being estimated.  G0 tightens to 1%, the
#      cores_busy witness of s9.3 is promoted to a gate at +0.30 cores, and every arm runs at
#      HIGH_PRIORITY_CLASS -- uniformly, so it cancels in every within-sweep difference.
set -e
R=../results/e5/arms4.json
rm -f "$R"
W5=D:/_ktmp/e3/S05_th.bin
WT=D:/_ktmp/e3/T10_th.bin

one() {  # one <weights> <shape> <bench> <attn> <attnr> <rep>
  python e3_bench.py --weights "$1" --bench "$3" --reps 1 --threads 6 --prio high \
      --attn "$4" --attnr "$5" --label "$2_$4_$5_b$3_r$6" --out "$R"
}

pass() {  # pass <weights> <shape> <bench> <rep>
  one "$1" "$2" "$3" serial  none "$4"
  one "$1" "$2" "$3" serial2 none "$4"
  for A in none sm1 sm2 sm3 av1 av2 av3 fork2; do one "$1" "$2" "$3" avx4 "$A" "$4"; done
}

for REP in 1 2 3 4 5 6; do
  for B in 300 800; do pass "$W5" S05 "$B" "$REP"; done
  for B in 300 800; do pass "$WT" T10 "$B" "$REP"; done
  echo "PASS $REP DONE"
done
echo "E5 RUN4 DONE"
