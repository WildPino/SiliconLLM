#!/usr/bin/env bash
# E5 run 2 -- BRIEF_E5_DECOMPOSE_R.md s8.4, pushed before this file ran.  Strictly sequential.
#
# Run 1 was ARM-MAJOR: three repetitions of an arm back to back, then the next arm.  A contention
# episode therefore landed entirely inside one arm and presented itself as that arm's result -- the
# weight path, which no E5 arm touches, moved +30.4% across the T10 @800 cell and three arms came
# back with NEGATIVE components.  Run 2 is REP-MAJOR: one repetition of every arm, three times over,
# so an episode is spread across arms and G0 can discard the individual measurements it touched
# instead of voiding the cell.
set -e
R=../results/e5/arms2.json
rm -f "$R"
W5=D:/_ktmp/e3/S05_th.bin
WT=D:/_ktmp/e3/T10_th.bin

one() {  # one <weights> <shape> <bench> <attn> <attnr> <rep>
  python e3_bench.py --weights "$1" --bench "$3" --reps 1 --threads 6 \
      --attn "$4" --attnr "$5" --label "$2_$4_$5_b$3_r$6" --out "$R"
}

pass() {  # pass <weights> <shape> <bench> <rep>
  one "$1" "$2" "$3" serial  none "$4"
  one "$1" "$2" "$3" serial2 none "$4"
  for A in none sm2 sm3 av2 av3 fork2; do one "$1" "$2" "$3" avx4 "$A" "$4"; done
}

for REP in 1 2 3; do
  for B in 300 800; do pass "$W5" S05 "$B" "$REP"; done
  for B in 300 800; do pass "$WT" T10 "$B" "$REP"; done
  echo "PASS $REP DONE"
done
echo "E5 RUN2 DONE"
