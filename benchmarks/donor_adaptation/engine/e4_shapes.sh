#!/usr/bin/env bash
# E4 -> the donor screen.  INDEX s7's re-run fits f = A*L*NH*HD*pos + B*L*D on E3's twelve points,
# and E4 halved the attention term, so A is now wrong by ~2x.  Refitting on SCALED numbers would be
# a guess; this re-measures all six shapes under --attn avx4 in ONE sweep so the fit stays a
# measurement.  One sweep also matters because between-sweep dispersion here is 5-10% (E4 s2.4).
set -e
R=../results/e4/shapes_avx4.json
rm -f "$R"
for B in 300 800; do
  for S in S05 S15 S3 M7 Q8 T10; do
    python e3_bench.py --weights D:/_ktmp/e3/${S}_th.bin --bench $B --reps 3         --attn avx4 --label "${S}_b${B}" --out "$R"
  done
done
echo "SHAPES DONE"
