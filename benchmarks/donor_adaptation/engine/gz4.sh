#!/bin/bash
# E8 G-Z4, the headline: pre-registered at brief section 4 (pushed 406b02b).
# ARM2 is passed in: whichever of {2,4} won G-Z3.
if tasklist //FI "IMAGENAME eq donor_engine_e8.exe" 2>/dev/null | grep -qi donor_engine; then echo "ABORT: engine already running"; exit 1; fi
cd /d/_THINGS/Progetti/SiliconLLM/benchmarks/donor_adaptation/engine
P7=D:/_ktmp/e7/qwen25-coder7b_p.bin
E=./donor_engine_e8.exe
ARM2=${1:-4}

echo "=== G-Z4  10 interleaved pairs, Coder-7B packed, --bench 300, mvacc 1 vs $ARM2"
echo "    bands fixed before the run: >=1.25 CONFIRMED, 1.06-1.25 UNDECIDED, <=1.06 REFUTED"
for k in $(seq 1 10); do
  a=$($E --weights $P7 --threads 6 --bench 300 --mvacc 1     2>/dev/null | grep BENCH)
  b=$($E --weights $P7 --threads 6 --bench 300 --mvacc $ARM2 2>/dev/null | grep BENCH)
  echo "pair $k m1     $a"
  echo "pair $k m$ARM2 $b"
done
echo "=== done"
