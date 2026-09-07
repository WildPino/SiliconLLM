#!/bin/bash
# E8 G-Z1 / G-Z3 / G-Z5: pre-registered at brief section 4 (pushed 406b02b).
if tasklist //FI "IMAGENAME eq donor_engine_e8.exe" 2>/dev/null | grep -qi donor_engine; then echo "ABORT: an engine process is already running"; exit 1; fi
cd /d/_THINGS/Progetti/SiliconLLM/benchmarks/donor_adaptation/engine
W=D:/_ktmp/e1/qwen25-05b_tqh.bin
F32=D:/_ktmp/e7/ctl_f32load.bin
P7=D:/_ktmp/e7/qwen25-coder7b_p.bin
E=./donor_engine_e8.exe

echo "=== G-Z1  the FFN decomposed -- 0.5B packed, --profile --bench 300"
$E --weights $W --threads 6 --bench 300 --profile --mvacc 1 2>/dev/null

echo "=== G-Z1  the FFN decomposed -- Coder-7B packed, --profile --bench 100"
$E --weights $P7 --threads 6 --bench 100 --profile --mvacc 1 2>/dev/null

echo "=== G-Z3  10 interleaved triples, 0.5B packed, --bench 300"
for k in $(seq 1 10); do
  a=$($E --weights $W --threads 6 --bench 300 --mvacc 1 2>/dev/null | grep BENCH)
  b=$($E --weights $W --threads 6 --bench 300 --mvacc 2 2>/dev/null | grep BENCH)
  c=$($E --weights $W --threads 6 --bench 300 --mvacc 4 2>/dev/null | grep BENCH)
  echo "trip $k m1 $a"
  echo "trip $k m2 $b"
  echo "trip $k m4 $c"
done

echo "=== G-Z5  NEGATIVE CONTROL -- 8 interleaved pairs, 0.5B fp32, --bench 300"
echo "    predicted 0.99-1.03; >=1.10 refutes the mechanism in brief section 2"
for k in $(seq 1 8); do
  a=$($E --weights $F32 --threads 6 --bench 300 --mvacc 1 2>/dev/null | grep BENCH)
  b=$($E --weights $F32 --threads 6 --bench 300 --mvacc 4 2>/dev/null | grep BENCH)
  echo "pair $k f32m1 $a"
  echo "pair $k f32m4 $b"
done
echo "=== done"
