#!/bin/bash
# E9 G-G1 / G-G2 / G-G3: pre-registered at brief section 4 (pushed 82dbb54).
if tasklist //FI "IMAGENAME eq donor_engine_e9.exe" 2>/dev/null | grep -qi donor_engine; then echo "ABORT: engine already running"; exit 1; fi
cd /d/_THINGS/Progetti/SiliconLLM/benchmarks/donor_adaptation/engine
W=D:/_ktmp/e1/qwen25-05b_tqh.bin
IDS=D:/_ktmp/e1/ids_qwen25-05b_tqh.bin
P7=D:/_ktmp/e7/qwen25-coder7b_p.bin
OLD=./donor_engine.exe
NEW=./donor_engine_e9.exe
O=D:/_ktmp/e8

echo "=== G-G1  sha256, before vs after.  Gate: BYTE-IDENTICAL, not small."
$OLD --weights $W --threads 6 --logits $IDS 8 $O/g1_pre.bin  >/dev/null 2>&1; echo "  pre  rc=$?"
$NEW --weights $W --threads 6 --logits $IDS 8 $O/g1_post.bin >/dev/null 2>&1; echo "  post rc=$?"
sha256sum $O/g1_pre.bin $O/g1_post.bin

echo "=== G-G3  profile, Coder-7B packed, --bench 100"
echo "-- before:"; $OLD --weights $P7 --threads 6 --bench 100 --profile 2>/dev/null | grep -E "BENCH|glue|ffn|FFN-SUM|TOTAL"
echo "-- after:";  $NEW --weights $P7 --threads 6 --bench 100 --profile 2>/dev/null | grep -E "BENCH|glue|ffn|FFN-SUM|TOTAL"

echo "=== G-G2  10 interleaved pairs, Coder-7B packed, --bench 300"
echo "    bands fixed before the run: >=1.035 CONFIRMED, 1.015-1.035 UNDECIDED, <=1.015 REFUTED"
for k in $(seq 1 10); do
  a=$($OLD --weights $P7 --threads 6 --bench 300 2>/dev/null | grep BENCH)
  b=$($NEW --weights $P7 --threads 6 --bench 300 2>/dev/null | grep BENCH)
  echo "pair $k pre  $a"
  echo "pair $k post $b"
done
echo "=== done"
