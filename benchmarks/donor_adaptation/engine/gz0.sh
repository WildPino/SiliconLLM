#!/bin/bash
# E8 G-Z0 and G-Z2: pre-registered at brief sections 4 (pushed 406b02b).
cd /d/_THINGS/Progetti/SiliconLLM/benchmarks/donor_adaptation/engine
W=D:/_ktmp/e1/qwen25-05b_tqh.bin
F32=D:/_ktmp/e7/ctl_f32load.bin
IDS=D:/_ktmp/e1/ids_qwen25-05b_tqh.bin
OLD=./donor_engine.exe
NEW=./donor_engine_e8.exe
O=D:/_ktmp/e8

echo "=== G-Z0a  logits sha256, E8 --mvacc 1 vs pre-E8 binary"
$OLD --weights $W --threads 6 --logits $IDS 8 $O/z0_old.bin           >/dev/null 2>&1; echo "  old rc=$?"
$NEW --weights $W --threads 6 --logits $IDS 8 $O/z0_new.bin --mvacc 1 >/dev/null 2>&1; echo "  new rc=$?"
ls -la $O/z0_old.bin $O/z0_new.bin
sha256sum $O/z0_old.bin $O/z0_new.bin

echo "=== G-Z0b  8 interleaved pairs, 0.5B packed, --bench 300"
for k in $(seq 1 8); do
  a=$($NEW --weights $W --threads 6 --bench 300 --mvacc 1 2>/dev/null | grep BENCH)
  b=$($OLD --weights $W --threads 6 --bench 300           2>/dev/null | grep BENCH)
  echo "pair $k E8   $a"
  echo "pair $k PRE  $b"
done

echo "=== G-Z2a  parity of the accumulator arms, 0.5B fp32, --logits 8"
for m in 1 2 4; do
  $NEW --weights $F32 --threads 6 --logits $IDS 8 $O/z2_m$m.bin --mvacc $m >/dev/null 2>&1
  echo "  mvacc $m rc=$?"
done
ls -la $O/z2_m*.bin

echo "=== G-Z2b  greedy generation, 0.5B fp32, 160 new tokens"
for m in 1 2 4; do
  $NEW --weights $F32 --threads 6 --generate $IDS 160 $O/z2g_m$m --mvacc $m >/dev/null 2>&1
  echo "  mvacc $m rc=$?"
done
ls -la $O/z2g_m*
sha256sum $O/z2g_m1 $O/z2g_m2 $O/z2g_m4 2>/dev/null
echo "=== done"
