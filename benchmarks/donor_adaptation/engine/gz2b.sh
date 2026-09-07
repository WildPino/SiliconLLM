#!/bin/bash
# E8 G-Z2b, re-run. The first attempt handed --generate the whole 12,288-token corpus ids
# file as a "prompt", which prefilled all of it and dumped 7.5 GB of prefill logits per arm.
# Operator error, recorded in the probe. Same gate, 32-token prompt, 160 new tokens.
cd /d/_THINGS/Progetti/SiliconLLM/benchmarks/donor_adaptation/engine
F32=D:/_ktmp/e7/ctl_f32load.bin
IDS=D:/_ktmp/e8/prompt32.ids.bin
E=./donor_engine_e8.exe
O=D:/_ktmp/e8

echo "=== G-Z2b  greedy generation, 0.5B fp32, 32-token prompt, 160 new tokens"
for m in 1 2 4; do
  rm -f $O/z2b_m$m $O/z2b_m$m.ids.bin $O/z2b_m$m.prefill.bin
  $E --weights $F32 --threads 6 --generate $IDS 160 $O/z2b_m$m --mvacc $m >/dev/null 2>&1
  echo "  mvacc $m rc=$?"
done
ls -la $O/z2b_m*
echo "-- generated token sequences (prompt+160), sha256:"
sha256sum $O/z2b_m1.ids.bin $O/z2b_m2.ids.bin $O/z2b_m4.ids.bin
echo "=== done"
