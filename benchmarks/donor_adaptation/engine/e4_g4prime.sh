#!/usr/bin/env bash
# E4 G4' -- run E3's OWN binary (git d7977c8^, rebuilt, logits bit-identical) in THIS session,
# against serial_e3.  If the old binary is also ~5% faster than E3's published table, the drift is
# the machine between sessions and not the code; if it reproduces E3, the code is the cause.
set -e
S="$1"
R=../results/e4/g4prime.json
rm -f "$R"
for B in 300 800; do
  for W in T10 S05; do
    python e3_bench.py --weights D:/_ktmp/e3/${W}_th.bin --bench $B --reps 3         --exe "$S/engine_e3.exe" --label "${W}_e3binary_b${B}" --out "$R"
    python e3_bench.py --weights D:/_ktmp/e3/${W}_th.bin --bench $B --reps 3         --attn serial_e3 --label "${W}_serial_e3_b${B}" --out "$R"
  done
done
echo "G4PRIME DONE"
