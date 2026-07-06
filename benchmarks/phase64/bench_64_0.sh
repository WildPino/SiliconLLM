#!/usr/bin/env bash
# Phase 64.0 — ground-truth refresh (instrumentation only, NO gates). Feeds the 64.1 budget document.
#   One command, portable: `bash benchmarks/phase64/bench_64_0.sh`. Same build policy as 63.T (generic x86-64-v3,
#   -ffp-contract=on, NO -march; znver2 = optional TUNE=-march=znver2). Per §8 push-before-run: this apparatus + the
#   §8 protocol are pushed BEFORE this is run for the record — the numbers below feed 64.1, so their provenance is dated.
#
#   (a) Streamed thread ceiling  — AR-cold emulation buffer, parallel cold-stream read at threads {1,2,3,6} -> GB/s.
#       Buffer MUST be >> L3 (default --emu-mb 128) or threads>1 reads L3-resident bandwidth, not the DRAM ceiling.
#   (b) Compute-floor decomposition — per-component us/tok (scan-recur / proj-GEMV / SWA-attn / LUT-MLP / head / norms
#       / glue), dense + MoE, threads {1,6}, end-to-end. Anchor for extrapolating C(model dims) in 64.1.
set -u
cd "$(dirname "$0")/../.."
CC=${CC:-clang}
BASE="-O3 -mavx2 -mfma -ffp-contract=on"
TUNE=${TUNE:-}
E1=results/phase60/e1_model.bin; E4=results/phase60/e4_model.bin
NG=results/phase63/ngram_ts_N5.bin
EMU_MB=${EMU_MB:-128}; TT=${TT:-4000}
mkdir -p bin
echo "== build: $CC $BASE $TUNE -fopenmp =="
$CC $BASE $TUNE -fopenmp benchmarks/phase60/engine.c -o bin/engine.exe -lm || { echo "BUILD FAIL"; exit 1; }
echo "== kernel selftest =="; bin/engine.exe --kselftest 2>&1 | tail -2

if [ ! -f "$E1" ]; then echo "== weights absent -> see REPRODUCE.md; STOP =="; exit 0; fi

echo
echo "== 64.0(a) streamed-thread bandwidth ceiling (>> L3 buffer = ${EMU_MB} MB; DRAM cold-stream GB/s per N) =="
if [ -f "$NG" ]; then
  bin/engine.exe --threads 1 --weights "$E1" --g3c --ngram "$NG" --emu-mb "$EMU_MB" 2>/dev/null | grep -E "64.0\(a\)|threads="
else
  echo "  n-gram asset absent ($NG) -> build it: python benchmarks/phase63/ngram_asset.py --N 5"
fi

echo
echo "== 64.0(b) compute-floor decomposition (us/tok per component; dense + MoE; threads {1,6}) =="
for cfg in "dense|$E1|--mlp lut --skip on --exp fast" "MoE|$E4|--mlp lut --exp fast"; do
  nm="${cfg%%|*}"; r="${cfg#*|}"; w="${r%%|*}"; fl="${r##*|}"
  [ -f "$w" ] || { echo "  $nm weights absent ($w) -> skip"; continue; }
  echo "  --- $nm ---"
  for TH in 1 6; do
    printf "    threads=%s: " "$TH"
    bin/engine.exe --threads $TH --weights "$w" $fl --timing --time-tok $TT 2>/dev/null | grep -E "scan-recur"
  done
done

echo
echo "STOP. 64.0 (a)+(b) instrumentation above — no gates. Feeds the 64.1 budget document. No commit."
