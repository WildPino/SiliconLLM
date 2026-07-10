#!/usr/bin/env bash
# Phase 64.1b — two CPU microbenches (synthetic, NO weights, NO gate) that tighten the 64.1 budget model.
#   One command, portable: `bash benchmarks/phase64/bench_64_1b.sh`. Same build policy as 63.T/64.0 (generic x86-64-v3,
#   -ffp-contract=on, NO -march; znver2 = optional TUNE=-march=znver2). Pinning is external (default close/cores);
#   report it with the numbers. Push-before-run: this apparatus + brief are pushed BEFORE the registered run.
#
#   (1) proj-GEMV size sweep  — real row-partitioned fp32 matvec over synthetic weights {4..96} MB, threads {1,6}.
#       Validates the spill model: where the t1 curve breaks (CCX L3 = 16 MB) and whether the t6 plateau holds past L3.
#   (2) expert-pool DRAM rate — real ternary LUT kernel over a 512 MB pool (>> L3) of 48 KB experts, 8 random x 6 layers
#       per token (i.i.d. gather), threads {1,6}. Tightens the model's widest bracket [4.2-11.4 GB/s].
set -u
cd "$(dirname "$0")/../.."
CC=${CC:-clang}
BASE="-O3 -mavx2 -mfma -ffp-contract=on"
TUNE=${TUNE:-}
export OMP_PLACES=${OMP_PLACES:-cores} OMP_PROC_BIND=${OMP_PROC_BIND:-close}
mkdir -p bin
echo "== build: $CC $BASE $TUNE -fopenmp | pinning: OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES =="
$CC $BASE $TUNE -fopenmp benchmarks/phase60/engine.c -o bin/engine.exe -lm || { echo "BUILD FAIL"; exit 1; }
echo "== kernel selftest =="; bin/engine.exe --kselftest 2>&1 | tail -2
echo
echo "== 64.1b(1) proj-GEMV size sweep (real fp32 matvec; validates spill-slope + L3 break) =="
bin/engine.exe --gemv-sweep 2>/dev/null
echo
echo "== 64.1b(2) expert-pool DRAM rate (real LUT kernel; tightens [4.2-11.4 GB/s]) =="
bin/engine.exe --expert-rate 2>/dev/null
echo
echo "STOP. 64.1b microbenches above — no gates. Tightens the 64.1 budget model. No commit."
