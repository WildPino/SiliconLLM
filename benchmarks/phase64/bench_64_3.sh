#!/usr/bin/env bash
# Phase 64.3 — recall-tier engine-side smoke + parity control. The one 64.2 probe (PHASE64_DECISIONS.md §4).
#   One command, portable: `bash benchmarks/phase64/bench_64_3.sh`. Same build policy as 63.T/64.0/64.1b (generic
#   x86-64-v3, -ffp-contract=on, NO -march; znver2 = optional TUNE=-march=znver2). Pinning external (default close/cores).
#   Gate (sealed §4): two-stage query <= 50 us/token @128K on the reference AND zero engine perturbation when disabled.
#
#   (a) recall two-stage query us/token at threads {1,6} (omp build) + serial-native baseline (no-omp: isolates the
#       omp nt=1 tax, cf. 63.C). (b) codes+index / value footprint. (c) parity control: the probe is STANDALONE
#       (engine.c untouched) -> P43 golden + kselftest must be bit-identical == zero perturbation, by construction.
set -u
cd "$(dirname "$0")/../.."
CC=${CC:-clang}
BASE="-O3 -mavx2 -mfma -ffp-contract=on"
TUNE=${TUNE:-}
export OMP_PLACES=${OMP_PLACES:-cores} OMP_PROC_BIND=${OMP_PROC_BIND:-close}
E1=results/phase60/e1_model.bin
mkdir -p bin
echo "== build recall probe: $CC $BASE $TUNE -fopenmp | pinning: $OMP_PROC_BIND/$OMP_PLACES =="
$CC $BASE $TUNE -fopenmp benchmarks/phase64/recall_probe.c -o bin/recall_probe.exe -lm || { echo "BUILD FAIL"; exit 1; }

echo
echo "== 64.3(a,b) recall two-stage query (IVF+Hadamard / 4-bit ADC / exact top-16 rerank @128K) =="
echo "   note: t6 is the operating figure (engine runs threaded, D9); t1 carries the omp nt=1 tax + single-thread"
echo "   DRAM-random rerank latency (higher variance). best-of-3 per row."
bin/recall_probe.exe 2>/dev/null

echo
echo "== 64.3(c) parity control: engine.c UNTOUCHED by the probe -> zero perturbation must be bit-exact =="
$CC $BASE $TUNE -fopenmp benchmarks/phase60/engine.c -o bin/engine.exe -lm || { echo "BUILD FAIL"; exit 1; }
bin/engine.exe --kselftest 2>&1 | tail -2
if [ -f "$E1" ]; then
  T="${TMPDIR:-/tmp}/b643"; mkdir -p "$T"
  bin/engine.exe --threads 1 --weights "$E1" --mlp lut --skip on --exp fast --dumplogits "$T/e.bin" --ntok 2048 --seq 512 >/dev/null 2>&1
  m=$(md5sum "$T/e.bin"|cut -d' ' -f1)
  [ "$m" = "f395980c803e6e5358e0fd67dc62f432" ] && echo "  P43 golden (E35full): BIT-IDENTICAL ($m) -> zero perturbation" || echo "  P43 golden MISMATCH: $m"
else echo "  (E1 weights absent -> parity control skipped; probe is standalone so engine is unchanged by construction)"; fi

echo
echo "STOP. 64.3 recall smoke + parity control above. gate: <=50 us/token @128K (operating t6) + zero perturbation. No commit."
