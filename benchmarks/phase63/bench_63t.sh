#!/usr/bin/env bash
# Phase 63.T — one-command portable bench: build baseline → selftest → 7/7 parity → T-G2 tok/s table.
#   The portable artifact: on any AVX2+FMA machine (or a rented box), re-run the whole oracle-chain + speedup
#   protocol identically with `bash benchmarks/phase63/bench_63t.sh`. No hard-fit to the 3600X — thread partition
#   comes from --threads N + OpenMP; pinning is external (OMP_PROC_BIND/OMP_PLACES). Build policy per §Principles:
#   baseline = x86-64-v3 generic, -ffp-contract=on explicit, NO -march (znver2 is optional tuning, measured apart).
set -u
cd "$(dirname "$0")/../.."
CC=${CC:-clang}
BASE="-O3 -mavx2 -mfma -ffp-contract=on"          # generic baseline (portable); add TUNE=-march=znver2 to compare
TUNE=${TUNE:-}
E1=results/phase60/e1_model.bin; E4=results/phase60/e4_model.bin
IDS=results/phase55/ids.u16; META=results/phase55/meta.bin
mkdir -p bin
echo "== build: $CC $BASE $TUNE -fopenmp =="
$CC $BASE $TUNE -fopenmp benchmarks/phase60/engine.c -o bin/engine.exe -lm || { echo "BUILD FAIL"; exit 1; }
for e in e1_engine e2_engine e3_engine e35_engine e4_engine; do
  $CC $BASE $TUNE archive/benchmarks/phase60_stage_engines/$e.c -o bin/$e.exe -lm; done

echo "== S0.1 kernel selftest (no weights) =="
for e in engine e2_engine e3_engine e35_engine e4_engine; do printf "  %-10s " $e; bin/$e.exe --kselftest 2>&1 | tail -1; done

if [ ! -f "$E1" ] || [ ! -f "$IDS" ]; then echo "== weights/ids absent -> skipping parity + T-G2 (see REPRODUCE.md) =="; exit 0; fi

echo "== S0.2 7/7 parity vs P43 record (threads=1) =="
declare -A REC=( [E1]=db2413043d213673a45d025447f33f99 [E2]=d318292495282fcdea3a11524d7eb83a [E3]=d318292495282fcdea3a11524d7eb83a [E35iso]=237e0e49d4ac419f812df7a1d538073b [E35full]=f395980c803e6e5358e0fd67dc62f432 [E4ref]=834b3c09710f2fee0e112e9ace57de01 [E4full]=d3a722532eb8180252984065b50f80d0 )
declare -A CF=( [E1]="$E1|--mlp fp32 --skip off --exp exact" [E2]="$E1|--mlp lut --skip off --exp exact" [E3]="$E1|--mlp lut --skip on --exp exact" [E35iso]="$E1|--mlp fp32 --skip off --exp fast" [E35full]="$E1|--mlp lut --skip on --exp fast" [E4ref]="$E4|--mlp fp32 --exp exact" [E4full]="$E4|--mlp lut --exp fast" )
T="${TMPDIR:-/tmp}/b63t"; mkdir -p "$T"; pass=0
for k in E1 E2 E3 E35iso E35full E4ref E4full; do w="${CF[$k]%%|*}"; fl="${CF[$k]##*|}"
  bin/engine.exe --threads 1 --weights "$w" $fl --dumplogits "$T/$k.bin" --ntok 2048 --seq 512 >/dev/null 2>&1
  m=$(md5sum "$T/$k.bin"|cut -d' ' -f1); [ "$m" = "${REC[$k]}" ] && { pass=$((pass+1)); } || echo "  $k MISMATCH $m"; done
echo "  7/7 parity: $pass/7"

echo "== T-G1 bit-identity {2,3,6} vs single-thread (3 golden configs) =="
for name in E1 E35full E4full; do w="${CF[$name]%%|*}"; fl="${CF[$name]##*|}"; ref="${REC[$name]}"; ok=1
  for TH in 2 3 6; do bin/engine.exe --threads $TH --weights "$w" $fl --dumplogits "$T/${name}_$TH.bin" --ntok 2048 --seq 512 >/dev/null 2>&1
    [ "$(md5sum "$T/${name}_$TH.bin"|cut -d' ' -f1)" = "$ref" ] || ok=0; done
  echo "  $name: $([ $ok = 1 ] && echo BIT-IDENTICAL || echo FAIL)"; done

echo "== T-G2 tok/s (threads × pinning) =="
export OMP_PLACES=cores; TT=${TT:-5000}
tps(){ local b="$4"; [ "$b" = none ] && export OMP_PROC_BIND=false || export OMP_PROC_BIND=$b
  bin/engine.exe --threads $3 --weights "$1" $2 --timing --time-tok $TT 2>/dev/null | grep -oE "[0-9]+\.[0-9]+ tok/s" | grep -oE "^[0-9]+\.[0-9]+"; }
for cfg in "dense|$E1|--mlp lut --skip on --exp fast" "MoE|$E4|--mlp lut --exp fast"; do
  nm="${cfg%%|*}"; r="${cfg#*|}"; w="${r%%|*}"; fl="${r##*|}"
  base=$(tps "$w" "$fl" 1 none); echo "  --- $nm-full: t1=$base tok/s ---"; printf "    %-8s %-16s %-16s %-16s\n" threads no-pin close spread
  for TH in 2 3 6; do row=""; for b in none close spread; do v=$(tps "$w" "$fl" $TH $b); row+=$(printf "%-16s" "$v ($(python -c "print(f'{$v/$base:.2f}x')" 2>/dev/null))"); done; printf "    %-8s %s\n" $TH "$row"; done
  v=$(tps "$w" "$fl" 12 close); printf "    %-8s %s (SMT)\n" 12 "$v ($(python -c "print(f'{$v/$base:.2f}x')" 2>/dev/null))"
done
unset OMP_PROC_BIND OMP_PLACES

# ---- 63.V/63.C consolidated phase tables (block-verify chassis + emulation + accounting + t1 attribution) ----
NG=results/phase63/ngram_ts_N5.bin
if [ -f "$NG" ]; then
  echo "== 63.V block-verify: V-G1 identity + V-G2 tpp + V-G3a in-cache (dense, threads 1) =="
  bin/engine.exe --threads 1 --weights "$E1" --gen-verify --ngram "$NG" --block 8 --gen-len 800 --seeds 3 2>/dev/null | grep -E "V-G1|V-G2|V-G3a"
  echo "== 63.V V-G3b expert-union accounting (MoE) =="
  bin/engine.exe --threads 1 --weights "$E4" --g3b --ngram "$NG" --gen-len 3000 2>/dev/null | grep -E "^    [0-9]|amortiz"
  echo "== 63.V V-G3c all-weights-cold emulation (dense; mechanism PASS, claim scoped-out at 8.3M) =="
  bin/engine.exe --threads 1 --weights "$E1" --g3c --ngram "$NG" --emu-mb 128 2>/dev/null | grep -E "calibration|ratio|penalty|^    [0-9]|CLAIM|verdict"
else
  echo "== n-gram asset absent ($NG) -> skipping 63.V/63.C tables (build it: python benchmarks/phase63/ngram_asset.py --N 5) =="
fi

echo "== 63.C t1 attribution: OMP if-clause build vs OMP-compiled-out (the ~6% N=1 tax; accepted per portability law) =="
$CC $BASE $TUNE benchmarks/phase60/engine.c -o bin/engine_noomp.exe -lm 2>/dev/null
t1(){ OMP_NUM_THREADS=1 bin/"$1".exe --threads 1 --weights "$2" $3 --timing --time-tok ${TT:-5000} 2>/dev/null | grep -oE "[0-9]+\.[0-9]+ tok/s" | head -1; }
printf "    %-14s dense=%-12s MoE=%s\n" "omp(if-clause)" "$(t1 engine "$E1" '--mlp lut --skip on --exp fast')" "$(t1 engine "$E4" '--mlp lut --exp fast')"
printf "    %-14s dense=%-12s MoE=%s\n" "no-omp"        "$(t1 engine_noomp "$E1" '--mlp lut --skip on --exp fast')" "$(t1 engine_noomp "$E4" '--mlp lut --exp fast')"
echo "    (no-omp reproduces the pre-63.T historical singles; the ~6% delta is the pragma-outlining tax, not the build; znver2 is not faster.)"

echo "STOP. 63.T + 63.V/63.C consolidated bench above. No commit."
