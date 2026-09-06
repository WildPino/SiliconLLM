#!/usr/bin/env bash
# E4 arms -- BRIEF_E4_ATTENTION_ACCUMULATORS.md.  Strictly sequential: two timers must never
# overlap, and the parity runs (which are CPU-heavy) finish BEFORE the first timer starts.
set -e
P=../results/e4/parity.txt
R=../results/e4/arms.json
rm -f "$P" "$R"

# ---- G2: parity on a REAL donor, the pinned 24x512 slice.  Deterministic, order-of-accumulation
# only.  Runs first so that nothing is competing with a timer later.
W=D:/_ktmp/e1/qwen25-05b_tqh.bin
I=D:/_ktmp/e1/ids_qwen25-05b_tqh.bin
for A in serial serial2 ilp4 avx1 avx4; do
  echo "== parity $A ==" >> "$P"
  ./donor_engine.exe --weights "$W" --bpb "$I" --threads 6 --attn $A 2>/dev/null >> "$P"
done
echo "PARITY DONE"

# ---- G4 (baseline reproduces E3), G1 (planted positive), and the arms.
for B in 300 800; do
  for A in serial serial2 ilp4 avx1 avx4; do
    python e3_bench.py --weights D:/_ktmp/e3/T10_th.bin --bench $B --reps 3         --attn $A --label "T10_${A}_b${B}" --out "$R"
  done
done

# ---- the other head dimension (HD=64) and the shape the ledger was written on.
for B in 300 800; do
  for A in serial avx4; do
    python e3_bench.py --weights D:/_ktmp/e3/S05_th.bin --bench $B --reps 3         --attn $A --label "S05_${A}_b${B}" --out "$R"
  done
done
echo "ARMS DONE"
