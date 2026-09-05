#!/usr/bin/env bash
# SUPERSEDED by bench_matrix.py. Kept because its first and only run is the evidence for
# why it was replaced: blocked reps (all three of a config back to back, medians compared
# ACROSS blocks) gave within-config spreads of 3-5 tok/s on effects of 1-5%, and put
# --lut at 0.955x where an idle measurement had it at 1.06x. Recorded here, not deleted:
#
#   config                       policy    rep1   rep2   rep3   median  vs base
#   packed (baseline)            default  47.49  49.06  49.29   49.06   1.000x
#   --fuse                       default  49.28  49.88  49.50   49.50   1.009x
#   --lut --lut-group 32         default  50.01  44.88  46.87   46.87   0.955x
#   --fuse --lut --lut-group 32  default  46.12  48.69  48.66   48.66   0.992x
#   packed (baseline)            active   47.85  47.08  46.91   47.08   0.960x
#   --fuse                       active   49.19  47.86  48.65   48.65   0.992x
#   --lut --lut-group 32         active   48.26  48.24  47.37   48.24   0.983x
#   --fuse --lut --lut-group 32  active   49.23  48.04  47.68   48.04   0.979x
#
# Every one of those ratios is inside the noise of the row above it. Use bench_matrix.py:
# rounds instead of blocks, rotated order, --profile, and the baseline's own spread printed
# first as the witness for whether anything may be read at all.
#
# bench_matrix.sh -- WHY is this runtime at 12.4 GB/s when DRAM is 40-44 and the instruction
# issue bound is 16x away? Two hypotheses, crossed, three reps each.
#
#   H1  OpenMP REGION COUNT. A token opens 169 parallel regions. qkv delivers 4.1 GB/s against
#       the head's 17.6 on the same kernel; the excess is 2.32 ms over 72 calls = 32 us/call.
#       --fuse cuts 169 -> 97 and removes the two 128-row matvecs. Bit-identical, verified.
#
#   H2  OpenMP WAIT POLICY. 32 us is far too much for a barrier (1-5 us). If the regions are
#       short enough that idle threads SLEEP between them, each region pays a wake-up.
#       OMP_WAIT_POLICY=active keeps them spinning. This costs one environment variable and
#       would explain the size of the gap where a plain barrier cannot.
#
# The two are independent: if H2 alone closes the gap, --fuse is redundant; if H1 alone does,
# the wait policy is a red herring; if both matter they should be roughly additive in ms.
#
# Run ONLY on an idle machine. SPEED_LEDGER.md s11 exists because a contended timing is not a
# timing: the figure it withdrew was off by 35%.
#
# Usage: ./bench_matrix.sh [weights.bin] [n_tokens]
set -u
W="${1:-D:/_ktmp/qwen05b_packed.bin}"
N="${2:-300}"
E=./donor_engine.exe
REPS=3

printf '%-28s %-10s' "config" "policy"
for r in $(seq 1 $REPS); do printf ' %8s' "rep$r"; done
printf ' %9s %8s\n' "median" "vs base"

BASE=""
for POL in default active; do
  if [ "$POL" = "active" ]; then export OMP_WAIT_POLICY=active; else unset OMP_WAIT_POLICY; fi
  while IFS='|' read -r NAME FLAGS; do
    [ -z "$NAME" ] && continue
    printf '%-28s %-10s' "$NAME" "$POL"
    VALS=()
    for r in $(seq 1 $REPS); do
      T=$($E --weights "$W" --threads 6 $FLAGS --bench "$N" 2>/dev/null | grep -o '[0-9.]* tok/s' | cut -d' ' -f1)
      VALS+=("$T"); printf ' %8s' "$T"
    done
    MED=$(printf '%s\n' "${VALS[@]}" | sort -g | sed -n '2p')
    [ -z "$BASE" ] && BASE="$MED"
    printf ' %9s %7sx\n' "$MED" "$(python -c "print('%.3f'%($MED/$BASE))")"
  done <<'CFG'
packed (baseline)|
--fuse|--fuse
--lut --lut-group 32|--lut --lut-group 32
--fuse --lut --lut-group 32|--fuse --lut --lut-group 32
CFG
done
unset OMP_WAIT_POLICY
echo
echo "Reference, same binary, uncontended, from SPEED_LEDGER.md s11:"
echo "  packed 48.24 / 48.28 / 48.33 tok/s, matvec 19.69 ms, 25.1 G-weights/s"
echo "  per-organ GB/s at 4 bits: qkv 4.1 | o_proj 14.3 | ffn 12.7 | head 17.6 | aggregate 12.4"
echo "  ceiling if every organ hit the head's 17.6 GB/s: 65.7 tok/s (1.37x)"
