#!/usr/bin/env bash
# Stage 2 part 1, test 2: the registered order, one run at a time.  For each organ: train (GPU), export (CPU), panels
# (CPU), then the q4 file moves to E: to free the fast disk.  Resumable by hand: a step whose record exists is skipped.
# Usage (Git Bash, from the repo root): bash benchmarks/stage2/s2o_chain.sh
set -euo pipefail
R=benchmarks/stage2/results
FAST=C:/SiliconLLM_fast/s2o
KEEP=E:/SiliconLLM_Artifacts/stage2/s2o/models
LOG=E:/SiliconLLM_Artifacts/stage2/s2o/logs
WORK=E:/SiliconLLM_Artifacts/stage1/s1a
mkdir -p "$FAST" "$KEEP" "$LOG"

sha() { python -I -c "import json,sys; print(json.load(open(sys.argv[1]))['q4_sha256'])" "$1"; }

panels() {   # arm
  local q="$FAST/$1-q4.gguf"
  [ -f "$q" ] || return 0          # already run and moved
  python -I benchmarks/stage2/s2o_run.py --work "$WORK" --arm "$1" --model "$q" --model-sha "$(sha "$R/s2o_export_$1.json")" \
    --out "$R/s2o_run.json" >> "$LOG/run.out.log" 2>> "$LOG/run.err.log"
  mv "$q" "$KEEP/"
}

export_arm() {   # organ arm [tensor to shuffle]
  [ -f "$R/s2o_export_$2.json" ] && return 0
  python -I benchmarks/stage2/s2o_export.py --organ "$1" --trained "$FAST/$1" --out-q4 "$FAST/$2-q4.gguf" \
    --json "$R/s2o_export_$2.json" ${3:+--shuffle-rows "$3"} > "$LOG/export_$2.out.log" 2> "$LOG/export_$2.err.log"
}

for organ in dense attn shexp exps head; do
  if [ ! -f "$R/s2o_train_$organ.json" ]; then
    python -I benchmarks/stage2/s2o_train.py --organ "$organ" --out "$FAST/$organ" --json "$R/s2o_train_$organ.json" \
      > "$LOG/train_$organ.out.log" 2> "$LOG/train_$organ.err.log"
  fi
  export_arm "$organ" "org_$organ"
  panels "org_$organ"
  if [ "$organ" = attn ]; then
    export_arm attn org_attn_shuf attention.dense.weight
    panels org_attn_shuf
  fi
done
echo "chain done"
